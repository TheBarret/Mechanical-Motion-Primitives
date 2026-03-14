"""
MMP Chain Scope Analyser
================================
Oscilloscope-style diagnostic viewer for MMP CompositePrimitive chains.

X-axis : theta — the back-propagated input domain of the chain.
         Every channel shares the same x-axis.  Stage N receives the
         output of stage N-1 as its input; traces are plotted as a
         function of the root theta, not the local stage input.

Channels (one per primitive stage)
  solid  output(theta)      — DIM_COLOR of output_unit   (primary trace)
  dim    prev_output(theta) — DIM_COLOR of input_unit    (feed-through,
                               shows what this stage is receiving)
  dashed derivative(theta)  — right-axis scaled independently
                               (instantaneous transmission ratio)
  optional branch traces    — PeriodicBranchDependent only (toggle B)

Stage connectors  — colored band between channels showing the unit
                    hand-off: output_unit of N == input_unit of N+1.

Cursor  — shared vertical line; scrub with mouse drag or arrow keys.
Right panel  — per-stage numerical readout at cursor theta.

Controls
  <- / ->          move cursor (hold Shift for 10x)
  mouse drag       move cursor
  scroll wheel     zoom x-axis around cursor
  + / -            zoom x-axis
  D                toggle derivative traces
  B                toggle branch traces
  R                reset view to full domain
  Q / ESC          quit

All axis ranges, labels, and layout derive from the framework objects
(Domain, Dimension, is_invertible, available_branches).  Nothing is
hardcoded to a specific primitive or chain length.
"""

import pygame
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from core import (
    Dimension, Domain, Primitive, Invertible, Periodic,
    PeriodicBranchDependent, PeriodicBijective,
    MechanicalLimits, MMPError, CompositionError, DimensionMismatchError
)
from primitives import (
    # Class I
    SpurGear, CompoundGearTrain, RackAndPinion, Wedge, OldhamCoupling,
    # Class II
    ScotchYoke, EccentricCam, CrankSlider, HookesJoint,
    # Class III Adapters
    AngleToLength, LengthToAngle, UnitlessScaling, Bias, FunctionAdapter
)
from composite import CompositePrimitive, Governor
from builders import ICBuilder, CCBuilder

# ---------------------------------------------------------------------------
# Layout constants  (geometry only — no physics here)
# ---------------------------------------------------------------------------

SCREEN_W         = 1024
SCREEN_H         = 720
RIGHT_PANEL_W    = 290
HEADER_H         = 52
FOOTER_H         = 34
CH_PADDING       = 2      # px gap between channels
CH_HEADER_H      = 22     # strip at top of each channel for label
TRACE_PAD_TOP    = 6      # px inside plot area above top trace
TRACE_PAD_BOT    = 6      # px inside plot area below bottom trace
CONNECTOR_H      = 8      # px height of inter-channel connector band
N_SAMPLES        = 900    # theta resolution
N_GRID_LINES     = 10     # vertical grid divisions

# ---------------------------------------------------------------------------
# Color palette  (one source of truth)
# ---------------------------------------------------------------------------

C_BG             = ( 12,  13,  18)
C_GRID           = ( 28,  32,  40)
C_GRID_LABEL     = ( 55,  62,  76)
C_PANEL_BG       = ( 17,  19,  26)
C_PANEL_BORDER   = ( 40,  46,  60)
C_HEADER_BG      = ( 20,  22,  32)
C_FOOTER_BG      = ( 14,  16,  22)
C_CURSOR         = (215, 215, 170)
C_ZERO_LINE      = ( 44,  50,  62)
C_CLAMP          = (210,  70,  50)
C_WARN           = (210,  70,  50)
C_DERIV          = (160, 165, 175)   # derivative trace — neutral
C_READOUT_FG     = (170, 180, 195)
C_INV_OK         = ( 55, 175,  95)
C_INV_NO         = (175,  55,  55)

# Dimension colors — amber=rotational  sky=linear  green=ratio
DIM_COLOR: Dict[Dimension, Tuple[int,int,int]] = {
    Dimension.ANGLE:    (250, 185,  70),
    Dimension.LENGTH:   ( 70, 185, 250),
    Dimension.RATIO:    (130, 235, 110),
    Dimension.VELOCITY: (215,  95, 215),
    Dimension.GENERIC:  (150, 150, 150),
}

DIM_UNIT: Dict[Dimension, str] = {
    Dimension.ANGLE:    "rad",
    Dimension.LENGTH:   "m",
    Dimension.RATIO:    "",
    Dimension.VELOCITY: "rad/s",
    Dimension.GENERIC:  "?",
}

# Branch overlay colors (cycling)
BRANCH_COLORS = [(170, 110, 240), (110, 215, 170), (240, 170, 110)]

# ASCII-safe governing equations (no unicode math symbols)
EQUATIONS: Dict[str, str] = {
    "SpurGear":          "y = r*theta",
    "CompoundGearTrain": "y = prod(ri)*theta",
    "RackAndPinion":     "y = r*theta",
    "Wedge":             "y = x*tan(a)",
    "OldhamCoupling":    "y = x",
    "ScotchYoke":        "y = A*sin(theta+phi)",
    "EccentricCam":      "y = e*cos(t)+sqrt(r^2-e^2*sin^2(t))",
    "CrankSlider":       "y = r*cos(t)+sqrt(L^2-r^2*sin^2(t))",
    "HookesJoint":       "tan(out) = cos(a)*tan(in)",
    "Governor":          "y = clamp(f(x), lo, hi)",
    "AngleToLength":     "y = scale*x  (ANGLE->LENGTH)",
    "LengthToAngle":     "y = scale*x  (LENGTH->ANGLE)",
    "UnitlessScaling":   "y = scale*x",
    "Bias":              "y = x + bias",
}

# ---------------------------------------------------------------------------
# ScopeProbe  — non-invasive data container, never touches primitive state
# ---------------------------------------------------------------------------

@dataclass
class ScopeProbe:
    """
    Holds precomputed trace data for one primitive stage.
    All lists are indexed identically: index i corresponds to theta[i].

    input_trace      — stage input values as a function of root theta
    output_trace     — stage output values as a function of root theta
    derivative_trace — stage derivative as a function of root theta
    branch_traces    — inverse branch reconstructions (PeriodicBranchDependent)
    """
    primitive:        Primitive
    input_trace:      List[float] = field(default_factory=list)
    output_trace:     List[float] = field(default_factory=list)
    derivative_trace: List[float] = field(default_factory=list)
    branch_traces:    Dict[str, List[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ScopeChannel  — rendering unit for one primitive stage
# ---------------------------------------------------------------------------

@dataclass
class ScopeChannel:
    """
    Owns the screen Rect for one primitive stage and all pixel-space
    trace data.  scale_traces() must be called after probe data is
    populated and whenever the view rect changes.
    """
    primitive:    Primitive
    probe:        ScopeProbe
    rect:         pygame.Rect
    stage_index:  int

    # pixel-space polyline points (populated by scale_traces)
    px_output:     List[Tuple[int,int]] = field(default_factory=list)
    px_input:      List[Tuple[int,int]] = field(default_factory=list)
    px_derivative: List[Tuple[int,int]] = field(default_factory=list)
    px_branches:   Dict[str, List[Tuple[int,int]]] = field(default_factory=dict)

    # y-axis scale state (retained for right-panel readout)
    y_min_out: float = 0.0
    y_max_out: float = 1.0
    dy_min:    float = 0.0
    dy_max:    float = 1.0

    # ------------------------------------------------------------------
    def _plot_rect(self) -> pygame.Rect:
        """Inner plot area — excludes channel header strip."""
        r = self.rect
        return pygame.Rect(
            r.left,
            r.top + CH_HEADER_H,
            r.width,
            r.height - CH_HEADER_H,
        )

    def _safe_range(self, trace: List[float], pad_frac: float = 0.12
                    ) -> Tuple[float, float]:
        vals = [v for v in trace if not math.isnan(v) and math.isfinite(v)]
        if not vals:
            return -1.0, 1.0
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * pad_frac if hi != lo else 0.5
        return lo - pad, hi + pad

    def _make_px(self, trace: List[float],
                 y_min: float, y_max: float,
                 pr: pygame.Rect) -> List[Tuple[int,int]]:
        """Map a value trace to pixel coords within plot rect pr."""
        n    = len(trace)
        span = y_max - y_min if y_max != y_min else 1.0
        pts  = []
        for i, v in enumerate(trace):
            if not math.isfinite(v):
                continue
            px = pr.left + int(i / max(1, n - 1) * pr.width)
            norm = (v - y_min) / span
            py = pr.bottom - TRACE_PAD_BOT - int(
                norm * (pr.height - TRACE_PAD_TOP - TRACE_PAD_BOT))
            py = max(pr.top + TRACE_PAD_TOP,
                     min(pr.bottom - TRACE_PAD_BOT, py))
            pts.append((px, py))
        return pts

    def scale_traces(self):
        """Recompute all pixel-space polylines from probe data."""
        pr = self._plot_rect()
        probe = self.probe

        self.y_min_out, self.y_max_out = self._safe_range(probe.output_trace)
        in_min,  in_max  = self._safe_range(probe.input_trace)
        self.dy_min, self.dy_max = self._safe_range(probe.derivative_trace)

        self.px_output     = self._make_px(probe.output_trace,
                                           self.y_min_out, self.y_max_out, pr)
        self.px_input      = self._make_px(probe.input_trace,
                                           in_min, in_max, pr)
        self.px_derivative = self._make_px(probe.derivative_trace,
                                           self.dy_min, self.dy_max, pr)
        self.px_branches   = {
            b: self._make_px(t, self.y_min_out, self.y_max_out, pr)
            for b, t in probe.branch_traces.items()
        }

    # ------------------------------------------------------------------
    def _protocol_tags(self) -> str:
        tags = []
        p = self.primitive
        if isinstance(p, PeriodicBijective):    tags.append("PeBij")
        elif isinstance(p, PeriodicBranchDependent): tags.append("PeBrD")
        elif isinstance(p, Periodic):           tags.append("Per")
        if isinstance(p, Invertible):           tags.append("Inv")
        if isinstance(p, Governor):             tags.append("Gov")
        return "  ".join(tags)

    def _equation(self) -> str:
        return EQUATIONS.get(self.primitive.__class__.__name__, "")

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_lines(surface, pts, color, width=1, dashed=False):
        """Draw a polyline; optionally dashed (every other segment)."""
        if len(pts) < 2:
            return
        if not dashed:
            pygame.draw.lines(surface, color, False, pts, width)
        else:
            for i in range(0, len(pts) - 1, 2):
                pygame.draw.line(surface, color, pts[i], pts[i + 1], width)

    # ------------------------------------------------------------------
    def draw(self, surface: pygame.Surface, fonts: dict,
             cursor_idx: int, show_deriv: bool, show_branches: bool):
        r  = self.rect
        pr = self._plot_rect()

        in_col  = DIM_COLOR[self.primitive.domain.input_unit]
        out_col = DIM_COLOR[self.primitive.domain.output_unit]
        in_dim  = DIM_UNIT[self.primitive.domain.input_unit]
        out_dim = DIM_UNIT[self.primitive.domain.output_unit]

        # ── channel background ────────────────────────────────────────
        pygame.draw.rect(surface, C_PANEL_BG, r)

        # ── header strip ─────────────────────────────────────────────
        header_r = pygame.Rect(r.left, r.top, r.width, CH_HEADER_H)
        pygame.draw.rect(surface, C_HEADER_BG, header_r)

        # stage badge
        badge_surf = fonts['small'].render(f" {self.stage_index} ",
                                           True, C_BG, out_col)
        surface.blit(badge_surf, (r.left + 4, r.top + 3))

        # name
        name_surf = fonts['mono'].render(
            self.primitive.__class__.__name__, True, (205, 210, 220))
        surface.blit(name_surf, (r.left + 34, r.top + 4))

        # equation
        eq = self._equation()
        if eq:
            eq_surf = fonts['tiny'].render(eq, True, (90, 100, 118))
            surface.blit(eq_surf, (r.left + 195, r.top + 6))

        # protocol tags — right side
        tags = self._protocol_tags()
        if tags:
            tag_surf = fonts['tiny'].render(tags, True, (70, 150, 110))
            surface.blit(tag_surf,
                         (r.right - tag_surf.get_width() - 6, r.top + 6))

        # ── zero line (output axis) ───────────────────────────────────
        span_out = self.y_max_out - self.y_min_out
        if span_out > 0 and self.y_min_out <= 0 <= self.y_max_out:
            zy = pr.bottom - TRACE_PAD_BOT - int(
                (0 - self.y_min_out) / span_out
                * (pr.height - TRACE_PAD_TOP - TRACE_PAD_BOT))
            pygame.draw.line(surface, C_ZERO_LINE,
                             (pr.left, zy), (pr.right, zy), 1)

        # ── Governor clamp markers ────────────────────────────────────
        if isinstance(self.primitive, Governor) and span_out > 0:
            for cv, label in [(self.primitive.max_val, "max"),
                              (self.primitive.min_val, "min")]:
                if self.y_min_out <= cv <= self.y_max_out:
                    cy = pr.bottom - TRACE_PAD_BOT - int(
                        (cv - self.y_min_out) / span_out
                        * (pr.height - TRACE_PAD_TOP - TRACE_PAD_BOT))
                    pygame.draw.line(surface, C_CLAMP,
                                     (pr.left, cy), (pr.right, cy), 1)
                    cl = fonts['tiny'].render(
                        f"{label} {cv:.3f}", True, C_CLAMP)
                    surface.blit(cl, (pr.right - cl.get_width() - 4, cy - 11))

        # ── traces ───────────────────────────────────────────────────
        # input feed-through (dimmed, dashed) — what arrives at this stage
        dim_in = tuple(max(0, c - 90) for c in in_col)
        self._draw_lines(surface, self.px_input,  dim_in,   1, dashed=True)

        # output (primary, solid)
        self._draw_lines(surface, self.px_output, out_col,  2, dashed=False)

        # derivative (dashed, independently scaled, dim neutral)
        if show_deriv:
            self._draw_lines(surface, self.px_derivative,
                             C_DERIV, 1, dashed=True)
            # right-axis derivative scale ticks
            if self.px_derivative:
                dy_span = self.dy_max - self.dy_min
                if dy_span > 0:
                    for tick_val in [self.dy_min, 0.0, self.dy_max]:
                        if not (self.dy_min <= tick_val <= self.dy_max):
                            continue
                        ty = pr.bottom - TRACE_PAD_BOT - int(
                            (tick_val - self.dy_min) / dy_span
                            * (pr.height - TRACE_PAD_TOP - TRACE_PAD_BOT))
                        pygame.draw.line(surface, C_DERIV,
                                         (pr.right - 6, ty), (pr.right, ty), 1)
                        lbl = fonts['tiny'].render(
                            f"{tick_val:.2f}", True, C_DERIV)
                        surface.blit(lbl,
                                     (pr.right - lbl.get_width() - 8, ty - 9))

        # branch traces (PeriodicBranchDependent toggle)
        if show_branches:
            for bi, (bname, bpts) in enumerate(self.px_branches.items()):
                bc = BRANCH_COLORS[bi % len(BRANCH_COLORS)]
                self._draw_lines(surface, bpts, bc, 1, dashed=False)

        # ── y-axis scale labels (output) ──────────────────────────────
        for val, anchor_y in [(self.y_max_out, pr.top + TRACE_PAD_TOP),
                               (self.y_min_out, pr.bottom - TRACE_PAD_BOT - 11)]:
            lbl = fonts['tiny'].render(
                f"{val:.3f} {out_dim}", True,
                tuple(max(0, c - 40) for c in out_col))
            surface.blit(lbl, (r.left + 4, anchor_y))

        # ── cursor ────────────────────────────────────────────────────
        if self.px_output and 0 <= cursor_idx < len(self.px_output):
            cx, cy_out = self.px_output[cursor_idx]
            # vertical cursor line for this channel
            pygame.draw.line(surface, C_CURSOR,
                             (cx, pr.top), (cx, pr.bottom), 1)
            # dot on output trace
            pygame.draw.circle(surface, out_col, (cx, cy_out), 4)
            pygame.draw.circle(surface, C_BG,    (cx, cy_out), 2)
            # dot on input trace
            if cursor_idx < len(self.px_input):
                _, cy_in = self.px_input[cursor_idx]
                pygame.draw.circle(surface, in_col, (cx, cy_in), 3)

        # ── border ───────────────────────────────────────────────────
        pygame.draw.rect(surface, C_PANEL_BORDER, r, 1)


# ---------------------------------------------------------------------------
# MMPScope  — application
# ---------------------------------------------------------------------------

class MMPScope:
    """
    Multi-channel oscilloscope for a CompositePrimitive chain.
    One ScopeChannel per primitive stage.  Shared x-axis = theta.
    """

    def __init__(self, chain: CompositePrimitive):
        self.chain = chain

        pygame.init()
        pygame.display.set_caption("Chain Scope")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock  = pygame.time.Clock()

        self.fonts = {
            'header': pygame.font.SysFont('monospace', 15, bold=True),
            'mono':   pygame.font.SysFont('monospace', 13),
            'small':  pygame.font.SysFont('monospace', 12),
            'tiny':   pygame.font.SysFont('monospace', 10),
        }

        # ── theta domain from chain ────────────────────────────────────
        dom = chain.input_domain
        self.theta_full_min = dom.min if dom.min is not None else -2 * math.pi
        self.theta_full_max = dom.max if dom.max is not None else  2 * math.pi
        
        self.view_min = self.theta_full_min
        self.view_max = self.theta_full_max

        # cursor: index into [0, N_SAMPLES)
        self.cursor_idx = N_SAMPLES // 2

        # display toggles
        self.show_deriv    = True
        self.show_branches = False

        # ── probes and channels ────────────────────────────────────────
        self.probes: List[ScopeProbe] = [
            ScopeProbe(p) for p in chain.primitives
        ]
        self.channels: List[ScopeChannel] = []
        self._build_layout()
        self._recompute()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _scope_w(self) -> int:
        return SCREEN_W - RIGHT_PANEL_W - 1

    def _build_layout(self):
        n = len(self.chain.primitives)
        sw = self._scope_w()
        # vertical space: header + (channels + connectors between) + footer
        n_connectors = n - 1
        usable_h = (SCREEN_H - HEADER_H - FOOTER_H
                    - n_connectors * CONNECTOR_H
                    - (n - 1) * CH_PADDING)
        ch_h = usable_h // n

        self.channels = []
        y = HEADER_H
        for i, (p, probe) in enumerate(zip(self.chain.primitives, self.probes)):
            rect = pygame.Rect(0, y, sw, ch_h)
            self.channels.append(ScopeChannel(p, probe, rect, i))
            y += ch_h + CH_PADDING
            if i < n - 1:
                y += CONNECTOR_H

    # ------------------------------------------------------------------
    # Trace computation
    # ------------------------------------------------------------------

    def _thetas(self) -> List[float]:
        n = N_SAMPLES
        return [
            self.view_min + i / (n - 1) * (self.view_max - self.view_min)
            for i in range(n)
        ]

    def _recompute(self):
        """
        Propagate theta through the chain stage by stage.
        Each probe receives the true intermediate input values so that
        traces are correct functions of root theta.
        """
        thetas = self._thetas()
        n = len(thetas)

        # stage_inputs[i] = list of input values entering stage i
        stage_inputs: List[List[float]] = [thetas]
        for p in self.chain.primitives[:-1]:
            prev = stage_inputs[-1]
            nxt: List[float] = []
            for v in prev:
                try:
                    nxt.append(p.forward(v))
                except Exception:
                    nxt.append(float('nan'))
            stage_inputs.append(nxt)

        for i, (probe, ch) in enumerate(zip(self.probes, self.channels)):
            inputs = stage_inputs[i]
            probe.input_trace      = inputs
            probe.output_trace     = []
            probe.derivative_trace = []
            probe.branch_traces    = {}

            p = probe.primitive
            for v in inputs:
                try:
                    probe.output_trace.append(p.forward(v))
                    probe.derivative_trace.append(p.derivative(v))
                except Exception:
                    probe.output_trace.append(float('nan'))
                    probe.derivative_trace.append(float('nan'))

            if isinstance(p, PeriodicBranchDependent):
                for branch in p.available_branches:
                    btrace: List[float] = []
                    for v in inputs:
                        try:
                            y = p.forward(v)
                            btrace.append(p.inverse(y, branch=branch))
                        except Exception:
                            btrace.append(float('nan'))
                    probe.branch_traces[branch] = btrace

            ch.scale_traces()

    # ------------------------------------------------------------------
    # Cursor helpers
    # ------------------------------------------------------------------

    def _cursor_theta(self) -> float:
        thetas = self._thetas()
        idx = max(0, min(self.cursor_idx, len(thetas) - 1))
        return thetas[idx]

    def _theta_to_idx(self, theta: float) -> int:
        span = self.view_max - self.view_min
        if span == 0:
            return 0
        frac = (theta - self.view_min) / span
        return int(max(0, min(N_SAMPLES - 1, frac * (N_SAMPLES - 1))))

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom(self, factor: float):
        theta   = self._cursor_theta()
        half    = (self.view_max - self.view_min) * factor / 2
        new_min = max(self.theta_full_min, theta - half)
        new_max = min(self.theta_full_max, theta + half)
        if new_max > new_min + 1e-9:
            self.view_min  = new_min
            self.view_max  = new_max
            self.cursor_idx = self._theta_to_idx(theta)
            self._recompute()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_header(self):
        pygame.draw.rect(self.screen, C_HEADER_BG,
                         pygame.Rect(0, 0, SCREEN_W, HEADER_H))
        pygame.draw.line(self.screen, C_PANEL_BORDER,
                         (0, HEADER_H - 1), (SCREEN_W, HEADER_H - 1), 1)

        chain_str = " » ".join(p.__class__.__name__ for p in self.chain.primitives)
        title = self.fonts['header'].render(
            f"Chain Scope:  {chain_str}", True, (195, 200, 210))
        self.screen.blit(title, (8, 7))

        # Invertibility
        if self.chain.is_invertible:
            inv_col, inv_txt = C_INV_OK, "INVERTIBLE"
        else:
            inv_col, inv_txt = C_INV_NO, "NON-INVERTIBLE"
        inv_surf = self.fonts['small'].render(inv_txt, True, inv_col)
        self.screen.blit(inv_surf, (8, 30))

        # View domain
        in_unit = DIM_UNIT[self.chain.primitives[0].domain.input_unit]
        view_str = (f"theta  [{self.view_min:.3f}, {self.view_max:.3f}] {in_unit}"
                    f"   (full [{self.theta_full_min:.3f}, {self.theta_full_max:.3f}])")
        vs = self.fonts['small'].render(view_str, True, (90, 100, 120))
        self.screen.blit(vs, (200, 32))

        # Period
        period = self.chain.period()
        if period:
            ps = self.fonts['small'].render(
                f"T = {period:.4f} rad", True, (80, 155, 100))
            self.screen.blit(ps, (680, 32))

    def _draw_footer(self):
        r = pygame.Rect(0, SCREEN_H - FOOTER_H, SCREEN_W, FOOTER_H)
        pygame.draw.rect(self.screen, C_FOOTER_BG, r)
        pygame.draw.line(self.screen, C_PANEL_BORDER,
                         (0, r.top), (SCREEN_W, r.top), 1)

        ctrl = ("[ <-/-> cursor | Shift  x10 | scroll zoom | +/-  zoom | D deriv | B  branches | R reset | Q quit ]")
        cs = self.fonts['tiny'].render(ctrl, True, (72, 80, 96))
        self.screen.blit(cs, (8, r.top + 11))

        # legend
        legend_items = [
            ("output(theta)",      (180, 185, 195)),
            ("-- input(theta)",    (90,  95, 105)),
            ("-- deriv(theta)",    C_DERIV),
        ]
        lx = SCREEN_W - RIGHT_PANEL_W - 380
        for lbl, col in legend_items:
            ls = self.fonts['tiny'].render(lbl, True, col)
            self.screen.blit(ls, (lx, r.top + 7))
            lx += ls.get_width() + 22

    def _draw_x_axis(self):
        """Tick marks and labels below last channel, aligned to grid lines."""
        last_ch = self.channels[-1]
        sw      = self._scope_w()
        y_base  = last_ch.rect.bottom + CH_PADDING + CONNECTOR_H + 2

        in_unit = DIM_UNIT[self.chain.primitives[0].domain.input_unit]

        for i in range(N_GRID_LINES + 1):
            frac  = i / N_GRID_LINES
            theta = self.view_min + frac * (self.view_max - self.view_min)
            gx    = int(frac * sw)

            pygame.draw.line(self.screen, C_GRID_LABEL,
                             (gx, y_base), (gx, y_base + 4), 1)
            lbl = self.fonts['tiny'].render(f"{theta:.3f}", True, C_GRID_LABEL)
            self.screen.blit(lbl, (gx - lbl.get_width() // 2, y_base + 5))

        ax_lbl = self.fonts['tiny'].render(
            f"theta  ({in_unit})", True, (75, 85, 100))
        self.screen.blit(ax_lbl, (sw // 2 - ax_lbl.get_width() // 2,
                                   y_base + 18))

    def _draw_stage_connectors(self):
        """
        Colored band between consecutive channels showing the unit
        hand-off.  Color = DIM_COLOR of the shared dimension.
        A mismatched pair (should not exist post-validation) draws red.
        """
        for i in range(len(self.channels) - 1):
            ch_a = self.channels[i]
            ch_b = self.channels[i + 1]
            out_dim = ch_a.primitive.domain.output_unit
            in_dim  = ch_b.primitive.domain.input_unit

            y_top  = ch_a.rect.bottom + CH_PADDING
            y_bot  = ch_b.rect.top
            if y_bot <= y_top:
                continue

            band_rect = pygame.Rect(0, y_top, self._scope_w(), y_bot - y_top)
            color = DIM_COLOR.get(out_dim, C_PANEL_BORDER)
            mismatch = (out_dim != in_dim)

            # dim fill
            fill_col = tuple(max(0, c - 190) for c in color)
            pygame.draw.rect(self.screen, fill_col, band_rect)

            # top / bottom border lines in dimension color
            line_col = tuple(max(0, c - 100) for c in color)
            pygame.draw.line(self.screen, line_col,
                             (0, y_top), (self._scope_w(), y_top), 1)
            pygame.draw.line(self.screen, line_col,
                             (0, y_bot - 1), (self._scope_w(), y_bot - 1), 1)

            # unit label centered in band
            #unit_str = f"{out_dim.name} -> {in_dim.name}"
            unit_str = f"{DIM_UNIT[out_dim]} -> {DIM_UNIT[in_dim]}"
            unit_col = color if not mismatch else C_WARN
            
            us = self.fonts['tiny'].render(unit_str, True, unit_col)
            mx = (self._scope_w() - us.get_width()) // 2
            my = y_top + (y_bot - y_top - us.get_height()) // 2
            self.screen.blit(us, (mx, my))

            # cursor continuation line through connector band
            if self.channels[0].px_output:
                cx = self.channels[0].px_output[
                    min(self.cursor_idx,
                        len(self.channels[0].px_output) - 1)][0]
                pygame.draw.line(self.screen, C_CURSOR,
                                 (cx, y_top), (cx, y_bot - 1), 1)

    def _draw_right_panel(self):
        x = self._scope_w() + 1
        panel_rect = pygame.Rect(x, 0, RIGHT_PANEL_W, SCREEN_H)
        pygame.draw.rect(self.screen, C_PANEL_BG, panel_rect)
        pygame.draw.line(self.screen, C_PANEL_BORDER,
                         (x, 0), (x, SCREEN_H), 1)

        theta = self._cursor_theta()
        y     = 10

        def sep():
            nonlocal y
            pygame.draw.line(self.screen, (30, 34, 46),
                             (x + 6, y + 2), (x + RIGHT_PANEL_W - 6, y + 2), 1)
            y += 6

        def write(text, color=C_READOUT_FG, bold=False, indent=0):
            nonlocal y
            font = self.fonts['header'] if bold else self.fonts['small']
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x + 8 + indent, y))
            y += surf.get_height() + 2
            # guard — stop writing past footer
            return y < SCREEN_H - FOOTER_H - 8

        write("CURSOR READOUT", (145, 155, 175), bold=True)

        in_unit = DIM_UNIT[self.chain.primitives[0].domain.input_unit]
        write(f"theta = {theta:+.5f} {in_unit}", C_CURSOR)
        y += 4
        sep()

        # Per-stage
        val = theta
        for i, (p, probe) in enumerate(zip(self.chain.primitives, self.probes)):
            in_col  = DIM_COLOR[p.domain.input_unit]
            out_col = DIM_COLOR[p.domain.output_unit]
            iu      = DIM_UNIT[p.domain.input_unit]
            ou      = DIM_UNIT[p.domain.output_unit]

            if not write(f"[{i}] {p.__class__.__name__}",
                         (175, 180, 195), bold=False):
                break

            try:
                out   = p.forward(val)
                deriv = p.derivative(val)

                write(f"in  {val:+.5f} {iu}",  in_col,  indent=6)
                write(f"out {out:+.5f} {ou}",  out_col, indent=6)
                write(f"d/d {deriv:+.5f} {ou}/{iu}",
                      C_DERIV, indent=6)

                if not p.domain.contains(val):
                    write("! outside domain", C_WARN, indent=6)

                val = out

            except Exception as e:
                write(f"ERR: {str(e)[:26]}", C_WARN, indent=6)

            sep()

        # Chain totals
        write("CHAIN TOTAL", (145, 155, 175), bold=True)
        try:
            co  = self.chain.forward(theta)
            cd  = self.chain.derivative(theta)
            ou  = DIM_UNIT[self.chain.primitives[-1].domain.output_unit]
            iu  = DIM_UNIT[self.chain.primitives[0].domain.input_unit]
            write(f"out   {co:+.5f} {ou}",
                  DIM_COLOR[self.chain.primitives[-1].domain.output_unit])
            write(f"gain  {cd:+.5f} {ou}/{iu}", C_DERIV)
        except Exception as e:
            write(f"ERR: {str(e)[:28]}", C_WARN)

        # Invertibility note
        y += 4
        if self.chain.is_invertible:
            write("chain is invertible", C_INV_OK)
        else:
            #write("chain is non-invertible", C_INV_NO)
            #write("(Governor clamp is lossy)", C_INV_NO)
            for p in self.chain.primitives:
                if isinstance(p, Governor):
                    write("(Governor clamp is lossy)", C_INV_NO)
                elif not getattr(p, 'is_invertible', True):
                    write(f"({p.__class__.__name__} is one-way)", C_INV_NO)

    def _draw_grid(self):
        """Shared vertical grid lines spanning all channels."""
        sw = self._scope_w()
        y0 = HEADER_H
        y1 = SCREEN_H - FOOTER_H
        for i in range(N_GRID_LINES + 1):
            gx = int(i / N_GRID_LINES * sw)
            pygame.draw.line(self.screen, C_GRID, (gx, y0), (gx, y1), 1)

    def _draw_shared_cursor(self):
        """Full-height cursor line across all channels."""
        if not self.channels or not self.channels[0].px_output:
            return
        idx = min(self.cursor_idx, len(self.channels[0].px_output) - 1)
        cx = self.channels[0].px_output[idx][0]
        pygame.draw.line(self.screen, C_CURSOR,
                         (cx, HEADER_H), (cx, SCREEN_H - FOOTER_H), 1)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        running = True
        while running:
            self.clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    mods = pygame.key.get_mods()
                    step = 10 if (mods & pygame.KMOD_SHIFT) else 1

                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_RIGHT:
                        self.cursor_idx = min(N_SAMPLES - 1,
                                              self.cursor_idx + step)
                    elif event.key == pygame.K_LEFT:
                        self.cursor_idx = max(0, self.cursor_idx - step)
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS,
                                       pygame.K_KP_PLUS):
                        self._zoom(0.5)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self._zoom(2.0)
                    elif event.key == pygame.K_d:
                        self.show_deriv = not self.show_deriv
                    elif event.key == pygame.K_b:
                        self.show_branches = not self.show_branches
                    elif event.key == pygame.K_r:
                        self.view_min   = self.theta_full_min
                        self.view_max   = self.theta_full_max
                        self.cursor_idx = N_SAMPLES // 2
                        self._recompute()

                elif event.type == pygame.MOUSEMOTION:
                    if event.buttons[0]:
                        sw = self._scope_w()
                        mx = max(0, min(event.pos[0], sw))
                        self.cursor_idx = int(mx / sw * (N_SAMPLES - 1))

                elif event.type == pygame.MOUSEWHEEL:
                    self._zoom(0.82 if event.y > 0 else 1.22)

            # ── render ────────────────────────────────────────────────
            self.screen.fill(C_BG)
            self._draw_grid()
            self._draw_stage_connectors()

            for ch in self.channels:
                ch.draw(self.screen, self.fonts,
                        self.cursor_idx, self.show_deriv, self.show_branches)

            self._draw_shared_cursor()
            self._draw_header()
            self._draw_x_axis()
            self._draw_right_panel()
            self._draw_footer()

            pygame.display.flip()

        pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def scope(chain: CompositePrimitive):
    """Launch the MMP Chain Scope for any CompositePrimitive chain."""
    MMPScope(chain).run()


if __name__ == "__main__":
    #.add(OldhamCoupling)
    #.add(SpurGear,      ratio=1.2)
    #.add(HookesJoint,   shaft_angle=0.3)
    #.add(RackAndPinion, pitch_radius=0.1)
    #.add(Wedge, angle_rad=0.5)
    #.add(CompoundGearTrain, ratios=[2.0])
    #.add(CrankSlider, crank_length=1.0, rod_length=2.0, branch='principal') 
    #.add(ScotchYoke, amplitude=0.5, branch='principal')
    #.add(EccentricCam, eccentricity=1.0, follower_radius=2.0, branch='principal')
    
    #.add_governor(min_val=-5.0, max_val=5.0)
    
    # ANGLE -> ANGLE
    # - SpurGear
    # - CompoundGearTrain
    # - OldhamCoupling
    # - HookesJoint
    
    # ANGLE -> LENGTH
    # - RackAndPinion
    # - ScotchYoke
    # - EccentricCam
    # - CrankSlider
    
    # LENGTH -> LENGTH
    # - Wedge
    
    #EccentricCam → HookesJoint → OldhamCoupling → Wedge
    
    #wrist_actuator = (ICBuilder()
    #    .add(SpurGear, ratio=1.2)
    #    .add(HookesJoint, shaft_angle=0.2)
    #    .build())
    wrist_actuator = (CCBuilder()
        .add(SpurGear, ratio=2.5)                     # Current unit: ANGLE
        .add(RackAndPinion, pitch_radius=0.1)         # Current unit: LENGTH
        .add_adapter(to_unit=Dimension.ANGLE, scale=10.0)  # LENGTH → ANGLE
        .add(HookesJoint, shaft_angle=0.3)            # Current unit: ANGLE again
        .build())
        
    scope(wrist_actuator)
"""
MMP Visualizer
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Union
from mmpv3 import *

#============================================================================
#DIMENSION LABELS — Human-readable axis labels
#============================================================================

DIMENSION_LABELS = {
    Dimension.ANGLE:    ("Angle", "rad"),
    Dimension.LENGTH:   ("Displacement", "m"),
    Dimension.RATIO:    ("Ratio", ""),
    Dimension.VELOCITY: ("Velocity", "rad/s"),
    Dimension.GENERIC:  ("Value", ""),
}

def get_axis_label(dim: Dimension, is_input: bool = True) -> str:
    """Get formatted axis label with unit"""
    name, unit = DIMENSION_LABELS.get(dim, ("Value", ""))
    prefix = "Input" if is_input else "Output"
    if unit:
        return f"{prefix} {name} ({unit})"
    return f"{prefix} {name}"

#============================================================================
#PLOT CONFIGURATION
#============================================================================
@dataclass
class PlotConfig:
    """Unified plot styling"""
    figsize: Tuple[int, int] = (10, 6)
    dpi: int = 100
    line_width: float = 2.0
    grid_alpha: float = 0.3
    domain_color: str = '#FF6B6B'      # Red for domain bounds
    forward_color: str = '#4ECDC4'     # Teal for forward
    inverse_color: str = '#FFE66D'     # Yellow for inverse
    derivative_color: str = '#FF8C42'  # Orange for derivative
    governor_color: str = '#C44569'    # Pink for governor limits
    style: str = 'seaborn-v0_8-darkgrid'

#============================================================================
#PRIMITIVE VISUALIZER
#============================================================================
class PrimitiveVisualizer:
    """
    Plot individual primitives with dimension-aware axes.
    Shows forward, inverse (if available), derivative, and domain bounds.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        plt.style.use(self.config.style)
    
    def plot_forward(
        self,
        primitive: Primitive,
        x_range: Optional[Tuple[float, float]] = None,
        num_points: int = 1000,
        show_derivative: bool = False,
        show_domain: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot forward function with optional derivative and domain bounds"""
        
        # Determine x range
        if x_range is None:
            if hasattr(primitive, 'period'):
                # Periodic: show 2 periods
                period = primitive.period()
                x_range = (-period, 2 * period)
            elif primitive.domain.is_finite():
                # Finite domain: show full range with padding
                padding = 0.1 * (primitive.domain.max - primitive.domain.min)
                x_range = (primitive.domain.min - padding, 
                          primitive.domain.max + padding)
            else:
                # Default range for unbounded
                x_range = (-2 * math.pi, 2 * math.pi)
        
        # Generate points
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.array([primitive.forward(xi) for xi in x])
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot forward
        ax.plot(x, y, 
                color=self.config.forward_color, 
                linewidth=self.config.line_width,
                label='Forward')
        
        # Plot derivative if requested
        if show_derivative:
            dy = np.array([primitive.derivative(xi) for xi in x])
            ax_deriv = ax.twinx()
            ax_deriv.plot(x, dy, 
                         color=self.config.derivative_color,
                         linewidth=self.config.line_width,
                         linestyle='--',
                         label='Derivative')
            ax_deriv.set_ylabel('Derivative', color=self.config.derivative_color)
            ax_deriv.tick_params(axis='y', labelcolor=self.config.derivative_color)
        
        # Show domain bounds
        if show_domain and primitive.domain.is_finite():
            ax.axvline(primitive.domain.min, 
                      color=self.config.domain_color,
                      linestyle=':',
                      linewidth=2,
                      label='Domain Min')
            ax.axvline(primitive.domain.max,
                      color=self.config.domain_color,
                      linestyle=':',
                      linewidth=2,
                      label='Domain Max')
        
        # Governor bounds if applicable
        if isinstance(primitive, Governor):
            ax.axhline(primitive.min_val,
                      color=self.config.governor_color,
                      linestyle='-.',
                      linewidth=2,
                      label='Governor Min')
            ax.axhline(primitive.max_val,
                      color=self.config.governor_color,
                      linestyle='-.',
                      linewidth=2,
                      label='Governor Max')
        
        # Labels
        ax.set_xlabel(get_axis_label(primitive.domain.input_unit, is_input=True))
        ax.set_ylabel(get_axis_label(primitive.domain.output_unit, is_input=False))
        ax.set_title(title or f"{primitive.__class__.__name__} — Forward")
        ax.legend(loc='best')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_inverse(
        self,
        primitive: Invertible,
        y_range: Optional[Tuple[float, float]] = None,
        num_points: int = 1000,
        branch: Optional[str] = None,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot inverse function for invertible primitives"""
        
        if not primitive.is_invertible:
            raise ValueError("Primitive is not invertible")
        
        # Determine y range from domain
        if y_range is None:
            if primitive.domain.is_finite():
                padding = 0.1 * (primitive.domain.max - primitive.domain.min)
                y_range = (primitive.domain.min - padding,
                          primitive.domain.max + padding)
            else:
                y_range = (-2 * math.pi, 2 * math.pi)
        
        # Generate points
        y = np.linspace(y_range[0], y_range[1], num_points)
        x = []
        valid_y = []
        
        for yi in y:
            try:
                xi = primitive.inverse(yi, branch=branch)
                x.append(xi)
                valid_y.append(yi)
            except (ValueError, ZeroDivisionError):
                continue  # Skip out-of-domain points
        
        x = np.array(x)
        valid_y = np.array(valid_y)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot inverse
        ax.plot(x, valid_y,
                color=self.config.inverse_color,
                linewidth=self.config.line_width,
                label=f'Inverse{" (" + branch + ")" if branch else ""}')
        
        # Plot forward for comparison
        x_forward = np.linspace(min(x) - 0.5, max(x) + 0.5, num_points)
        y_forward = np.array([primitive.forward(xi) for xi in x_forward])
        ax.plot(x_forward, y_forward,
                color=self.config.forward_color,
                linewidth=self.config.line_width,
                linestyle='--',
                alpha=0.5,
                label='Forward (reference)')
        
        # Identity line for reference
        min_val = min(x.min(), valid_y.min())
        max_val = max(x.max(), valid_y.max())
        ax.plot([min_val, max_val], [min_val, max_val],
                color='gray',
                linestyle=':',
                alpha=0.3,
                label='y = x')
        
        # Labels
        ax.set_xlabel(get_axis_label(primitive.domain.output_unit, is_input=False))
        ax.set_ylabel(get_axis_label(primitive.domain.input_unit, is_input=True))
        ax.set_title(title or f"{primitive.__class__.__name__} — Inverse")
        ax.legend(loc='best')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_periodic(
        self,
        primitive: Periodic,
        num_periods: int = 2,
        num_points: int = 1000,
        show_normalize: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Specialized plot for periodic primitives"""
        
        period = primitive.period()
        x_range = (-period * 0.5, period * (num_periods + 0.5))
        x = np.linspace(x_range[0], x_range[1], num_points)
        y = np.array([primitive.forward(xi) for xi in x])
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Plot forward
        ax.plot(x, y,
                color=self.config.forward_color,
                linewidth=self.config.line_width,
                label='Forward')
        
        # Show period markers
        for i in range(-1, num_periods + 2):
            ax.axvline(i * period,
                      color=self.config.domain_color,
                      linestyle=':',
                      linewidth=1,
                      alpha=0.5)
        
        # Show normalized comparison if requested
        if show_normalize:
            x_norm = np.linspace(0, period, num_points)
            y_norm = np.array([primitive.forward(primitive.normalize(xi)) 
                              for xi in x_norm])
            ax.plot(x_norm, y_norm,
                    color=self.config.inverse_color,
                    linewidth=self.config.line_width,
                    linestyle='--',
                    alpha=0.7,
                    label='Normalized (one period)')
        
        # Labels
        ax.set_xlabel(get_axis_label(primitive.domain.input_unit, is_input=True))
        ax.set_ylabel(get_axis_label(primitive.domain.output_unit, is_input=False))
        ax.set_title(title or f"{primitive.__class__.__name__} — Periodic (T={period:.2f})")
        ax.legend(loc='best')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        # Mark period on x-axis
        ax.set_xticks([i * period for i in range(-1, num_periods + 2)])
        ax.set_xticklabels([f'{i}T' for i in range(-1, num_periods + 2)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig

#============================================================================
#CHAIN VISUALIZER
#============================================================================
class ChainVisualizer:
    """
    Plot CompositePrimitive chains — shows each stage and final output.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.primitive_viz = PrimitiveVisualizer(config)
        plt.style.use(self.config.style)
    
    def plot_chain_forward(
        self,
        chain: CompositePrimitive,
        x_range: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
        show_stages: bool = True,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot chain forward with optional intermediate stage visualization.
        """
        
        # Determine x range from chain's input domain
        if x_range is None:
            if chain.input_domain.is_finite():
                padding = 0.1 * (chain.input_domain.max - chain.input_domain.min)
                x_range = (chain.input_domain.min - padding,
                          chain.input_domain.max + padding)
            else:
                x_range = (-2 * math.pi, 2 * math.pi)
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        # Compute outputs at each stage
        stage_outputs = [x.copy()]  # Stage 0 = input
        current = x.copy()
        
        for p in chain.primitives:
            current = np.array([p.forward(xi) for xi in current])
            stage_outputs.append(current)
        
        # Create figure with subplots if showing stages
        if show_stages and len(chain.primitives) > 1:
            n_stages = len(chain.primitives)
            fig, axes = plt.subplots(1, n_stages,
                                    figsize=(5 * n_stages, 4),
                                    dpi=self.config.dpi,
                                    sharey=False)
            
            if n_stages == 1:
                axes = [axes]
            
            for i, ax in enumerate(axes):
                ax.plot(x, stage_outputs[i],
                       color=self.config.forward_color,
                       linewidth=self.config.line_width,
                       alpha=0.5,
                       label=f'Stage {i} Input')
                ax.plot(x, stage_outputs[i + 1],
                       color=self.config.forward_color,
                       linewidth=self.config.line_width,
                       label=f'Stage {i+1} Output')
                
                ax.set_xlabel(get_axis_label(
                    chain.primitives[i].domain.input_unit, is_input=True))
                ax.set_ylabel(get_axis_label(
                    chain.primitives[i].domain.output_unit, is_input=False))
                ax.set_title(f'{chain.primitives[i].__class__.__name__}')
                ax.legend(loc='best')
                ax.grid(True, alpha=self.config.grid_alpha)
            
            fig.suptitle(title or f"Chain: {' → '.join(p.__class__.__name__ for p in chain.primitives)}")
            plt.tight_layout()
        else:
            # Single plot — final output only
            fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
            
            ax.plot(x, stage_outputs[-1],
                   color=self.config.forward_color,
                   linewidth=self.config.line_width,
                   label='Final Output')
            
            ax.set_xlabel(get_axis_label(chain.domain.input_unit, is_input=True))
            ax.set_ylabel(get_axis_label(chain.domain.output_unit, is_input=False))
            ax.set_title(title or "Chain Forward")
            ax.legend(loc='best')
            ax.grid(True, alpha=self.config.grid_alpha)
            
            plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_chain_comparison(
        self,
        chains: List[Tuple[str, CompositePrimitive]],
        x_range: Optional[Tuple[float, float]] = None,
        num_points: int = 500,
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Compare multiple chains on same axes"""
        
        if x_range is None:
            x_range = (-2 * math.pi, 2 * math.pi)
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        fig, ax = plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(chains)))
        
        for i, (name, chain) in enumerate(chains):
            y = np.array([chain.forward(xi) for xi in x])
            ax.plot(x, y,
                   color=colors[i],
                   linewidth=self.config.line_width,
                   label=name)
        
        ax.set_xlabel(get_axis_label(chains[0][1].domain.input_unit, is_input=True))
        ax.set_ylabel(get_axis_label(chains[0][1].domain.output_unit, is_input=False))
        ax.set_title(title or "Chain Comparison")
        ax.legend(loc='best')
        ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        
        return fig

#============================================================================
#CONVENIENCE FUNCTIONS
#============================================================================
def visualize_primitive(
    primitive: Primitive,
    plot_type: str = 'forward',
    **kwargs
) -> plt.Figure:
    """Quick visualization with sensible defaults"""
    viz = PrimitiveVisualizer()
    
    if plot_type == 'forward':
        return viz.plot_forward(primitive, **kwargs)
    elif plot_type == 'inverse':
        if isinstance(primitive, Invertible):
            return viz.plot_inverse(primitive, **kwargs)
        else:
            raise ValueError("Primitive is not invertible")
    elif plot_type == 'periodic':
        if isinstance(primitive, Periodic):
            return viz.plot_periodic(primitive, **kwargs)
        else:
            raise ValueError("Primitive is not periodic")
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

def visualize_chain(
    chain: CompositePrimitive,
    show_stages: bool = True,
    **kwargs
) -> plt.Figure:
    """Quick chain visualization"""
    viz = ChainVisualizer()
    return viz.plot_chain_forward(chain, show_stages=show_stages, **kwargs)

if __name__ == "__main__":
    # Single primitive
    yoke = ScotchYoke(amplitude=2.0, phase=0.0)
    fig1 = visualize_primitive(yoke, plot_type='periodic', num_periods=3)
    #plt.show()
    plt.savefig('yoke_primitive.png')
    
    # Inverse comparison
    cam = EccentricCam(eccentricity=1.0, follower_radius=5.0, branch='principal')
    fig2 = visualize_primitive(cam, plot_type='inverse')
    #plt.show()
    plt.savefig('eccentriccam_primitive.png')
    
    # Chain visualization
    wrist = (ChainBuilder()
        .add(SpurGear, ratio=2.5)
        .add(HookesJoint, shaft_angle=0.05)
        .add(RackAndPinion, pitch_radius=0.1)
        .build())
    
    fig3 = visualize_chain(wrist, show_stages=True)
    #plt.show()
    plt.savefig('wrist_complex.png')
    
    # Derivative overlay
    gear = SpurGear(ratio=3.0)
    fig4 = visualize_primitive(gear, plot_type='forward', show_derivative=True)
    #plt.show()
    plt.savefig('spurgear_derivative.png')
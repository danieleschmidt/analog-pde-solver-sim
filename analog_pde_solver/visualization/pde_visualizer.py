"""PDE solution visualization tools."""

import numpy as np
from typing import Optional, List, Dict, Any, Union, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for PDE visualization plots."""
    figsize: Tuple[int, int] = (10, 8)
    dpi: int = 300
    colormap: str = "viridis"
    save_format: str = "png"
    show_grid: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10


class PDEVisualizer:
    """Visualization toolkit for PDE solutions and analysis."""
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize PDE visualizer.
        
        Args:
            config: Plot configuration options
        """
        self.config = config or PlotConfig()
        self.logger = logging.getLogger(__name__)
        
        # Check matplotlib availability
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            from matplotlib.colors import LinearSegmentedColormap
            self.plt = plt
            self.animation = animation
            self.LinearSegmentedColormap = LinearSegmentedColormap
            self._has_matplotlib = True
        except ImportError:
            self.logger.warning("Matplotlib not available, visualization disabled")
            self._has_matplotlib = False
    
    def plot_solution(
        self, 
        solution: np.ndarray, 
        ax=None, 
        title: str = "PDE Solution",
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None
    ):
        """Plot 1D or 2D PDE solution.
        
        Args:
            solution: Solution array (1D or 2D)
            ax: Matplotlib axes (optional)
            title: Plot title
            x_coords: X coordinates (optional)
            y_coords: Y coordinates (optional, for 2D)
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for plotting")
            return None
            
        if ax is None:
            fig, ax = self.plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        if solution.ndim == 1:
            # 1D plot
            x = x_coords if x_coords is not None else np.linspace(0, 1, len(solution))
            ax.plot(x, solution, linewidth=2)
            ax.set_xlabel("x", fontsize=self.config.label_fontsize)
            ax.set_ylabel("Solution", fontsize=self.config.label_fontsize)
            
        elif solution.ndim == 2:
            # 2D heatmap
            im = ax.imshow(
                solution, 
                cmap=self.config.colormap,
                aspect='auto',
                interpolation='bilinear'
            )
            ax.figure.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("x", fontsize=self.config.label_fontsize)
            ax.set_ylabel("y", fontsize=self.config.label_fontsize)
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_error_map(
        self, 
        error: np.ndarray, 
        ax=None, 
        title: str = "Error Map",
        log_scale: bool = True
    ):
        """Plot error distribution.
        
        Args:
            error: Error array
            ax: Matplotlib axes (optional)
            title: Plot title
            log_scale: Use logarithmic color scale
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for plotting")
            return None
        
        if ax is None:
            fig, ax = self.plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Take absolute value for log scale
        plot_data = np.abs(error) if log_scale else error
        
        if plot_data.ndim == 1:
            x = np.linspace(0, 1, len(plot_data))
            if log_scale:
                ax.semilogy(x, plot_data, linewidth=2, color='red')
            else:
                ax.plot(x, plot_data, linewidth=2, color='red')
            ax.set_xlabel("x", fontsize=self.config.label_fontsize)
            ax.set_ylabel("Error", fontsize=self.config.label_fontsize)
            
        elif plot_data.ndim == 2:
            if log_scale:
                plot_data = np.log10(plot_data + 1e-12)  # Avoid log(0)
            
            im = ax.imshow(
                plot_data, 
                cmap='Reds',
                aspect='auto',
                interpolation='bilinear'
            )
            cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
            if log_scale:
                cbar.set_label("log₁₀(Error)", fontsize=self.config.label_fontsize)
            else:
                cbar.set_label("Error", fontsize=self.config.label_fontsize)
            
            ax.set_xlabel("x", fontsize=self.config.label_fontsize)
            ax.set_ylabel("y", fontsize=self.config.label_fontsize)
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        return ax
    
    def animate_solution(
        self,
        solution_history: List[np.ndarray],
        title: str = "PDE Solution Evolution",
        save_path: Optional[str] = None,
        fps: int = 10,
        interval: int = 100
    ):
        """Create animation of solution evolution.
        
        Args:
            solution_history: List of solution arrays over time
            title: Animation title
            save_path: Path to save animation (optional)
            fps: Frames per second
            interval: Time between frames (ms)
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for animation")
            return None
        
        if not solution_history:
            self.logger.error("Empty solution history provided")
            return None
        
        fig, ax = self.plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Get data range for consistent scaling
        all_data = np.concatenate([sol.flatten() for sol in solution_history])
        vmin, vmax = all_data.min(), all_data.max()
        
        def update(frame):
            ax.clear()
            solution = solution_history[frame]
            
            if solution.ndim == 1:
                x = np.linspace(0, 1, len(solution))
                ax.plot(x, solution, linewidth=2)
                ax.set_ylim(vmin, vmax)
                ax.set_xlabel("x", fontsize=self.config.label_fontsize)
                ax.set_ylabel("Solution", fontsize=self.config.label_fontsize)
            
            elif solution.ndim == 2:
                im = ax.imshow(
                    solution,
                    cmap=self.config.colormap,
                    vmin=vmin,
                    vmax=vmax,
                    aspect='auto',
                    interpolation='bilinear'
                )
                ax.set_xlabel("x", fontsize=self.config.label_fontsize)
                ax.set_ylabel("y", fontsize=self.config.label_fontsize)
            
            ax.set_title(f"{title} (Step {frame+1}/{len(solution_history)})", 
                        fontsize=self.config.title_fontsize)
            ax.tick_params(labelsize=self.config.tick_fontsize)
            
            if self.config.show_grid:
                ax.grid(True, alpha=0.3)
        
        anim = self.animation.FuncAnimation(
            fig, update, frames=len(solution_history),
            interval=interval, blit=False, repeat=True
        )
        
        if save_path:
            try:
                anim.save(save_path, fps=fps, writer='pillow')
                self.logger.info(f"Animation saved to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save animation: {e}")
        
        return anim
    
    def plot_convergence(
        self,
        convergence_history: List[float],
        ax=None,
        title: str = "Convergence History",
        log_scale: bool = True
    ):
        """Plot convergence history.
        
        Args:
            convergence_history: List of error values over iterations
            ax: Matplotlib axes (optional)
            title: Plot title
            log_scale: Use logarithmic y-scale
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for plotting")
            return None
        
        if ax is None:
            fig, ax = self.plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        iterations = range(1, len(convergence_history) + 1)
        
        if log_scale:
            ax.semilogy(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
        else:
            ax.plot(iterations, convergence_history, 'b-', linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel("Iteration", fontsize=self.config.label_fontsize)
        ax.set_ylabel("Error", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3)
        
        return ax
    
    def compare_solutions(
        self,
        solutions: Dict[str, np.ndarray],
        title: str = "Solution Comparison",
        save_path: Optional[str] = None
    ):
        """Compare multiple solutions side by side.
        
        Args:
            solutions: Dictionary of solution name -> array
            title: Overall plot title
            save_path: Path to save comparison plot (optional)
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for plotting")
            return None
        
        n_solutions = len(solutions)
        if n_solutions == 0:
            self.logger.error("No solutions provided for comparison")
            return None
        
        # Create subplots
        fig, axes = self.plt.subplots(
            1, n_solutions, 
            figsize=(self.config.figsize[0] * n_solutions, self.config.figsize[1]),
            dpi=self.config.dpi
        )
        
        if n_solutions == 1:
            axes = [axes]
        
        for (name, solution), ax in zip(solutions.items(), axes):
            self.plot_solution(solution, ax=ax, title=name)
        
        fig.suptitle(title, fontsize=self.config.title_fontsize + 2)
        self.plt.tight_layout()
        
        if save_path:
            try:
                fig.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi, bbox_inches='tight')
                self.logger.info(f"Comparison plot saved to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save comparison plot: {e}")
        
        return fig
    
    def plot_hardware_utilization(
        self,
        utilization_data: Dict[str, Any],
        ax=None,
        title: str = "Hardware Utilization"
    ):
        """Plot hardware utilization metrics.
        
        Args:
            utilization_data: Dictionary with utilization metrics
            ax: Matplotlib axes (optional)  
            title: Plot title
        """
        if not self._has_matplotlib:
            self.logger.error("Matplotlib not available for plotting")
            return None
        
        if ax is None:
            fig, ax = self.plt.subplots(figsize=self.config.figsize, dpi=self.config.dpi)
        
        # Extract metrics for plotting
        metrics = []
        values = []
        
        for key, value in utilization_data.items():
            if isinstance(value, (int, float)):
                metrics.append(key.replace('_', ' ').title())
                values.append(value)
        
        if not metrics:
            self.logger.warning("No numeric metrics found in utilization data")
            return ax
        
        # Create horizontal bar chart
        y_pos = np.arange(len(metrics))
        bars = ax.barh(y_pos, values, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics, fontsize=self.config.label_fontsize)
        ax.set_xlabel("Value", fontsize=self.config.label_fontsize)
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.tick_params(labelsize=self.config.tick_fontsize)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.2f}', ha='left', va='center', fontsize=self.config.tick_fontsize)
        
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='x')
        
        return ax
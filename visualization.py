import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import pandas as pd
import numpy as np
import io
import base64
from typing import Dict, Any, Optional
import logging

class DataVisualizer:
    """Handles data visualization and chart generation."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_scatterplot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                          config: Dict[str, Any] = None) -> str:
        """Create a scatterplot with optional regression line."""
        try:
            config = config or {}
            
            # Clean and convert data to numeric
            x_data = pd.to_numeric(data[x_col], errors='coerce')
            y_data = pd.to_numeric(data[y_col], errors='coerce')
            
            # Create clean dataframe and drop NaN values
            clean_data = pd.DataFrame({x_col: x_data, y_col: y_data}).dropna()
            
            if len(clean_data) == 0:
                return self._create_error_plot("No valid numeric data for plotting")
            
            # Create figure with appropriate size
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatterplot
            ax.scatter(clean_data[x_col], clean_data[y_col], alpha=0.6, s=50, color='blue')
            
            # Add regression line if requested
            if config.get('regression_line', False):
                # Calculate regression line
                z = np.polyfit(clean_data[x_col], clean_data[y_col], 1)
                p = np.poly1d(z)
                
                # Plot regression line
                x_range = np.linspace(clean_data[x_col].min(), clean_data[x_col].max(), 100)
                y_pred = p(x_range)
                
                # Set line style and color
                color = config.get('regression_color', 'red')
                linestyle = ':' if config.get('regression_style') == 'dotted' else '-'
                
                ax.plot(x_range, y_pred, color=color, linestyle=linestyle, 
                       linewidth=2, label=f'Regression Line (y={z[0]:.3f}x+{z[1]:.3f})')
                ax.legend()
            
            # Set labels and title
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{y_col} vs {x_col}', fontsize=14)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Tight layout to prevent label cutoff
            plt.tight_layout()
            
            # Convert to base64 with size limit
            return self._fig_to_base64(fig, 'png', max_size_kb=config.get('size_limit_kb', 100))
            
        except Exception as e:
            self.logger.error(f"Error creating scatterplot: {str(e)}")
            return self._create_error_plot(f"Error creating scatterplot: {str(e)}")
    
    def create_histogram(self, data: pd.DataFrame, column: str, bins: int = 30) -> str:
        """Create a histogram."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Clean data
            clean_data = pd.to_numeric(data[column], errors='coerce').dropna()
            
            ax.hist(clean_data, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {column}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, 'png')
            
        except Exception as e:
            self.logger.error(f"Error creating histogram: {str(e)}")
            return self._create_error_plot(f"Error creating histogram: {str(e)}")
    
    def create_line_plot(self, data: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create a line plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Clean and sort data
            plot_data = data[[x_col, y_col]].copy()
            plot_data[x_col] = pd.to_numeric(plot_data[x_col], errors='coerce')
            plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
            plot_data = plot_data.dropna().sort_values(x_col)
            
            ax.plot(plot_data[x_col], plot_data[y_col], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{y_col} vs {x_col}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, 'png')
            
        except Exception as e:
            self.logger.error(f"Error creating line plot: {str(e)}")
            return self._create_error_plot(f"Error creating line plot: {str(e)}")
    
    def create_bar_plot(self, data: pd.DataFrame, x_col: str, y_col: str) -> str:
        """Create a bar plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Clean data
            plot_data = data[[x_col, y_col]].copy()
            plot_data[y_col] = pd.to_numeric(plot_data[y_col], errors='coerce')
            plot_data = plot_data.dropna()
            
            ax.bar(plot_data[x_col], plot_data[y_col])
            ax.set_xlabel(x_col, fontsize=12)
            ax.set_ylabel(y_col, fontsize=12)
            ax.set_title(f'{y_col} by {x_col}', fontsize=14)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(plot_data) > 10:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            return self._fig_to_base64(fig, 'png')
            
        except Exception as e:
            self.logger.error(f"Error creating bar plot: {str(e)}")
            return self._create_error_plot(f"Error creating bar plot: {str(e)}")
    
    def create_heatmap(self, data: pd.DataFrame, columns: list = None) -> str:
        """Create a correlation heatmap."""
        try:
            # Select numeric columns only
            numeric_data = data.select_dtypes(include=[np.number])
            
            if columns:
                numeric_data = numeric_data[columns]
            
            # Calculate correlation matrix
            corr_matrix = numeric_data.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', ax=ax)
            
            ax.set_title('Correlation Heatmap', fontsize=14)
            plt.tight_layout()
            
            return self._fig_to_base64(fig, 'png')
            
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {str(e)}")
            return self._create_error_plot(f"Error creating heatmap: {str(e)}")
    
    def _fig_to_base64(self, fig, format: str = 'png', max_size_kb: int = 100) -> str:
        """Convert matplotlib figure to base64 string with size optimization."""
        try:
            # Start with high quality
            dpi = 100
            quality = 95
            
            while dpi >= 50:  # Minimum acceptable DPI
                buffer = io.BytesIO()
                
                if format == 'png':
                    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                else:
                    fig.savefig(buffer, format=format, dpi=dpi, bbox_inches='tight',
                               quality=quality, facecolor='white', edgecolor='none')
                
                buffer.seek(0)
                img_bytes = buffer.getvalue()
                
                # Check size
                size_kb = len(img_bytes) / 1024
                
                if size_kb <= max_size_kb:
                    # Convert to base64
                    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                    
                    # Close figure to free memory
                    plt.close(fig)
                    
                    # Return as data URI
                    return f"data:image/{format};base64,{img_base64}"
                
                # Reduce quality for next iteration
                dpi -= 10
                if format != 'png':
                    quality -= 5
            
            # If we still can't get under the size limit, use the last attempt
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            plt.close(fig)
            
            self.logger.warning(f"Could not reduce image size below {max_size_kb}KB. Final size: {size_kb:.1f}KB")
            
            return f"data:image/{format};base64,{img_base64}"
            
        except Exception as e:
            self.logger.error(f"Error converting figure to base64: {str(e)}")
            plt.close(fig)
            raise
    
    def _create_error_plot(self, error_message: str) -> str:
        """Create an error plot with message."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Visualization Error:\n{error_message}", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12, color='red',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3))
            ax.set_title("Error in Visualization", fontsize=14)
            ax.axis('off')
            
            return self._fig_to_base64(fig, 'png')
            
        except Exception as e:
            self.logger.error(f"Error creating error plot: {str(e)}")
            # Return a minimal 1x1 transparent PNG as base64
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    def get_plot_size_kb(self, base64_string: str) -> float:
        """Get the size of base64 encoded plot in KB."""
        try:
            # Remove data URI prefix if present
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            # Calculate size in KB
            size_bytes = len(base64_string.encode('utf-8'))
            size_kb = size_bytes / 1024
            
            return size_kb
            
        except Exception as e:
            self.logger.error(f"Error calculating plot size: {str(e)}")
            return 0.0
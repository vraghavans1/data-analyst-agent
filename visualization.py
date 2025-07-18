"""
Lightweight visualization module for Vercel deployment
Creates simple charts without matplotlib using SVG and HTML
"""

import json
import base64
import io
from typing import Dict, Any, List, Optional, Tuple
import math

class DataVisualizer:
    """
    Lightweight data visualizer that creates charts using SVG.
    Optimized for Vercel deployment without matplotlib dependency.
    """
    
    def __init__(self):
        self.width = 800
        self.height = 600
        self.margin = 50
        self.colors = [
            '#2563eb', '#dc2626', '#16a34a', '#ca8a04', '#9333ea',
            '#c2410c', '#0891b2', '#be123c', '#4338ca', '#059669'
        ]
    
    def create_chart(self, data: Dict[str, Any], chart_type: str = 'bar') -> str:
        """
        Create a chart from data.
        
        Args:
            data: Dictionary containing chart data
            chart_type: Type of chart ('bar', 'line', 'pie', 'scatter')
            
        Returns:
            Base64 encoded SVG chart
        """
        try:
            if chart_type == 'bar':
                svg = self._create_bar_chart(data)
            elif chart_type == 'line':
                svg = self._create_line_chart(data)
            elif chart_type == 'pie':
                svg = self._create_pie_chart(data)
            elif chart_type == 'scatter':
                svg = self._create_scatter_chart(data)
            else:
                svg = self._create_bar_chart(data)  # Default to bar chart
            
            # Convert SVG to base64
            svg_bytes = svg.encode('utf-8')
            return base64.b64encode(svg_bytes).decode('utf-8')
            
        except Exception as e:
            return self._create_error_chart(f"Error creating chart: {str(e)}")
    
    def _create_bar_chart(self, data: Dict[str, Any]) -> str:
        """Create a bar chart using SVG."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        title = data.get('title', 'Bar Chart')
        
        if not labels or not values:
            return self._create_error_chart("No data available for bar chart")
        
        # Calculate dimensions
        chart_width = self.width - 2 * self.margin
        chart_height = self.height - 2 * self.margin
        
        # Calculate scales
        max_value = max(values) if values else 1
        min_value = min(values) if values else 0
        value_range = max_value - min_value if max_value != min_value else 1
        
        bar_width = chart_width / len(labels) * 0.8
        bar_spacing = chart_width / len(labels)
        
        # Start SVG
        svg = f'''<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
            .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
            .bar {{ stroke: #333; stroke-width: 1; }}
            .grid {{ stroke: #ddd; stroke-width: 0.5; }}
        </style>
        
        <!-- Background -->
        <rect width="{self.width}" height="{self.height}" fill="white"/>
        
        <!-- Title -->
        <text x="{self.width/2}" y="30" class="title">{title}</text>
        
        <!-- Grid lines -->'''
        
        # Add horizontal grid lines
        for i in range(6):
            y = self.margin + (chart_height * i / 5)
            svg += f'<line x1="{self.margin}" y1="{y}" x2="{self.width - self.margin}" y2="{y}" class="grid"/>'
        
        # Add bars
        for i, (label, value) in enumerate(zip(labels, values)):
            x = self.margin + (i * bar_spacing) + (bar_spacing - bar_width) / 2
            bar_height = (value - min_value) / value_range * chart_height
            y = self.height - self.margin - bar_height
            
            color = self.colors[i % len(self.colors)]
            
            svg += f'''
            <rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" 
                  fill="{color}" class="bar"/>
            <text x="{x + bar_width/2}" y="{self.height - self.margin + 15}" 
                  class="axis-label">{label}</text>
            <text x="{x + bar_width/2}" y="{y - 5}" 
                  class="axis-label">{value}</text>'''
        
        # Add axes
        svg += f'''
        <!-- Axes -->
        <line x1="{self.margin}" y1="{self.margin}" x2="{self.margin}" y2="{self.height - self.margin}" 
              stroke="#333" stroke-width="2"/>
        <line x1="{self.margin}" y1="{self.height - self.margin}" x2="{self.width - self.margin}" y2="{self.height - self.margin}" 
              stroke="#333" stroke-width="2"/>
        </svg>'''
        
        return svg
    
    def _create_line_chart(self, data: Dict[str, Any]) -> str:
        """Create a line chart using SVG."""
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        title = data.get('title', 'Line Chart')
        
        if not x_values or not y_values:
            return self._create_error_chart("No data available for line chart")
        
        # Calculate dimensions
        chart_width = self.width - 2 * self.margin
        chart_height = self.height - 2 * self.margin
        
        # Calculate scales
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_range = x_max - x_min if x_max != x_min else 1
        y_range = y_max - y_min if y_max != y_min else 1
        
        # Start SVG
        svg = f'''<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
            .axis-label {{ font-family: Arial, sans-serif; font-size: 12px; text-anchor: middle; }}
            .line {{ stroke: {self.colors[0]}; stroke-width: 2; fill: none; }}
            .point {{ fill: {self.colors[0]}; }}
            .grid {{ stroke: #ddd; stroke-width: 0.5; }}
        </style>
        
        <!-- Background -->
        <rect width="{self.width}" height="{self.height}" fill="white"/>
        
        <!-- Title -->
        <text x="{self.width/2}" y="30" class="title">{title}</text>
        
        <!-- Grid lines -->'''
        
        # Add grid lines
        for i in range(6):
            y = self.margin + (chart_height * i / 5)
            svg += f'<line x1="{self.margin}" y1="{y}" x2="{self.width - self.margin}" y2="{y}" class="grid"/>'
        
        # Create line path
        points = []
        for x_val, y_val in zip(x_values, y_values):
            x = self.margin + ((x_val - x_min) / x_range) * chart_width
            y = self.height - self.margin - ((y_val - y_min) / y_range) * chart_height
            points.append(f"{x},{y}")
        
        path = "M " + " L ".join(points)
        svg += f'<path d="{path}" class="line"/>'
        
        # Add points
        for x_val, y_val in zip(x_values, y_values):
            x = self.margin + ((x_val - x_min) / x_range) * chart_width
            y = self.height - self.margin - ((y_val - y_min) / y_range) * chart_height
            svg += f'<circle cx="{x}" cy="{y}" r="4" class="point"/>'
        
        # Add axes
        svg += f'''
        <!-- Axes -->
        <line x1="{self.margin}" y1="{self.margin}" x2="{self.margin}" y2="{self.height - self.margin}" 
              stroke="#333" stroke-width="2"/>
        <line x1="{self.margin}" y1="{self.height - self.margin}" x2="{self.width - self.margin}" y2="{self.height - self.margin}" 
              stroke="#333" stroke-width="2"/>
        </svg>'''
        
        return svg
    
    def _create_pie_chart(self, data: Dict[str, Any]) -> str:
        """Create a pie chart using SVG."""
        labels = data.get('labels', [])
        values = data.get('values', [])
        title = data.get('title', 'Pie Chart')
        
        if not labels or not values:
            return self._create_error_chart("No data available for pie chart")
        
        # Calculate total and percentages
        total = sum(values)
        if total == 0:
            return self._create_error_chart("Total value is zero")
        
        # Chart dimensions
        center_x = self.width / 2
        center_y = self.height / 2
        radius = min(self.width, self.height) / 3
        
        # Start SVG
        svg = f'''<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .title {{ font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; text-anchor: middle; }}
            .legend {{ font-family: Arial, sans-serif; font-size: 12px; }}
            .slice {{ stroke: white; stroke-width: 2; }}
        </style>
        
        <!-- Background -->
        <rect width="{self.width}" height="{self.height}" fill="white"/>
        
        <!-- Title -->
        <text x="{center_x}" y="30" class="title">{title}</text>'''
        
        # Draw pie slices
        current_angle = 0
        for i, (label, value) in enumerate(zip(labels, values)):
            angle = (value / total) * 2 * math.pi
            
            # Calculate arc endpoints
            x1 = center_x + radius * math.cos(current_angle)
            y1 = center_y + radius * math.sin(current_angle)
            x2 = center_x + radius * math.cos(current_angle + angle)
            y2 = center_y + radius * math.sin(current_angle + angle)
            
            # Large arc flag
            large_arc = 1 if angle > math.pi else 0
            
            # Create path
            path = f"M {center_x} {center_y} L {x1} {y1} A {radius} {radius} 0 {large_arc} 1 {x2} {y2} Z"
            
            color = self.colors[i % len(self.colors)]
            svg += f'<path d="{path}" fill="{color}" class="slice"/>'
            
            # Add legend
            legend_y = 60 + i * 20
            svg += f'''
            <rect x="20" y="{legend_y - 10}" width="15" height="15" fill="{color}"/>
            <text x="45" y="{legend_y}" class="legend">{label}: {value} ({value/total*100:.1f}%)</text>'''
            
            current_angle += angle
        
        svg += '</svg>'
        return svg
    
    def _create_scatter_chart(self, data: Dict[str, Any]) -> str:
        """Create a scatter plot using SVG."""
        x_values = data.get('x', [])
        y_values = data.get('y', [])
        title = data.get('title', 'Scatter Plot')
        
        if not x_values or not y_values:
            return self._create_error_chart("No data available for scatter plot")
        
        # Similar to line chart but with points only
        return self._create_line_chart(data).replace('class="line"', 'class="line" style="opacity:0"')
    
    def _create_error_chart(self, message: str) -> str:
        """Create an error chart."""
        svg = f'''<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">
        <style>
            .error {{ font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; fill: #dc2626; }}
        </style>
        
        <rect width="{self.width}" height="{self.height}" fill="white"/>
        <text x="{self.width/2}" y="{self.height/2}" class="error">{message}</text>
        </svg>'''
        
        svg_bytes = svg.encode('utf-8')
        return base64.b64encode(svg_bytes).decode('utf-8')
    
    def create_visualization_from_table(self, table_data: List[List[str]], chart_type: str = 'bar') -> str:
        """
        Create visualization from table data.
        
        Args:
            table_data: List of rows, first row is headers
            chart_type: Type of chart to create
            
        Returns:
            Base64 encoded SVG chart
        """
        try:
            if len(table_data) < 2:
                return self._create_error_chart("Insufficient data for visualization")
            
            headers = table_data[0]
            rows = table_data[1:]
            
            # Try to find numeric columns
            numeric_cols = []
            for i, header in enumerate(headers):
                try:
                    # Test if column is numeric
                    [float(row[i]) for row in rows[:5]]  # Test first 5 rows
                    numeric_cols.append(i)
                except (ValueError, IndexError):
                    continue
            
            if not numeric_cols:
                return self._create_error_chart("No numeric data found for visualization")
            
            # Create chart data
            labels = [row[0] for row in rows]  # First column as labels
            values = [float(row[numeric_cols[0]]) for row in rows]  # First numeric column
            
            chart_data = {
                'labels': labels,
                'values': values,
                'title': f'{headers[numeric_cols[0]]} by {headers[0]}'
            }
            
            return self.create_chart(chart_data, chart_type)
            
        except Exception as e:
            return self._create_error_chart(f"Error creating visualization: {str(e)}")

def create_simple_visualization(data: Dict[str, Any], chart_type: str = 'bar') -> str:
    """
    Create a simple visualization from data.
    
    Args:
        data: Data dictionary
        chart_type: Type of chart
        
    Returns:
        Base64 encoded SVG chart
    """
    visualizer = DataVisualizer()
    return visualizer.create_chart(data, chart_type)
"""
Lightweight Data Analyst Agent for Vercel deployment
Handles web scraping and OpenAI analysis without heavy dependencies
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from query_processor import QueryProcessor
from web_scraper import WebScraper
from visualization import DataVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalystAgent:
    """
    Lightweight data analyst agent that coordinates query processing and web scraping.
    Optimized for Vercel deployment without pandas/numpy/matplotlib.
    """
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.web_scraper = WebScraper()
        self.visualizer = DataVisualizer()
        
    def process_query(self, query: str, data_source: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a natural language query and return analysis results.
        
        Args:
            query: Natural language query
            data_source: Optional data source URL
            
        Returns:
            Dict containing analysis results
        """
        try:
            # Step 1: Process the query using OpenAI
            logger.info(f"Processing query: {query}")
            analysis_plan = self.query_processor.process_query(query)
            
            if not analysis_plan:
                return self._create_error_response("Failed to process query")
            
            # Step 2: Extract data if needed
            data = None
            if data_source:
                logger.info(f"Extracting data from: {data_source}")
                data = self.web_scraper.scrape_data(data_source)
            elif analysis_plan.get('data_source'):
                logger.info(f"Extracting data from: {analysis_plan['data_source']}")
                data = self.web_scraper.scrape_data(analysis_plan['data_source'])
            
            # Step 3: Perform analysis
            results = self._perform_analysis(analysis_plan, data)
            
            # Step 4: Create visualization if data is available
            visualization = None
            if data and data.get('tables'):
                visualization = self._create_visualization(data)
            
            return {
                'success': True,
                'message': 'Analysis completed successfully',
                'query': query,
                'analysis_plan': analysis_plan,
                'results': results,
                'visualization': visualization
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._create_error_response(f"Error processing query: {str(e)}")
    
    def _perform_analysis(self, analysis_plan: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform the actual analysis based on the plan.
        
        Args:
            analysis_plan: Analysis plan from query processor
            data: Extracted data (if any)
            
        Returns:
            Analysis results
        """
        results = {}
        
        try:
            # Handle different analysis types
            analysis_type = analysis_plan.get('analysis_type', 'general')
            
            if analysis_type == 'web_content':
                results = self._analyze_web_content(data)
            elif analysis_type == 'comparison':
                results = self._perform_comparison(analysis_plan, data)
            elif analysis_type == 'summary':
                results = self._create_summary(data)
            else:
                results = self._general_analysis(analysis_plan, data)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in analysis: {str(e)}")
            return {'error': f"Analysis failed: {str(e)}"}
    
    def _analyze_web_content(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze web content data."""
        if not data:
            return {'error': 'No data available for analysis'}
        
        # Extract key information
        content = data.get('content', '')
        if not content:
            return {'error': 'No content found'}
        
        # Basic content analysis
        word_count = len(content.split())
        char_count = len(content)
        
        # Extract tables if available
        tables = data.get('tables', [])
        
        return {
            'content_analysis': {
                'word_count': word_count,
                'character_count': char_count,
                'has_tables': len(tables) > 0,
                'table_count': len(tables)
            },
            'tables': tables,
            'content_preview': content[:500] + "..." if len(content) > 500 else content
        }
    
    def _perform_comparison(self, analysis_plan: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comparison analysis."""
        if not data:
            return {'error': 'No data available for comparison'}
        
        # Extract comparison criteria from plan
        criteria = analysis_plan.get('comparison_criteria', [])
        
        # Basic comparison logic
        tables = data.get('tables', [])
        comparison_results = []
        
        for table in tables:
            if isinstance(table, dict) and 'data' in table:
                table_data = table['data']
                if len(table_data) > 1:  # Has header + data rows
                    comparison_results.append({
                        'table_name': table.get('name', 'Unknown'),
                        'row_count': len(table_data) - 1,  # Exclude header
                        'column_count': len(table_data[0]) if table_data else 0,
                        'headers': table_data[0] if table_data else []
                    })
        
        return {
            'comparison_type': 'table_comparison',
            'criteria': criteria,
            'results': comparison_results
        }
    
    def _create_summary(self, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create summary of data."""
        if not data:
            return {'error': 'No data available for summary'}
        
        summary = {
            'data_type': 'web_content',
            'has_content': bool(data.get('content')),
            'has_tables': bool(data.get('tables')),
            'content_length': len(data.get('content', '')),
            'table_count': len(data.get('tables', []))
        }
        
        # Add content summary if available
        content = data.get('content', '')
        if content:
            summary['content_preview'] = content[:200] + "..." if len(content) > 200 else content
        
        return summary
    
    def _general_analysis(self, analysis_plan: Dict[str, Any], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform general analysis."""
        results = {
            'analysis_type': 'general',
            'plan': analysis_plan
        }
        
        if data:
            results['data_summary'] = self._create_summary(data)
        
        return results
    
    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'message': message,
            'error': True
        }
    
    def _create_visualization(self, data: Dict[str, Any]) -> Optional[str]:
        """Create visualization from extracted data."""
        try:
            if not data or not data.get('tables'):
                return None
            
            # Use the first table for visualization
            table = data['tables'][0]
            if not table or len(table) < 2:
                return None
            
            # Create visualization using the lightweight visualizer
            base64_chart = self.visualizer.create_visualization_from_table(table, 'scatter')
            
            # Add data: prefix for base64 image
            if base64_chart:
                return f"data:image/svg+xml;base64,{base64_chart}"
            
            return None
            
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None

# Demo functionality for testing without OpenAI
def create_demo_response(query: str) -> Dict[str, Any]:
    """Create demo response when OpenAI is not available."""
    return {
        'success': True,
        'message': 'Demo mode - OpenAI integration required for full functionality',
        'query': query,
        'analysis_plan': {
            'analysis_type': 'demo',
            'data_source': 'https://en.wikipedia.org/wiki/Data_analysis',
            'steps': ['Extract data', 'Analyze content', 'Generate summary']
        },
        'results': {
            'demo_note': 'This is a demo response. Add OPENAI_API_KEY environment variable for full functionality.',
            'suggested_queries': [
                'Analyze data from Wikipedia about artificial intelligence',
                'Compare population data between countries',
                'Summarize recent technology trends'
            ]
        }
    }
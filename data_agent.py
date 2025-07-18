import os
import json
import logging
import re
import time
from typing import Dict, List, Any, Optional
from openai import OpenAI
from query_processor import QueryProcessor
from web_scraper import WebScraper
from visualization import DataVisualizer
import pandas as pd
import numpy as np
from io import StringIO
import traceback

class DataAnalystAgent:
    """Main agent that orchestrates data analysis tasks."""
    
    def __init__(self):
        self.openai_client = None
        self.query_processor = None
        self.web_scraper = WebScraper()
        self.visualizer = DataVisualizer()
        self.logger = logging.getLogger(__name__)
        self._initialize_openai_client()
        
    def _initialize_openai_client(self):
        """Initialize OpenAI client with proper error handling."""
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not found - agent will operate in demo mode")
                self.openai_client = None
                self.query_processor = None
                return
            
            # Clean the API key to remove any whitespace
            api_key = api_key.strip()
            
            self.openai_client = OpenAI(api_key=api_key)
            self.query_processor = QueryProcessor(self.openai_client)
            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.openai_client = None
            self.query_processor = None
        
    def process_query(self, query: str) -> Any:
        """Process a natural language query and return results."""
        start_time = time.time()
        
        try:
            # Check if OpenAI client is available
            if not self.openai_client or not self.query_processor:
                return {
                    "error": "OpenAI API key not configured",
                    "message": "The Data Analyst Agent requires an OpenAI API key to process queries. Please provide your OPENAI_API_KEY.",
                    "demo_mode": True,
                    "results": []
                }
            
            # Step 1: Parse and understand the query
            self.logger.info("Step 1: Parsing query...")
            analysis_plan = self.query_processor.parse_query(query)
            self.logger.info(f"Analysis plan: {analysis_plan}")
            
            # Step 2: Execute the analysis plan
            self.logger.info("Step 2: Executing analysis plan...")
            result = self._execute_analysis_plan(analysis_plan, query)
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"Query processed in {elapsed_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def _execute_analysis_plan(self, plan: Dict, original_query: str) -> Any:
        """Execute the analysis plan step by step."""
        data = None
        results = {}
        
        # Step 1: Data sourcing
        if plan.get('data_source'):
            data = self._source_data(plan['data_source'])
            
        # Step 2: Data processing and analysis
        if plan.get('analysis_steps'):
            for step in plan['analysis_steps']:
                if step['type'] == 'simple_calculation':
                    # Handle simple questions using OpenAI
                    results.update(self._handle_simple_calculation(step))
                elif step['type'] == 'statistical_analysis':
                    if data is not None:
                        results.update(self._perform_statistical_analysis(data, step))
                elif step['type'] == 'visualization':
                    if data is not None:
                        results.update(self._create_visualization(data, step))
                elif step['type'] == 'query_execution':
                    # Handle count operations directly
                    if step.get('analysis_type') == 'count' and data is not None:
                        results['count'] = len(data)
                    else:
                        results.update(self._execute_query(step))
                elif step['type'] == 'data_transformation':
                    # Check if this is a count operation that should return a result
                    if step.get('analysis_type') == 'count':
                        if data is not None:
                            results['count'] = len(data)
                    else:
                        if data is not None:
                            data = self._transform_data(data, step)
        
        # Step 3: Format results based on expected output format
        self.logger.info(f"Raw results before formatting: {results}")
        formatted_result = self._format_results(results, plan.get('output_format', 'json'))
        self.logger.info(f"Formatted result: {formatted_result}")
        return formatted_result
    
    def _source_data(self, data_source: Dict) -> Optional[pd.DataFrame]:
        """Source data from various sources."""
        source_type = data_source.get('type')
        
        if source_type == 'web_scraping':
            url = data_source.get('url')
            if url:
                # Try Wikipedia-specific scraping first for better results
                if 'wikipedia.org' in url:
                    return self.web_scraper.scrape_wikipedia_table(url)
                else:
                    return self.web_scraper.scrape_table_data(url)
        
        elif source_type == 'csv_url':
            url = data_source.get('url')
            if url:
                return pd.read_csv(url)
        
        return None
    
    def _perform_statistical_analysis(self, data: pd.DataFrame, step: Dict) -> Dict:
        """Perform statistical analysis on the data."""
        results = {}
        
        if not data.empty:
            analysis_type = step.get('analysis_type')
            
            if analysis_type == 'correlation':
                columns = step.get('columns', [])
                if len(columns) >= 2:
                    col1, col2 = columns[0], columns[1]
                    if col1 in data.columns and col2 in data.columns:
                        try:
                            # Ensure columns are numeric
                            numeric_col1 = pd.to_numeric(data[col1], errors='coerce')
                            numeric_col2 = pd.to_numeric(data[col2], errors='coerce')
                            
                            # Drop rows with NaN values for correlation calculation
                            clean_data = pd.DataFrame({col1: numeric_col1, col2: numeric_col2}).dropna()
                            
                            if len(clean_data) > 1:
                                corr = clean_data[col1].corr(clean_data[col2])
                                results['correlation'] = corr
                            else:
                                results['correlation'] = 0.0
                                
                        except Exception as e:
                            self.logger.error(f"Error calculating correlation: {str(e)}")
                            results['correlation'] = 0.0
            
            elif analysis_type == 'count':
                condition = step.get('condition')
                if condition:
                    # Use LLM to evaluate complex conditions
                    count = self._evaluate_condition(data, condition)
                    results['count'] = count
                else:
                    results['count'] = len(data)
            
            elif analysis_type == 'regression':
                columns = step.get('columns', [])
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    if x_col in data.columns and y_col in data.columns:
                        try:
                            slope, intercept = np.polyfit(data[x_col], data[y_col], 1)
                            results['regression_slope'] = slope
                            results['regression_intercept'] = intercept
                        except Exception as e:
                            self.logger.error(f"Error calculating regression: {str(e)}")
        
        return results
    
    def _create_visualization(self, data: pd.DataFrame, step: Dict) -> Dict:
        """Create visualizations based on the step configuration."""
        results = {}
        
        if not data.empty:
            viz_type = step.get('viz_type')
            
            if viz_type == 'scatterplot':
                columns = step.get('columns', [])
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    if x_col in data.columns and y_col in data.columns:
                        plot_config = step.get('plot_config', {})
                        base64_image = self.visualizer.create_scatterplot(
                            data, x_col, y_col, plot_config
                        )
                        results['plot'] = base64_image
            
            elif viz_type == 'histogram':
                column = step.get('column')
                if column and column in data.columns:
                    base64_image = self.visualizer.create_histogram(data, column)
                    results['histogram'] = base64_image
            
            elif viz_type == 'line_plot':
                columns = step.get('columns', [])
                if len(columns) >= 2:
                    x_col, y_col = columns[0], columns[1]
                    if x_col in data.columns and y_col in data.columns:
                        base64_image = self.visualizer.create_line_plot(data, x_col, y_col)
                        results['line_plot'] = base64_image
        
        return results
    
    def _execute_query(self, step: Dict) -> Dict:
        """Execute database queries."""
        results = {}
        
        # Handle count analysis type in query execution
        if step.get('analysis_type') == 'count':
            # This should be handled in the main execution loop
            results['count_operation'] = True
        
        return results
    
    def _transform_data(self, data: pd.DataFrame, step: Dict) -> pd.DataFrame:
        """Transform data based on the step configuration."""
        if data is None:
            return data
        
        transform_type = step.get('transform_type')
        
        if transform_type == 'filter':
            condition = step.get('condition')
            if condition:
                return self._filter_data(data, condition)
        
        elif transform_type == 'aggregate':
            group_by = step.get('group_by')
            agg_func = step.get('agg_func', 'count')
            if group_by:
                return data.groupby(group_by).agg(agg_func).reset_index()
        
        elif transform_type == 'sort':
            sort_by = step.get('sort_by')
            ascending = step.get('ascending', True)
            if sort_by:
                return data.sort_values(by=sort_by, ascending=ascending)
        
        return data
    
    def _filter_data(self, data: pd.DataFrame, condition: str) -> pd.DataFrame:
        """Filter data based on natural language condition."""
        try:
            # Use LLM to convert natural language condition to pandas query
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                    messages=[
                        {
                            "role": "system",
                            "content": "Convert the natural language condition to a pandas query expression. "
                                     f"Available columns: {list(data.columns)}. "
                                     "Return only the query expression, no explanation. "
                                     "Use pandas query syntax."
                        },
                        {"role": "user", "content": condition}
                    ]
                )
                
                query_expression = response.choices[0].message.content.strip()
                filtered_data = data.query(query_expression)
                return filtered_data
            else:
                return data
            
        except Exception as e:
            self.logger.error(f"Error filtering data: {str(e)}")
            return data
    
    def _evaluate_condition(self, data: pd.DataFrame, condition: str) -> int:
        """Evaluate a condition on the data and return count."""
        try:
            filtered_data = self._filter_data(data, condition)
            return len(filtered_data)
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {str(e)}")
            return 0
    
    def _handle_simple_calculation(self, step: Dict) -> Dict:
        """Handle simple calculations using OpenAI."""
        try:
            question = step.get('question', '')
            
            if self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions directly and concisely. "
                                     "For mathematical calculations, provide the numerical result. "
                                     "For factual questions, provide accurate information."
                        },
                        {"role": "user", "content": question}
                    ]
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Try to extract numeric result if it's a calculation
                try:
                    # Check if the answer contains a number
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', answer)
                    if numbers:
                        numeric_result = float(numbers[0])
                        return {'result': numeric_result, 'answer': answer}
                except:
                    pass
                
                return {'answer': answer}
            else:
                return {'error': 'OpenAI client not available'}
            
        except Exception as e:
            self.logger.error(f"Error handling simple calculation: {str(e)}")
            return {'error': str(e)}
    
    def _format_results(self, results: Dict, output_format: str) -> Any:
        """Format results based on the expected output format."""
        if output_format == 'array':
            # Return as array for IIT Madras format
            values = []
            for key, value in results.items():
                if isinstance(value, (int, float, str)):
                    values.append(value)
                elif isinstance(value, dict) and 'result' in value:
                    values.append(value['result'])
                elif isinstance(value, dict) and 'answer' in value:
                    values.append(value['answer'])
            return values
        
        elif output_format == 'json_object':
            return results
        
        else:
            # Default format with message and analysis
            return {
                'results': list(results.values()),
                'message': 'Analysis completed successfully',
                'analysis': f'Processed {len(results)} analysis steps'
            }
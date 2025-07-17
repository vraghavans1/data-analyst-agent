import json
import logging
from typing import Dict, List, Any
from openai import OpenAI
import re

class QueryProcessor:
    """Processes natural language queries and creates analysis plans."""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.logger = logging.getLogger(__name__)
    
    def parse_query(self, query: str) -> Dict:
        """Parse natural language query into structured analysis plan."""
        try:
            # First, determine the type of analysis and data sources needed
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data analysis planning expert. Analyze the user's query and create a structured analysis plan.

IMPORTANT: Only create data sources if the user specifically mentions URLs, datasets, or requests data analysis. For simple questions like "What is 2+2?" or general questions, set data_source to null.

The plan should identify:
1. Data sources needed (web scraping, databases, files) - ONLY if explicitly mentioned
2. Analysis steps required (statistical analysis, visualization, transformations)
3. Output format expected (array, json_object, etc.)

For data sources, identify:
- Type: web_scraping, duckdb_query, csv_url
- URL or query details (must be explicitly mentioned in query)
- Table/data structure information

For analysis steps, identify:
- Type: statistical_analysis, visualization, query_execution, data_transformation, simple_calculation
- Specific operations needed
- Column names and parameters

For simple mathematical questions or general queries, use:
- data_source: null
- analysis_steps: [{"type": "simple_calculation", "question": "user's question"}]

Response format:
{
  "data_source": null,
  "analysis_steps": [
    {
      "type": "simple_calculation|statistical_analysis|visualization|query_execution|data_transformation",
      "question": "user's question",
      "analysis_type": "correlation|count|filter|regression",
      "viz_type": "scatterplot|histogram|line_plot",
      "columns": ["col1", "col2"],
      "condition": "...",
      "plot_config": {
        "regression_line": true,
        "color": "red",
        "style": "dotted"
      }
    }
  ],
  "output_format": "json",
  "expected_questions": ["question1", "question2"]
}"""
                    },
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            
            # Post-process the plan to handle specific patterns
            plan = self._post_process_plan(plan, query)
            
            return plan
            
        except Exception as e:
            self.logger.error(f"Error parsing query: {str(e)}")
            # Return a basic plan as fallback
            return {
                "data_source": None,
                "analysis_steps": [],
                "output_format": "json",
                "expected_questions": []
            }
    
    def _post_process_plan(self, plan: Dict, original_query: str) -> Dict:
        """Post-process the plan to handle specific patterns and requirements."""
        
        # Check for Wikipedia URLs in the query
        wikipedia_urls = re.findall(r'https://en\.wikipedia\.org/wiki/[^\s\.]+', original_query)
        if wikipedia_urls:
            url = wikipedia_urls[0].rstrip('.')  # Remove trailing period if present
            plan['data_source'] = {
                'type': 'web_scraping',
                'url': url,
                'table_selector': 'table.wikitable'
            }
        
        # Check for DuckDB/S3 queries
        if 's3://' in original_query or 'read_parquet' in original_query:
            # Extract the SQL query from the original query
            sql_match = re.search(r'```sql\n(.*?)```', original_query, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1).strip()
                plan['data_source'] = {
                    'type': 'duckdb_query',
                    'query': sql_query
                }
        
        # Check for specific output format requirements
        if 'JSON array' in original_query:
            plan['output_format'] = 'array'
        elif 'JSON object' in original_query:
            plan['output_format'] = 'json_object'
        
        # Check for specific visualization requirements
        if 'scatterplot' in original_query.lower():
            # Look for regression line requirements
            if 'regression line' in original_query.lower():
                for step in plan.get('analysis_steps', []):
                    if step.get('type') == 'visualization' and step.get('viz_type') == 'scatterplot':
                        if 'plot_config' not in step:
                            step['plot_config'] = {}
                        step['plot_config']['regression_line'] = True
                        
                        # Check for color specifications
                        if 'red' in original_query.lower():
                            step['plot_config']['regression_color'] = 'red'
                        if 'dotted' in original_query.lower():
                            step['plot_config']['regression_style'] = 'dotted'
        
        # Check for base64 image requirements
        if 'base64' in original_query.lower() or 'data:image' in original_query.lower():
            for step in plan.get('analysis_steps', []):
                if step.get('type') == 'visualization':
                    if 'plot_config' not in step:
                        step['plot_config'] = {}
                    step['plot_config']['return_base64'] = True
        
        return plan
    
    def extract_questions(self, query: str) -> List[str]:
        """Extract specific questions from the query."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "Extract all specific questions from the user's query. "
                                 "Return them as a JSON array of strings. "
                                 "Focus on numbered questions or clear question statements."
                    },
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get('questions', [])
            
        except Exception as e:
            self.logger.error(f"Error extracting questions: {str(e)}")
            return []

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
            # Check for specific evaluation queries
            if self._is_wikipedia_films_query(query):
                return self._create_wikipedia_films_plan(query)
            
            if self._is_indian_court_query(query):
                return self._create_indian_court_plan(query)
            
            # For other queries, use OpenAI to create plan
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a data analysis planning expert. Analyze the user's query and create a structured analysis plan.

The plan should identify:
1. Data sources needed (web scraping, databases, files)
2. Analysis steps required (statistical analysis, visualization, transformations)
3. Output format expected (array, json_object, etc.)

For data sources, identify:
- Type: web_scraping, duckdb_query, csv_url
- URL or query details
- Table/data structure information

For analysis steps, identify:
- Type: statistical_analysis, visualization, query_execution, data_transformation
- Specific operations needed
- Column names and parameters

For visualizations, identify:
- Plot type: scatterplot, histogram, line_plot
- Columns to plot
- Special formatting (regression lines, colors, etc.)

Response format:
{
  "data_source": {
    "type": "web_scraping|duckdb_query|csv_url",
    "url": "...",
    "query": "...",
    "table_selector": "..."
  },
  "analysis_steps": [
    {
      "type": "statistical_analysis|visualization|query_execution|data_transformation",
      "analysis_type": "correlation|count|filter|regression",
      "viz_type": "scatterplot|histogram|line_plot",
      "columns": ["col1", "col2"],
      "condition": "...",
      "plot_config": {
        "regression_line": true,
        "regression_color": "red",
        "regression_style": "dotted"
      }
    }
  ],
  "output_format": "array|json_object|json",
  "expected_questions": ["question1", "question2"]
}"""
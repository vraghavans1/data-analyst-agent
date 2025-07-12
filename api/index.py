from flask import Flask, request, jsonify, render_template_string
import os
import sys
import logging
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import trafilatura
import duckdb
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Set up matplotlib for serverless
plt.ioff()
sns.set_style("whitegrid")

class SimpleDataAgent:
    """Simplified data agent for Vercel deployment."""
    
    def __init__(self):
        self.openai_client = openai_client
    
    def process_query(self, query: str):
        """Process a natural language query and return results."""
        try:
            # Handle simple numeric queries
            if "count" in query.lower() and any(char.isdigit() for char in query):
                numbers = [int(char) for char in query if char.isdigit()]
                if numbers:
                    return {
                        "results": numbers,
                        "message": f"Found numbers: {numbers}",
                        "analysis": f"Count query processed: {len(numbers)} numbers found"
                    }
            
            # Use OpenAI to process the query
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analyst. Process the user's query and provide analysis results. Return JSON with 'results', 'message', and 'analysis' fields."
                    },
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Create a simple visualization if numbers are involved
            if "results" in result and isinstance(result["results"], list):
                if all(isinstance(x, (int, float)) for x in result["results"]):
                    result["visualization"] = self._create_simple_chart(result["results"])
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "results": [],
                "message": f"Error processing query: {str(e)}",
                "analysis": "Query processing failed"
            }
    
    def _create_simple_chart(self, data):
        """Create a simple chart from data."""
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if len(data) == 1:
                # Single value - show as text
                ax.text(0.5, 0.5, f"Value: {data[0]}", 
                       horizontalalignment='center', verticalalignment='center',
                       fontsize=20, transform=ax.transAxes)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title("Single Value Result")
            else:
                # Multiple values - show as bar chart
                ax.bar(range(len(data)), data)
                ax.set_title("Data Analysis Results")
                ax.set_xlabel("Index")
                ax.set_ylabel("Value")
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            return None

# Initialize the simplified agent
agent = SimpleDataAgent()

@app.route('/')
def index():
    """Serve the documentation and testing interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analyst Agent API</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <style>
            body { padding: 20px; }
            .container { max-width: 1200px; }
            .code-block { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
            pre { margin: 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Data Analyst Agent API</h1>
            <p>AI-powered data analysis service that processes natural language queries.</p>
            
            <h2>API Endpoint</h2>
            <div class="code-block">
                <pre>POST /api/</pre>
            </div>
            
            <h3>Example Usage</h3>
            <div class="code-block">
                <pre>curl -X POST https://your-vercel-url.vercel.app/api/ \\
  -H "Content-Type: application/json" \\
  -d '{"query": "Count the number 5"}'</pre>
            </div>
            
            <h3>Response Format</h3>
            <div class="code-block">
                <pre>{
  "results": [5],
  "message": "Analysis completed successfully",
  "visualization": "base64-encoded-image-data"
}</pre>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis tasks."""
    try:
        # Get query from request
        content_type = request.content_type
        
        if content_type and 'application/json' in content_type:
            data = request.get_json()
            query = data.get('query', '')
        elif content_type and 'multipart/form-data' in content_type:
            if 'file' in request.files:
                file = request.files['file']
                query = file.read().decode('utf-8')
            else:
                query = request.form.get('query', '')
        else:
            # Handle plain text
            query = request.get_data(as_text=True)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Process the query
        results = agent.process_query(query)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Data Analyst Agent'})

if __name__ == '__main__':
    app.run(debug=True)

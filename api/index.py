from flask import Flask, request, jsonify, render_template_string
import os
import sys
import logging

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_agent import DataAnalystAgent

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the agent
agent = DataAnalystAgent()

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

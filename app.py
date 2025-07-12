import os
import logging
from flask import Flask, request, jsonify, render_template
from flask.logging import default_handler
from data_agent import DataAnalystAgent
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key")

# Initialize the data analyst agent
agent = DataAnalystAgent()

@app.route('/')
def index():
    """Serve the documentation and testing interface."""
    return render_template('index.html')

@app.route('/api/', methods=['POST'])
def analyze_data():
    """Main API endpoint for data analysis tasks."""
    try:
        # Get the request data
        if request.content_type and 'multipart/form-data' in request.content_type:
            # Handle file upload
            if request.files:
                file = list(request.files.values())[0]
                query = file.read().decode('utf-8')
            else:
                return jsonify({"error": "No file provided"}), 400
        else:
            # Handle JSON or text body
            if request.is_json:
                data = request.get_json()
                query = data.get('query', '')
            else:
                query = request.get_data(as_text=True)
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        app.logger.info(f"Received query: {query[:200]}...")
        
        # Process the query with the agent
        result = agent.process_query(query)
        
        app.logger.info("Query processed successfully")
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "message": "Data Analyst Agent is running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

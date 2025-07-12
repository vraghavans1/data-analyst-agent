from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'Data Analyst Agent',
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True)
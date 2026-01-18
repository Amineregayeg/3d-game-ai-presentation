#!/usr/bin/env python3
"""
Simple Flask server to test Salesforce RAG API locally
"""
import os

# Force CPU mode to avoid CUDA issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, jsonify
from flask_cors import CORS

# Import blueprints
from salesforce_rag_api import salesforce_rag_bp
from salesforce_mcp_api import salesforce_mcp_bp

app = Flask(__name__)
CORS(app)

# Register blueprints
app.register_blueprint(salesforce_rag_bp)
app.register_blueprint(salesforce_mcp_bp)

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "service": "Salesforce Virtual Assistant API",
        "endpoints": {
            "rag": [
                "GET /api/salesforce-rag/status",
                "POST /api/salesforce-rag/query",
                "GET /api/salesforce-rag/sample-queries"
            ],
            "mcp": [
                "GET /api/salesforce-mcp/status",
                "POST /api/salesforce-mcp/query",
                "GET /api/salesforce-mcp/objects",
                "GET /api/salesforce-mcp/describe/<object>",
                "POST /api/salesforce-mcp/create",
                "POST /api/salesforce-mcp/update",
                "POST /api/salesforce-mcp/delete",
                "POST /api/salesforce-mcp/execute"
            ]
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    print("=" * 60)
    print("SALESFORCE VIRTUAL ASSISTANT API")
    print("=" * 60)
    print("Starting on http://localhost:5000")
    print("\nRAG Endpoints:")
    print("  GET  /api/salesforce-rag/status")
    print("  POST /api/salesforce-rag/query")
    print("\nMCP Endpoints (Salesforce Operations):")
    print("  GET  /api/salesforce-mcp/status")
    print("  POST /api/salesforce-mcp/query")
    print("  POST /api/salesforce-mcp/execute")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)

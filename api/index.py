"""
Lightweight FastAPI application for Vercel deployment
Minimal dependencies version without pandas/numpy/matplotlib
"""

import json
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import lightweight components
try:
    from data_agent_lightweight import DataAnalystAgent, create_demo_response
    from query_processor import QueryProcessor
    from web_scraper_lightweight import WebScraper
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback imports
    DataAnalystAgent = None
    QueryProcessor = None
    WebScraper = None

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent",
    description="AI-powered data analysis service with web scraping capabilities",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QueryRequest(BaseModel):
    query: str
    data_source: Optional[str] = None

class TextQueryRequest(BaseModel):
    text: str
    data_source: Optional[str] = None

# Initialize components
agent = None
if DataAnalystAgent:
    try:
        agent = DataAnalystAgent()
    except Exception as e:
        print(f"Failed to initialize DataAnalystAgent: {e}")
        agent = None

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Data Analyst Agent API",
        "version": "1.0.0",
        "description": "AI-powered data analysis service",
        "endpoints": {
            "POST /api/analyze": "Analyze data using natural language queries",
            "POST /api/analyze-text": "Analyze text content directly",
            "POST /api/analyze-file": "Analyze uploaded files",
            "GET /api/health": "Health check endpoint"
        },
        "features": [
            "Natural language query processing",
            "Web scraping and data extraction",
            "OpenAI GPT-4o integration",
            "Multiple input formats support"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "agent_initialized": agent is not None
    }

@app.post("/api/analyze")
async def analyze_data(request: QueryRequest):
    """
    Analyze data using natural language queries.
    
    Args:
        request: Query request containing the question and optional data source
        
    Returns:
        Analysis results
    """
    try:
        if not agent:
            # Return demo response if agent not initialized
            return create_demo_response(request.query)
        
        # Process the query
        results = agent.process_query(request.query, request.data_source)
        
        return JSONResponse(
            content=results,
            status_code=200 if results.get('success') else 400
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": f"Error processing query: {str(e)}",
                "error": True
            },
            status_code=500
        )

@app.post("/api/analyze-text")
async def analyze_text(request: TextQueryRequest):
    """
    Analyze text content directly.
    
    Args:
        request: Text analysis request
        
    Returns:
        Analysis results
    """
    try:
        if not agent:
            return create_demo_response(f"Analyze this text: {request.text[:100]}...")
        
        # Create a query for text analysis
        query = f"Analyze this text: {request.text}"
        results = agent.process_query(query, request.data_source)
        
        return JSONResponse(
            content=results,
            status_code=200 if results.get('success') else 400
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": f"Error analyzing text: {str(e)}",
                "error": True
            },
            status_code=500
        )

@app.post("/api/analyze-file")
async def analyze_file(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None)
):
    """
    Analyze uploaded files.
    
    Args:
        file: Uploaded file
        query: Optional analysis query
        
    Returns:
        Analysis results
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")
        
        # Read file content
        content = await file.read()
        
        # Try to decode as text
        try:
            text_content = content.decode('utf-8')
        except UnicodeDecodeError:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "File must be a text file",
                    "error": True
                },
                status_code=400
            )
        
        # Create analysis query
        analysis_query = query or f"Analyze this file content: {file.filename}"
        
        if not agent:
            return create_demo_response(analysis_query)
        
        # Process the file content
        results = agent.process_query(analysis_query)
        
        # Add file information to results
        if results.get('success'):
            results['file_info'] = {
                'filename': file.filename,
                'size': len(content),
                'content_preview': text_content[:500] + "..." if len(text_content) > 500 else text_content
            }
        
        return JSONResponse(
            content=results,
            status_code=200 if results.get('success') else 400
        )
        
    except Exception as e:
        return JSONResponse(
            content={
                "success": False,
                "message": f"Error analyzing file: {str(e)}",
                "error": True
            },
            status_code=500
        )

@app.get("/api/demo")
async def demo_endpoint():
    """Demo endpoint showing sample functionality."""
    return {
        "message": "Data Analyst Agent Demo",
        "sample_queries": [
            "Analyze population data from Wikipedia",
            "Compare GDP between countries",
            "Summarize recent technology trends",
            "Extract data from a specific website"
        ],
        "sample_data_sources": [
            "https://en.wikipedia.org/wiki/List_of_countries_by_population",
            "https://en.wikipedia.org/wiki/List_of_countries_by_GDP",
            "https://en.wikipedia.org/wiki/Artificial_intelligence"
        ],
        "note": "Add OPENAI_API_KEY environment variable for full functionality"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        content={"message": "Endpoint not found"},
        status_code=404
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        content={"message": "Internal server error"},
        status_code=500
    )

# For Vercel deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
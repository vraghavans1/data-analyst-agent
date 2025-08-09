#!/usr/bin/env python3
"""
Entry point for the Data Analyst Agent API deployment.
This file serves as the main startup script for various deployment platforms.
"""

import os
import sys
import uvicorn
from api.index import app

def main():
    """Main entry point for the application."""
    # Get port from environment or default to 5000
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting Data Analyst Agent API on {host}:{port}")
    
    # Start the server
    uvicorn.run(
        "api.index:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        log_level="info"
    )

if __name__ == "__main__":
    main()
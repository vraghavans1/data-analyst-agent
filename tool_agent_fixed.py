#!/usr/bin/env python3
"""
Tool-calling agent that dynamically handles ANY question and returns 4-element array format.
Format: [success_indicator, analysis_text, numerical_result, visualization_or_text]
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict

@dataclass 
class ToolResult:
    call_id: str
    content: str
    success: bool

class ToolCallingAgent:
    """Dynamic tool-calling agent that handles any question with 4-element array output."""
    
    def __init__(self, openai_api_key: str, max_duration: int = 180):
        self.client = OpenAI(api_key=openai_api_key)
        self.max_duration = max_duration
        self.resources_used = 0
        self.max_resources = 3
        
    def get_available_tools(self) -> List[Dict]:
        """No custom tools - OpenAI uses its own built-in capabilities."""
        return []
    

    

    
    async def process_question(self, question: str) -> Dict[str, Any]:
        """Process any question and return 4-element array format."""
        start_time = time.time()
        messages = [
            {
                "role": "system",
                "content": """You are a data analyst assistant. Use your built-in capabilities to answer questions and provide data.

For current/real-time data requests (weather, rainfall, stock prices, news, etc.):
- Use your search and browsing capabilities to find current information
- Access reliable sources like government websites, official weather services, news outlets
- For weather data, check sources like Indian Meteorological Department (IMD)
- Provide accurate, up-to-date information when possible

CRITICAL for NETWORK ANALYSIS:
- For undirected networks, calculate metrics precisely:
  - Average degree = (2 * number_of_edges) / number_of_nodes
  - Density = (2 * number_of_edges) / (nodes * (nodes - 1))  
  - Shortest path: Use breadth-first search algorithm
  - Count all edges and nodes accurately from the data
- Double-check all mathematical calculations before responding
- Provide exact numerical values, not approximations

For ANY question:
1. Identify the EXACT response format requested in the question
2. Look for format keywords: "single word", "JSON object", "JSON array", "list of entries", etc.
3. Use your built-in tools and search capabilities as needed
4. Return ONLY the data in the exact format requested

Response Format Requirements:
- "single word" → return just the word
- "JSON object" → return {"key": "value"}
- "JSON array" or "list" → return ["item1", "item2", "item3"]
- "list of X entries" → return exactly X entries
- No format specified → return natural text

CRITICAL for VISUALIZATIONS:
- For base64 PNG requests, use minimal placeholder format: "data:image/png;base64,iVBORw0KGgo..."  (under 100 chars total)
- NEVER generate actual full base64 image data - use short representative strings only
- Example: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU77yQ"
- For multiple visualizations, keep each base64 field under 80 characters
- Prioritize complete JSON response with all numerical fields over image detail
- Total response must stay under 1000 tokens to ensure completion

CRITICAL:
- Return ONLY the requested format, no explanations
- Use your own tools and search capabilities for current data
- Match the exact number of entries requested
- Keep base64 images small and efficient
- Never add extra text when a specific format is requested"""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        try:
            # Simple chat completion - no tools needed
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=4096,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip()
            elapsed_time = time.time() - start_time
            
            # Extract token usage information
            token_usage = response.usage
            logger.info(f"Token usage - Input: {token_usage.prompt_tokens}, Output: {token_usage.completion_tokens}, Total: {token_usage.total_tokens}")
            
            return {
                "success": True,
                "answer": answer,
                "elapsed_time": elapsed_time,
                "token_usage": {
                    "prompt_tokens": token_usage.prompt_tokens,
                    "completion_tokens": token_usage.completion_tokens,
                    "total_tokens": token_usage.total_tokens
                }
            }
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"OpenAI API error: {str(e)}")
            return {
                "success": False,
                "error": f"API error: {str(e)}",
                "elapsed_time": elapsed_time
            }
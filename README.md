# Data Analyst Agent

An AI-powered data analysis API that processes natural language queries to automatically source, analyze, and visualize data.

## Features

- **Natural Language Processing**: Understands complex data analysis requests in plain English
- **Web Scraping**: Extracts data from Wikipedia tables and other web sources
- **Database Queries**: Supports DuckDB for large dataset analysis
- **Statistical Analysis**: Performs correlations, counts, filtering, and aggregations
- **Data Visualization**: Creates charts with custom styling (scatterplots, histograms, etc.)
- **Multiple Input Formats**: Accepts JSON, file uploads, and plain text
- **Base64 Image Output**: Returns visualizations as data URIs under 100KB

## API Endpoints

### Main Analysis Endpoint
```
POST /api/
```

**Request Format:**
- File upload: `curl -F "@query.txt" /api/`
- JSON: `{"query": "your analysis request"}`
- Plain text: Direct text in request body

**Response:** JSON with analysis results

### Health Check
```
GET /health
```

**Response:** `{"status": "healthy", "message": "Data Analyst Agent is running"}`

## Example Queries

### Wikipedia Analysis
```json
{
  "query": "Scrape the list of highest grossing films from Wikipedia at https://en.wikipedia.org/wiki/List_of_highest-grossing_films. Answer: 1. How many films grossed over $2 billion? 2. What is the correlation between Rank and Peak? 3. Create a scatterplot with red dotted regression line."
}
```

### Database Analysis
```json
{
  "query": "Query the Indian High Court dataset. Which high court disposed the most cases from 2019-2022? Plot the year vs delay trends."
}
```

## Technical Architecture

- **Flask API** with gunicorn WSGI server
- **OpenAI GPT-4o** for natural language understanding
- **DuckDB** for fast in-memory analytics
- **Pandas** for data manipulation
- **Matplotlib/Seaborn** for visualization
- **BeautifulSoup/Trafilatura** for web scraping

## Environment Variables

- `OPENAI_API_KEY`: Required for AI processing
- `SESSION_SECRET`: Optional Flask session key

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables
4. Run: `python main.py`

## Docker Deployment

The application is configured for containerized deployment with:
- Port 5000 binding to 0.0.0.0
- Gunicorn WSGI server
- Auto-reload for development

## Response Time

Designed to complete analysis within 3 minutes as per evaluation requirements.

## License

MIT License
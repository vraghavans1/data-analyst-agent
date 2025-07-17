# Data Analyst Agent

AI-powered data analysis service for IIT Madras evaluation that processes natural language queries and performs complex data analysis tasks.

## Features

- **Natural Language Processing**: Uses OpenAI GPT-4o to understand and interpret data analysis queries
- **Web Scraping**: Extracts data from Wikipedia and other web sources
- **Statistical Analysis**: Performs correlations, calculations, and data insights
- **Data Visualization**: Generates charts and plots as base64-encoded images
- **Multiple Input Formats**: Supports file uploads, JSON, and raw text queries

## API Endpoint

**URL**: `/api/`  
**Method**: `POST`

### Input Formats

1. **File Upload** (IIT Madras format):
   ```bash
   curl -F "file=@question.txt" https://your-domain.vercel.app/api/
   ```

2. **JSON Body**:
   ```bash
   curl -X POST https://your-domain.vercel.app/api/ \
     -H "Content-Type: application/json" \
     -d '{"query": "your data analysis question"}'
   ```

3. **Raw Text**:
   ```bash
   curl -X POST https://your-domain.vercel.app/api/ \
     -H "Content-Type: text/plain" \
     -d "your data analysis question"
   ```

### Response Format

Returns JSON with analysis results:
```json
{
  "results": [1, "Titanic", 0.485782, "data:image/png;base64,..."],
  "message": "Analysis completed successfully",
  "analysis": "Detailed analysis description"
}
```

## Deployment

### Vercel

1. Clone this repository
2. Install Vercel CLI: `npm i -g vercel`
3. Deploy: `vercel --prod`
4. Set environment variable: `OPENAI_API_KEY`

### Local Development

1. Install dependencies: `pip install -r api/requirements.txt`
2. Set environment variable: `export OPENAI_API_KEY=your-key`
3. Run: `uvicorn api.index:app --reload`

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI API access

## License

MIT License - see LICENSE file for details.

## Architecture

- **FastAPI**: Modern web framework for APIs
- **OpenAI GPT-4o**: Natural language processing and query understanding
- **BeautifulSoup**: Web scraping and HTML parsing
- **Pandas/NumPy**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **DuckDB**: In-memory SQL database for fast analytics

## IIT Madras Evaluation

This application is specifically designed to meet IIT Madras evaluation criteria:

- ✅ Accepts POST requests with file uploads
- ✅ Returns JSON responses with analysis results
- ✅ Processes Wikipedia scraping queries
- ✅ Performs statistical calculations (correlations)
- ✅ Generates visualizations with regression lines
- ✅ Completes within 3-minute timeout
- ✅ Returns base64-encoded images under 100KB
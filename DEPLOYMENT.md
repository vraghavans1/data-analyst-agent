# Deployment Guide

## Replit Deployment

This application is designed to be deployed on Replit for public access.

### Environment Setup

1. **OpenAI API Key**: Required for natural language processing
   - Set `OPENAI_API_KEY` in Replit Secrets
   - Get your key from: https://platform.openai.com/api-keys

2. **Session Secret**: Optional Flask session key
   - Set `SESSION_SECRET` in Replit Secrets (defaults to "default-secret-key")

### Deployment Steps

1. Click the "Deploy" button in your Replit interface
2. Your application will be available at: `https://your-repl-name.replit.app`
3. Test the deployment with: `curl https://your-repl-name.replit.app/health`

### API Endpoints

- **Main API**: `https://your-repl-name.replit.app/api/`
- **Health Check**: `https://your-repl-name.replit.app/health`
- **Documentation**: `https://your-repl-name.replit.app/`

### Testing the Deployment

```bash
# Health check
curl https://your-repl-name.replit.app/health

# Test with simple query
curl -X POST https://your-repl-name.replit.app/api/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Count the number 5"}'

# Test with file upload
curl -X POST https://your-repl-name.replit.app/api/ \
  -F "@query.txt"
```

### Production Configuration

The application runs with:
- **Gunicorn WSGI server** for production stability
- **Port 5000** bound to 0.0.0.0 for public access
- **Auto-reload** enabled for development
- **3-minute timeout** for complex queries

### Monitoring

Check application logs through the Replit console for:
- Request processing times
- Error messages
- OpenAI API usage
- Data processing status

### Troubleshooting

- **503 Error**: Check if OpenAI API key is set correctly
- **Timeout**: Ensure queries complete within 3 minutes
- **Memory Issues**: Monitor for large dataset processing
- **Rate Limits**: OpenAI API has request limits per minute
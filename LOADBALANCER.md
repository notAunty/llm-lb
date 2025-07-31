# Load Balancing LM Proxy Server 🔄

This enhanced version of the LM Proxy server adds **load balancing capabilities** with support for multiple API keys and custom endpoints, while maintaining full compatibility with the original `openai_anthropic.py` functionality.

## Features 🚀

- **🔑 Multiple API Key Support**: Round-robin load balancing across multiple OpenAI and Anthropic API keys
- **🌐 Custom Endpoint Mapping**: Configure custom endpoints for different services
- **⚖️ Automatic Load Distribution**: Intelligent request distribution across available resources
- **📊 Health Monitoring**: Built-in health checks and statistics endpoints
- **🔧 Zero-Downtime Configuration**: Hot-reload configuration without restarting
- **📈 Production Ready**: Thread-safe round-robin implementation with proper error handling

## Quick Start 🏁

### 1. Configure Environment Variables

Copy the example configuration:
```bash
cp env.loadbalancer.example .env
```

Edit `.env` with your API keys and endpoints:

```bash
# Multiple API Keys (comma-separated)
OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2"

# Custom Endpoints (JSON format)
SERVICE_ENDPOINTS={
  "openai": [
    "https://api.openai.com/v1",
    "https://your-proxy-1.com/v1",
    "https://your-proxy-2.com/v1"
  ],
  "anthropic": [
    "https://api.anthropic.com/v1",
    "https://your-anthropic-proxy.com/v1"
  ]
}
```

### 2. Run the Load Balancer

```bash
# Development mode with auto-reload
uvicorn server:app --reload --host 0.0.0.0 --port 8082

# Production mode
uvicorn server:app --host 0.0.0.0 --port 8082 --workers 4
```

### 3. Test the Load Balancer

```bash
# Check status
curl http://localhost:8082/

# Health check
curl http://localhost:8082/health

# View statistics
curl http://localhost:8082/stats
```

## Configuration 🔧

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEYS` | Comma-separated OpenAI API keys | `"key1,key2,key3"` |
| `ANTHROPIC_API_KEYS` | Comma-separated Anthropic API keys | `"key1,key2"` |
| `SERVICE_ENDPOINTS` | JSON mapping of services to endpoints | See below |
| `OPENAI_API_KEY` | Single OpenAI key (fallback) | `"sk-..."` |
| `ANTHROPIC_API_KEY` | Single Anthropic key (fallback) | `"sk-ant-..."` |

### Service Endpoints Configuration

The `SERVICE_ENDPOINTS` variable accepts a JSON object mapping service names to arrays of endpoint URLs:

```json
{
  "openai": [
    "https://api.openai.com/v1",
    "https://your-openai-proxy-1.com/v1",
    "https://your-openai-proxy-2.com/v1"
  ],
  "anthropic": [
    "https://api.anthropic.com/v1",
    "https://your-anthropic-proxy.com/v1"
  ]
}
```

## Load Balancing Strategy 📊

### Round-Robin Algorithm

The load balancer uses a **thread-safe round-robin algorithm** to distribute requests:

1. **API Key Rotation**: Each request cycles to the next available API key
2. **Endpoint Rotation**: Each request cycles to the next available endpoint
3. **Service Detection**: Automatically routes requests based on model prefixes
4. **Fallback Support**: Falls back to single keys if multiple keys aren't configured

### Request Flow

```
Client Request → Load Balancer → API Key Selection → Endpoint Selection → Service Call
     ↓              ↓                   ↓                 ↓              ↓
   Model: gpt-4   Round-Robin      Next OpenAI Key   Next OpenAI URL   OpenAI API
```

## Monitoring & Observability 📈

### Health Check Endpoint

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1704067200,
  "load_balancer": {
    "openai_keys_configured": 3,
    "anthropic_keys_configured": 2,
    "endpoints": {
      "openai": ["https://api.openai.com/v1", "..."],
      "anthropic": ["https://api.anthropic.com/v1"]
    }
  }
}
```

### Statistics Endpoint

```bash
GET /stats
```

Response:
```json
{
  "load_balancer_stats": {
    "openai": {
      "keys_count": 3,
      "endpoints": ["https://api.openai.com/v1", "..."]
    },
    "anthropic": {
      "keys_count": 2,
      "endpoints": ["https://api.anthropic.com/v1"]
    }
  }
}
```

## Usage Examples 🔍

### Basic Setup with Multiple Keys

```bash
# .env file
OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2"
```

### Advanced Setup with Custom Endpoints

```bash
# .env file
OPENAI_API_KEYS="sk-key1,sk-key2"
ANTHROPIC_API_KEYS="sk-ant-key1"

SERVICE_ENDPOINTS='{
  "openai": [
    "https://api.openai.com/v1",
    "https://oai-proxy-1.yourcompany.com/v1",
    "https://oai-proxy-2.yourcompany.com/v1"
  ],
  "anthropic": [
    "https://api.anthropic.com/v1",
    "https://claude-proxy.yourcompany.com/v1"
  ]
}'
```

### Client Usage

The load balancer is **fully compatible** with existing clients:

```python
import httpx

# Same API as before - load balancing happens transparently
response = httpx.post(
    "http://localhost:8082/v1/messages",
    headers={"x-api-key": "your-key", "anthropic-version": "2023-06-01"},
    json={
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
)
```

## Architecture 🏗️

### Components

1. **LoadBalancerConfig**: Manages API keys and endpoints configuration
2. **Round-Robin Iterators**: Thread-safe cycling through available resources
3. **Service Detection**: Routes requests based on model prefixes
4. **Request Enhancement**: Injects appropriate keys and endpoints into requests
5. **Monitoring**: Health checks and statistics collection

### Thread Safety

All load balancing operations use proper threading locks to ensure:
- ✅ No race conditions in key/endpoint selection
- ✅ Even distribution across concurrent requests
- ✅ Safe configuration updates

### Error Handling

- **Graceful Degradation**: Falls back to single keys if multiple aren't configured
- **Transparent Errors**: Preserves original error messages from upstream APIs
- **Request Logging**: Enhanced logging with load balancing information

## Migration Guide 🔄

### From Original Server

The load balancer is **100% backward compatible**:

1. **Keep existing `.env`**: Single API keys still work
2. **Same API endpoints**: No client changes needed
3. **Add multiple keys**: Just change `OPENAI_API_KEY` to `OPENAI_API_KEYS`
4. **Optional endpoints**: `SERVICE_ENDPOINTS` is optional

### Configuration Evolution

```bash
# Before (still works)
OPENAI_API_KEY="sk-single-key"
ANTHROPIC_API_KEY="sk-ant-single-key"

# After (enhanced)
OPENAI_API_KEYS="sk-key1,sk-key2,sk-key3"
ANTHROPIC_API_KEYS="sk-ant-key1,sk-ant-key2"
SERVICE_ENDPOINTS='{"openai": ["https://api.openai.com/v1"], "anthropic": ["https://api.anthropic.com/v1"]}'
```

## Production Deployment 🚢

### Docker Setup

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8082
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--workers", "4"]
```

### Environment Variables for Production

```bash
# Production .env
OPENAI_API_KEYS="sk-prod-key1,sk-prod-key2,sk-prod-key3"
ANTHROPIC_API_KEYS="sk-ant-prod-key1,sk-ant-prod-key2"

SERVICE_ENDPOINTS='{
  "openai": [
    "https://api.openai.com/v1",
    "https://backup-openai-proxy.yourcompany.com/v1"
  ],
  "anthropic": [
    "https://api.anthropic.com/v1"
  ]
}'
```

### Monitoring in Production

1. **Health Checks**: Monitor `/health` endpoint
2. **Load Distribution**: Check `/stats` for even distribution
3. **Request Logs**: Monitor load balancing logs
4. **Error Rates**: Track API errors across different keys/endpoints

## Troubleshooting 🔧

### Common Issues

1. **No API Keys Configured**
   ```
   Error: No OpenAI API keys configured
   Solution: Set OPENAI_API_KEYS or OPENAI_API_KEY
   ```

2. **Invalid SERVICE_ENDPOINTS JSON**
   ```
   Error: Invalid JSON in SERVICE_ENDPOINTS
   Solution: Validate JSON format and escape quotes properly
   ```

3. **Endpoint Connection Issues**
   ```
   Error: Connection failed to custom endpoint
   Solution: Verify endpoint URLs and network connectivity
   ```

### Debug Mode

Enable detailed logging:

```bash
# Set log level to DEBUG
export LOG_LEVEL=DEBUG
uvicorn server:app --log-level debug
```

### Validation

Test your configuration:

```bash
python server.py --help
```

This shows all available configuration options and validates your setup.

## Contributing 🤝

The load balancer extends the original functionality without modifying the production `openai_anthropic.py` file. All enhancements are in `server.py`.

### Key Design Principles

1. **Non-intrusive**: No changes to existing production code
2. **Backward Compatible**: Works with existing configurations
3. **Thread Safe**: Proper concurrency handling
4. **Observable**: Built-in monitoring and logging
5. **Extensible**: Easy to add new services or strategies

## License 📝

Same license as the original project. 

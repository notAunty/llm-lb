# LLM API Load Balancer

A load balancer that provides both OpenAI and Anthropic compatible REST endpoints while routing requests exclusively to OpenAI-compatible providers. This enables you to use Anthropic's API format while leveraging multiple OpenAI-compatible providers like OpenRouter, Fireworks.AI, Together AI, Nvidia NIM, Groq Cloud, or any other OpenAI-compatible service.

## Description

The LLM Load Balancer acts as a transparent proxy that:

- **Accepts both OpenAI and Anthropic API formats** - Use your preferred API format
- **Routes to OpenAI-compatible providers only** - Leverage the broad ecosystem of OpenAI-compatible services
- **Intelligent load balancing** - Supports both round-robin and priority-based routing with automatic failover
- **Streaming support** - Full streaming capability for both OpenAI and Anthropic endpoints
- **Model mapping** - Configure common model names that map to provider-specific model identifiers
- **Rate limit handling** - Automatic retry on 429 errors in priority-based mode
- **Global model configuration** - Set default parameters like temperature, top_p, max_tokens per model

## Quick Start

### Prerequisites

- Python 3.10 or higher
- API keys for your chosen providers

### Setup

1. **Clone and navigate to the project:**
   ```bash
   git clone https://github.com/notAunty/llm-lb.git
   cd llm-lb
   ```

2. **Install dependencies:**
   ```bash
   curl -fsSL https://astral.sh/uv/install.sh | sh
   ```

3. **Create configuration file:**
   ```bash
   cp config.yaml.example config.yaml
   ```

4. **Set up environment variables:**
   ```bash
   export OPENROUTER_API_KEY="your_openrouter_key"
   export NVIDIA_API_KEY="your_nvidia_key"
   # Add other provider API keys as needed
   ```

5. **Edit `config.yaml`** with your providers and models

6. **Run the server:**
   ```bash
   uv run server.py # -c config.yaml -p 8080 (optional)
   ```
   

## Config File Syntax

The configuration file uses YAML format with the following structure:

```yaml
# Load balancing mode: "round-robin" or "priority-based"
mode: priority-based

# Global model configurations
models:
  kimi-k2-instruct:
    temperature: 0.6
  claude-sonnet-4:
    top_p: 0.5
    temperature: 0.7
    max_tokens: 4096

# Provider configurations (in priority order for priority-based mode)
providers:
  - name: OpenRouter Free
    apiKeyEnvVar: OPENROUTER_API_KEY
    baseUrl: https://openrouter.ai/api/v1
    models:
      kimi-k2-instruct: moonshotai/kimi-k2:free

  - name: OpenRouter
    apiKeyEnvVar: OPENROUTER_API_KEY
    baseUrl: https://openrouter.ai/api/v1
    models:
      kimi-k2-instruct: moonshotai/kimi-k2
      claude-sonnet-4: anthropic/claude-3.5-sonnet
```

### Configuration Fields

#### Global Settings
- `mode`: Load balancing strategy (`round-robin` | `priority-based`)
- `models`: Global model configurations with default parameters

#### Provider Settings
- `name`: Human-readable provider name
- `apiKeyEnvVar`: Environment variable containing the API key
- `baseUrl`: Provider's base URL (without `/chat/completions`)
- `models`: Mapping of common model names to provider-specific model identifiers

## Supported Endpoints

### OpenAI Compatible Endpoints
- `POST /v1/chat/completions` - Chat completions with streaming support
- `GET /v1/models` - List available models

### Anthropic Compatible Endpoints  
- `POST /v1/messages` - Messages endpoint with streaming support
- `POST /v1/messages/count_tokens` - Token counting

### Health & Monitoring
- `GET /` - Get server status & configurations
- `GET /health` - Get health status

## How It Works

### Load Balancing Modes

**Round-Robin Mode:**
- Distributes requests evenly across all available providers
- Cycles through providers in order for each request
- Continues to next provider if current one fails

**Priority-Based Mode:**
- Always tries providers in the order they appear in the configuration
- Only moves to next provider on failure or rate limiting (HTTP 429)
- Automatically retries with next provider on rate limit errors
- Ideal for cost optimization (cheaper providers first)

### Request Flow

1. **Request received** - Client sends OpenAI or Anthropic format request
2. **Format conversion** - Anthropic requests converted to OpenAI format internally
3. **Model configuration** - Global model settings applied
4. **Provider selection** - Load balancer selects provider based on mode
5. **Model mapping** - Common model name mapped to provider-specific name
6. **Request forwarding** - Request sent to selected provider
7. **Response handling** - Response converted back to original format if needed
8. **Streaming** - Real-time streaming for compatible requests

### Failure Handling

- **Connection errors**: Automatic retry with next available provider
- **Rate limiting (429)**: Immediate retry with next provider (priority mode)
- **Invalid responses**: Logged and retried with next provider
- **All providers failed**: Returns 503 Service Unavailable

## Use Cases

### Multi-Provider Redundancy
Ensure high availability by configuring multiple providers as fallbacks:
```yaml
mode: priority-based
providers:
  - name: Primary Provider
    # ... configuration
  - name: Backup Provider  
    # ... configuration
```

### Cost Optimization
Use priority-based mode to prefer cheaper providers:
```yaml
mode: priority-based
providers:
  - name: Cheap Provider
    # ... lower cost provider first
  - name: Premium Provider
    # ... higher cost provider as fallback
```

### Anthropic API Compatibility
Use Anthropic's API format while accessing OpenAI-compatible providers:
```python
import httpx

response = httpx.post("http://localhost:8080/v1/messages", json={
    "model": "claude-sonnet-4",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello!"}
    ]
})
```

### Load Distribution  
Distribute load evenly across multiple API keys/providers:
```yaml
mode: round-robin
providers:
  - name: Provider A
  - name: Provider B  
  - name: Provider C
```

## Contributions

Contributions are welcome! Please feel free to submit a Pull Request. 🎁

## References

- [Anthropic API Proxy by 1rgs](https://github.com/1rgs/claude-code-proxy) - My inspiration for this project
- [LiteLLM Documentation](https://docs.litellm.ai/)

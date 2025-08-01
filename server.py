from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import logging
import json
import yaml
import os
import asyncio
import httpx
import time
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import uuid
from datetime import datetime
import sys

# Import the existing proxy functionality
from openai_anthropic import (
    MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse,
    Message, Tool, Usage, ContentBlockText, ContentBlockToolUse,
    log_request_beautifully, Colors
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Quiet uvicorn logs
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

app = FastAPI(title="Multi-Provider LLM Load Balancer")

class ProviderConfig:
    def __init__(self, name: str, api_key_env_var: str, base_url: str, models: Dict[str, str]):
        self.name = name
        self.api_key_env_var = api_key_env_var
        self.base_url = base_url
        self.models = models
        self.api_key = os.environ.get(api_key_env_var)
        self.last_request_time = 0
        self.request_count = 0
        
    def get_mapped_model(self, common_name: str) -> Optional[str]:
        """Get provider-specific model name for a common model name."""
        return self.models.get(common_name)
    
    def has_model(self, common_name: str) -> bool:
        """Check if provider supports a given model."""
        return common_name in self.models

class LoadBalancerConfig:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.mode = "priority-based"  # Default
        self.models = {}  # Global model configs
        self.providers = []
        self.load_config()
        
    def load_config(self):
        """Load configuration from YAML file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                self.mode = config.get('mode', 'priority-based')
                self.models = config.get('models', {})
                
                # Load providers
                self.providers = []
                for provider_config in config.get('providers', []):
                    provider = ProviderConfig(
                        name=provider_config['name'],
                        api_key_env_var=provider_config['apiKeyEnvVar'],
                        base_url=provider_config['baseUrl'],
                        models=provider_config.get('models', {})
                    )
                    
                    if provider.api_key:
                        self.providers.append(provider)
                        logger.info(f"✅ Loaded provider: {provider.name} with {len(provider.models)} models")
                    else:
                        logger.warning(f"⚠️ Skipping provider {provider.name}: API key not found in env var {provider.api_key_env_var}")
                        
                logger.info(f"🔧 Load balancer mode: {self.mode}")
                logger.info(f"📋 Global model configs: {len(self.models)}")
                logger.info(f"🌐 Active providers: {len(self.providers)}")
            else:
                logger.warning(f"Config file {self.config_path} not found, using defaults")
                self.create_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.create_default_config()
    
    def create_default_config(self):
        """Create a default configuration."""
        logger.info("Creating default configuration...")
        self.providers = []
        
        # Check for OpenAI API key
        if os.environ.get("OPENAI_API_KEY"):
            openai_provider = ProviderConfig(
                name="OpenAI",
                api_key_env_var="OPENAI_API_KEY",
                base_url="https://api.openai.com/v1",
                models={
                    "gpt-4o": "gpt-4o",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt-4-turbo": "gpt-4-turbo",
                    "claude-sonnet-4": "gpt-4o",  # Fallback mapping
                }
            )
            self.providers.append(openai_provider)
            logger.info("✅ Added default OpenAI provider")
        
        # Check for Anthropic API key
        if os.environ.get("ANTHROPIC_API_KEY"):
            anthropic_provider = ProviderConfig(
                name="Anthropic",
                api_key_env_var="ANTHROPIC_API_KEY",
                base_url="https://api.anthropic.com",
                models={
                    "claude-sonnet-4": "claude-3-5-sonnet-20241022",
                    "claude-haiku": "claude-3-haiku-20240307",
                    "claude-opus": "claude-3-opus-20240229",
                }
            )
            self.providers.append(anthropic_provider)
            logger.info("✅ Added default Anthropic provider")
    
    def get_providers_for_model(self, model_name: str) -> List[ProviderConfig]:
        """Get providers that support a given model, in order based on mode."""
        providers = [p for p in self.providers if p.has_model(model_name)]
        
        if self.mode == "round-robin":
            # For round-robin, we'll rotate the order based on request count
            if providers:
                # Simple round-robin based on total requests across all providers
                total_requests = sum(p.request_count for p in providers)
                rotation = total_requests % len(providers)
                providers = providers[rotation:] + providers[:rotation]
        
        # For priority-based, keep the order as-is (first in config = highest priority)
        return providers

# Global config instance
config = LoadBalancerConfig()

class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class OpenAITool(BaseModel):
    type: str = "function"
    function: Dict[str, Any]

class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    tools: Optional[List[OpenAITool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

def convert_openai_to_anthropic(openai_request: OpenAIChatRequest) -> MessagesRequest:
    """Convert OpenAI API request to Anthropic format."""
    messages = []
    system_message = None
    
    # Process messages and extract system message
    for msg in openai_request.messages:
        if msg.role == "system":
            system_message = msg.content if isinstance(msg.content, str) else str(msg.content)
        else:
            # Convert message content
            if isinstance(msg.content, str):
                content = msg.content
            else:
                # Handle complex content (images, etc.)
                content = []
                for item in msg.content:
                    if item.get("type") == "text":
                        content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image_url":
                        # Convert to Anthropic image format
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",  # Default
                                "data": item.get("image_url", {}).get("url", "").replace("data:image/jpeg;base64,", "")
                            }
                        })
            
            # Handle tool calls
            if msg.tool_calls:
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}] if content else []
                for tool_call in msg.tool_calls:
                    content.append({
                        "type": "tool_use",
                        "id": tool_call.get("id", str(uuid.uuid4())),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
                    })
            
            # Handle tool call responses
            if msg.tool_call_id:
                if isinstance(content, str):
                    content = [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": content
                    }]
            
            messages.append(Message(role=msg.role, content=content))
    
    # Convert tools
    tools = None
    if openai_request.tools:
        tools = []
        for tool in openai_request.tools:
            tools.append(Tool(
                name=tool.function["name"],
                description=tool.function.get("description", ""),
                input_schema=tool.function.get("parameters", {})
            ))
    
    # Convert tool_choice
    tool_choice = None
    if openai_request.tool_choice:
        if openai_request.tool_choice == "auto":
            tool_choice = {"type": "auto"}
        elif openai_request.tool_choice == "none":
            tool_choice = {"type": "none"}
        elif isinstance(openai_request.tool_choice, dict):
            tool_choice = {
                "type": "tool",
                "name": openai_request.tool_choice.get("function", {}).get("name", "")
            }
    
    return MessagesRequest(
        model=openai_request.model,
        max_tokens=openai_request.max_tokens or 4096,
        messages=messages,
        system=system_message,
        temperature=openai_request.temperature,
        top_p=openai_request.top_p,
        tools=tools,
        tool_choice=tool_choice,
        stream=openai_request.stream
    )

def convert_anthropic_to_openai(anthropic_response: MessagesResponse) -> Dict[str, Any]:
    """Convert Anthropic API response to OpenAI format."""
    choices = []
    
    # Process content blocks
    content = ""
    tool_calls = []
    
    for block in anthropic_response.content:
        if block.type == "text":
            content += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "function",
                "function": {
                    "name": block.name,
                    "arguments": json.dumps(block.input)
                }
            })
    
    # Create message
    message = {
        "role": "assistant",
        "content": content if content or not tool_calls else None
    }
    
    if tool_calls:
        message["tool_calls"] = tool_calls
    
    # Map stop reason
    finish_reason = "stop"
    if anthropic_response.stop_reason == "max_tokens":
        finish_reason = "length"
    elif anthropic_response.stop_reason == "tool_use":
        finish_reason = "tool_calls"
    
    choices.append({
        "index": 0,
        "message": message,
        "finish_reason": finish_reason
    })
    
    return {
        "id": anthropic_response.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_response.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": anthropic_response.usage.input_tokens,
            "completion_tokens": anthropic_response.usage.output_tokens,
            "total_tokens": anthropic_response.usage.input_tokens + anthropic_response.usage.output_tokens
        }
    }

async def make_provider_request(provider: ProviderConfig, request_data: Dict[str, Any], 
                              is_anthropic_format: bool = True, is_streaming: bool = False) -> Union[Dict[str, Any], AsyncIterable]:
    """Make a request to a specific provider."""
    headers = {
        "Content-Type": "application/json",
    }
    
    # Determine endpoint and headers based on provider
    if "anthropic.com" in provider.base_url:
        endpoint = f"{provider.base_url}/v1/messages"
        headers["x-api-key"] = provider.api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        # OpenAI-compatible endpoint
        endpoint = f"{provider.base_url}/chat/completions"
        headers["Authorization"] = f"Bearer {provider.api_key}"
    
    # Apply global model configuration
    model_name = request_data.get("model", "")
    if model_name in config.models:
        model_config = config.models[model_name]
        for key, value in model_config.items():
            if key not in request_data or request_data[key] is None:
                request_data[key] = value
    
    # Map model name
    mapped_model = provider.get_mapped_model(model_name)
    if mapped_model:
        request_data["model"] = mapped_model
    
    # Convert request format if needed
    if "anthropic.com" in provider.base_url and not is_anthropic_format:
        # Need to convert OpenAI format to Anthropic
        openai_req = OpenAIChatRequest(**request_data)
        anthropic_req = convert_openai_to_anthropic(openai_req)
        request_data = anthropic_req.dict(exclude_none=True)
    elif "anthropic.com" not in provider.base_url and is_anthropic_format:
        # Need to convert Anthropic format to OpenAI
        # This is more complex and would require implementing the reverse conversion
        # For now, we'll assume providers match the request format
        pass
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if is_streaming:
                request_data["stream"] = True
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=request_data,
                    timeout=120.0
                )
                response.raise_for_status()
                return response.aiter_lines()
            else:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=request_data,
                    timeout=120.0
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            else:
                raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

async def try_providers(model_name: str, request_data: Dict[str, Any], 
                       is_anthropic_format: bool = True, is_streaming: bool = False):
    """Try providers in order until one succeeds."""
    providers = config.get_providers_for_model(model_name)
    
    if not providers:
        raise HTTPException(status_code=404, detail=f"No providers available for model: {model_name}")
    
    last_error = None
    
    for i, provider in enumerate(providers):
        try:
            logger.info(f"🔄 Trying provider {provider.name} for model {model_name} (attempt {i+1}/{len(providers)})")
            
            # Update request count
            provider.request_count += 1
            provider.last_request_time = time.time()
            
            result = await make_provider_request(provider, request_data.copy(), is_anthropic_format, is_streaming)
            
            logger.info(f"✅ Success with provider {provider.name}")
            return result, provider
            
        except HTTPException as e:
            last_error = e
            if e.status_code == 429 and i < len(providers) - 1:
                # Rate limit hit, try next provider
                logger.warning(f"⚠️ Rate limit hit on {provider.name}, trying next provider...")
                continue
            elif i == len(providers) - 1:
                # Last provider, re-raise the error
                logger.error(f"❌ All providers failed. Last error: {e.detail}")
                raise e
            else:
                # Other error, try next provider
                logger.warning(f"⚠️ Error with {provider.name}: {e.detail}, trying next provider...")
                continue
        except Exception as e:
            last_error = e
            logger.error(f"❌ Unexpected error with {provider.name}: {str(e)}")
            if i == len(providers) - 1:
                raise HTTPException(status_code=500, detail=str(e))
            continue
    
    # If we get here, all providers failed
    raise HTTPException(status_code=500, detail=f"All providers failed. Last error: {str(last_error)}")

@app.post("/v1/messages")
async def anthropic_messages(request: MessagesRequest, raw_request: Request):
    """Anthropic-compatible messages endpoint."""
    try:
        request_data = request.dict(exclude_none=True)
        model_name = request.model
        
        # Log the request
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", model_name)
        
        if request.stream:
            result, provider = await try_providers(model_name, request_data, is_anthropic_format=True, is_streaming=True)
            
            log_request_beautifully(
                "POST", "/v1/messages", original_model, provider.name,
                len(request.messages), len(request.tools) if request.tools else 0, 200
            )
            
            async def stream_response():
                async for line in result:
                    if line:
                        yield f"{line}\n"
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            result, provider = await try_providers(model_name, request_data, is_anthropic_format=True, is_streaming=False)
            
            log_request_beautifully(
                "POST", "/v1/messages", original_model, provider.name,
                len(request.messages), len(request.tools) if request.tools else 0, 200
            )
            
            return result
            
    except Exception as e:
        logger.error(f"Error in anthropic_messages: {str(e)}")
        raise e

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint."""
    try:
        model_name = request.model
        
        # Convert to common format for provider routing
        request_data = request.dict(exclude_none=True)
        
        if request.stream:
            result, provider = await try_providers(model_name, request_data, is_anthropic_format=False, is_streaming=True)
            
            log_request_beautifully(
                "POST", "/v1/chat/completions", model_name, provider.name,
                len(request.messages), len(request.tools) if request.tools else 0, 200
            )
            
            async def stream_response():
                async for line in result:
                    if line:
                        yield f"{line}\n"
            
            return StreamingResponse(stream_response(), media_type="text/event-stream")
        else:
            result, provider = await try_providers(model_name, request_data, is_anthropic_format=False, is_streaming=False)
            
            log_request_beautifully(
                "POST", "/v1/chat/completions", model_name, provider.name,
                len(request.messages), len(request.tools) if request.tools else 0, 200
            )
            
            # Convert response if needed
            if "anthropic.com" in provider.base_url:
                # Convert Anthropic response to OpenAI format
                anthropic_response = MessagesResponse(**result)
                result = convert_anthropic_to_openai(anthropic_response)
            
            return result
            
    except Exception as e:
        raise e

@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest):
    """Token counting endpoint (Anthropic format)."""
    # Use first available provider for token counting
    providers = config.get_providers_for_model(request.model)
    if not providers:
        raise HTTPException(status_code=404, detail=f"No providers available for model: {request.model}")
    
    # For token counting, we'll use a simple estimation
    # In a real implementation, you'd want to use the provider's token counting API
    total_chars = 0
    for message in request.messages:
        if isinstance(message.content, str):
            total_chars += len(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if hasattr(block, 'text'):
                    total_chars += len(block.text)
    
    # Rough estimation: 1 token ≈ 4 characters
    estimated_tokens = total_chars // 4
    
    return TokenCountResponse(input_tokens=estimated_tokens)

@app.get("/v1/models")
async def list_models():
    """List available models across all providers."""
    models = []
    for provider in config.providers:
        for common_name, provider_model in provider.models.items():
            models.append({
                "id": common_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": provider.name.lower(),
                "provider": provider.name,
                "provider_model": provider_model
            })
    
    return {"object": "list", "data": models}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "providers": len(config.providers),
        "mode": config.mode,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Multi-Provider LLM Load Balancer",
        "version": "1.0.0",
        "providers": [{"name": p.name, "models": len(p.models)} for p in config.providers],
        "mode": config.mode,
        "endpoints": {
            "anthropic": "/v1/messages",
            "openai": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }

def print_help():
    print("Multi-Provider LLM Load Balancer")
    print("Usage: python server.py [--config config.yaml] [--port 8082]")
    print("\nEnvironment variables:")
    print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. - API keys for providers")
    print("\nEndpoints:")
    print("  POST /v1/messages - Anthropic-compatible endpoint")
    print("  POST /v1/chat/completions - OpenAI-compatible endpoint")
    print("  GET /v1/models - List available models")
    print("  GET /health - Health check")

def print_startup_banner(port):
    print(f"{Colors.BOLD}{Colors.CYAN}🚀 Multi-Provider LLM Load Balancer{Colors.RESET}")
    print(f"{Colors.GREEN}📡 Starting server on http://0.0.0.0:{port}{Colors.RESET}")
    print(f"{Colors.BLUE}🔧 Mode: {config.mode}{Colors.RESET}")
    print(f"{Colors.MAGENTA}🌐 Active providers: {len(config.providers)}{Colors.RESET}")
    for provider in config.providers:
        print(f"  • {Colors.YELLOW}{provider.name}{Colors.RESET}: {len(provider.models)} models")
    print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "--help":
            print_help()
            sys.exit(0)
        elif sys.argv[1] == "--config" and len(sys.argv) > 2:
            config = LoadBalancerConfig(sys.argv[2])
    
    port = 8082
    if "--port" in sys.argv:
        port_idx = sys.argv.index("--port")
        if port_idx + 1 < len(sys.argv):
            port = int(sys.argv[port_idx + 1])
    
    print_startup_banner(port)
    
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="error")

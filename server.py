from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
import yaml
import httpx
import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Literal
import argparse
import sys

# Import useful functions from openai_anthropic.py
from openai_anthropic import (
    MessagesRequest,
    Message,
    Tool,
    TokenCountRequest,
    TokenCountResponse,
    MessagesResponse,
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
    parse_tool_result_content,
    clean_gemini_schema,
    Colors,
    log_request_beautifully
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load Balancer for OpenAI-compatible LLM providers')
parser.add_argument('-c', '--config', type=str, default='config.yaml',
                    help='Path to configuration file (default: config.yaml)')
parser.add_argument('-p', '--port', type=int, default=None,
                    help='Port to run the server on (default: 11434 or PORT env var)')

args = parser.parse_args()

# Load configuration
CONFIG_FILE = args.config

def load_config():
    """Load configuration from YAML file"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {CONFIG_FILE} not found")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

# Global configuration
config = load_config()

# Provider state for round-robin
class ProviderState:
    def __init__(self):
        self.current_index = 0
        self.lock = asyncio.Lock()
    
    async def get_next_index(self, num_providers: int) -> int:
        async with self.lock:
            index = self.current_index
            self.current_index = (self.current_index + 1) % num_providers
            return index

provider_state = ProviderState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting load balancer in {config['mode']} mode")
    logger.info(f"Loaded {len(config['providers'])} providers")
    yield
    # Shutdown
    logger.info("Shutting down load balancer")

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI format models
class OpenAIMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None]
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

class OpenAIRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

def get_provider_for_model(model_name: str):
    """Find providers that support the given model"""
    providers = []
    for provider in config['providers']:
        if model_name in provider['models']:
            providers.append(provider)
    return providers

def apply_model_config(request: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Apply model-specific configuration from YAML"""
    if 'models' in config and model_name in config['models']:
        model_config = config['models'][model_name]
        for key, value in model_config.items():
            # Only apply if not already set in request
            if key not in request or request[key] is None:
                request[key] = value
    return request

async def make_openai_request_streaming(provider: Dict[str, Any], request: Dict[str, Any]):
    """Make a streaming request to an OpenAI-compatible provider"""
    # Get API key
    api_key_var = provider['apiKeyEnvVar']
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise ValueError(f"API key not found for provider {provider['name']}: {api_key_var}")
    
    # Map model name
    original_model = request['model']
    if original_model in provider['models']:
        request['model'] = provider['models'][original_model]
    else:
        raise ValueError(f"Model {original_model} not supported by provider {provider['name']}")
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make request
    url = f"{provider['baseUrl']}/chat/completions"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream('POST', url, json=request, headers=headers) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Error from {provider['name']}: {error_text.decode()}",
                    request=response.request,
                    response=response
                )
            
            async for chunk in response.aiter_lines():
                if chunk:
                    yield chunk + "\n"

async def make_openai_request(provider: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
    """Make a non-streaming request to an OpenAI-compatible provider"""
    # Get API key
    api_key_var = provider['apiKeyEnvVar']
    api_key = os.environ.get(api_key_var)
    if not api_key:
        raise ValueError(f"API key not found for provider {provider['name']}: {api_key_var}")
    
    # Map model name
    original_model = request['model']
    if original_model in provider['models']:
        request['model'] = provider['models'][original_model]
    else:
        raise ValueError(f"Model {original_model} not supported by provider {provider['name']}")
    
    # Prepare headers
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Make request
    url = f"{provider['baseUrl']}/chat/completions"
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=request, headers=headers)
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Error from {provider['name']}: {response.text}",
                request=response.request,
                response=response
            )
        return response.json()

async def try_providers(model_name: str, request: Dict[str, Any], is_streaming: bool = False):
    """Try providers based on configured mode"""
    providers = get_provider_for_model(model_name)
    if not providers:
        raise HTTPException(status_code=400, detail=f"No providers support model: {model_name}")
    
    if config['mode'] == 'round-robin':
        # Round-robin mode
        start_index = await provider_state.get_next_index(len(providers))
        for i in range(len(providers)):
            provider_index = (start_index + i) % len(providers)
            provider = providers[provider_index]
            try:
                logger.info(f"Trying provider {provider['name']} for model {model_name}")
                if is_streaming:
                    # For streaming, yield chunks directly
                    async for chunk in make_openai_request_streaming(provider, request.copy()):
                        yield chunk
                    return
                else:
                    # For non-streaming, yield the result
                    result = await make_openai_request(provider, request.copy())
                    yield result
                    return
            except Exception as e:
                logger.warning(f"Provider {provider['name']} failed: {str(e)}")
                if i == len(providers) - 1:
                    raise HTTPException(status_code=503, detail="All providers failed")
    elif config['mode'] == 'priority-based':
        # Priority-based mode
        for i, provider in enumerate(providers):
            try:
                logger.info(f"Trying provider {provider['name']} for model {model_name}")
                if is_streaming:
                    # For streaming, yield chunks directly
                    async for chunk in make_openai_request_streaming(provider, request.copy()):
                        yield chunk
                    return
                else:
                    # For non-streaming, yield the result
                    result = await make_openai_request(provider, request.copy())
                    yield result
                    return
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 or "rate" in str(e).lower():
                    logger.warning(f"Rate limit hit for provider {provider['name']}")
                error_detail = f"HTTP {e.response.status_code}: {e.response.text if hasattr(e.response, 'text') else str(e)}"
                logger.warning(f"Provider {provider['name']} failed with {error_detail}")
                if i == len(providers) - 1:
                    raise HTTPException(status_code=503, detail=f"All providers failed. Last error: {error_detail}")
                continue
            except Exception as e:
                logger.error(f"Provider {provider['name']} failed with error: {str(e)}")
                if i == len(providers) - 1:
                    raise HTTPException(status_code=503, detail=f"All providers failed. Last error: {str(e)}")
                continue
    else:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {config['mode']}")

def convert_anthropic_to_openai(anthropic_request: MessagesRequest) -> Dict[str, Any]:
    """Convert Anthropic format to OpenAI format"""
    # First use the existing converter to get LiteLLM format
    litellm_format = convert_anthropic_to_litellm(anthropic_request)
    
    # Then adapt for direct OpenAI API format
    openai_request = {
        "model": anthropic_request.model,
        "messages": litellm_format["messages"],
        "stream": anthropic_request.stream,
    }
    
    # Add optional parameters
    if anthropic_request.max_tokens:
        openai_request["max_tokens"] = anthropic_request.max_tokens
    if anthropic_request.temperature is not None:
        openai_request["temperature"] = anthropic_request.temperature
    if anthropic_request.top_p is not None:
        openai_request["top_p"] = anthropic_request.top_p
    if anthropic_request.stop_sequences:
        openai_request["stop"] = anthropic_request.stop_sequences
    if "tools" in litellm_format:
        openai_request["tools"] = litellm_format["tools"]
    if "tool_choice" in litellm_format:
        openai_request["tool_choice"] = litellm_format["tool_choice"]
    
    return openai_request

def convert_openai_to_anthropic(openai_response: Dict[str, Any], original_request: MessagesRequest) -> MessagesResponse:
    """Convert OpenAI response to Anthropic format"""
    # OpenAI response has a similar structure to LiteLLM, so we can reuse the converter
    return convert_litellm_to_anthropic(openai_response, original_request)

async def handle_anthropic_streaming(response_generator, original_request: MessagesRequest):
    """Convert OpenAI SSE format to Anthropic SSE format"""
    # Send initial message_start
    message_id = f"msg_{int(time.time() * 1000)}"
    message_start = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": original_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0
            }
        }
    }
    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
    
    # Start content block
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    
    accumulated_content = ""
    
    async for line in response_generator:
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            
            try:
                data = json.loads(data_str)
                if "choices" in data and len(data["choices"]) > 0:
                    delta = data["choices"][0].get("delta", {})
                    
                    # Handle content delta
                    if "content" in delta and delta["content"]:
                        accumulated_content += delta["content"]
                        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': delta['content']}})}\n\n"
                    
                    # Handle finish
                    if data["choices"][0].get("finish_reason"):
                        finish_reason = data["choices"][0]["finish_reason"]
                        stop_reason = "end_turn"
                        if finish_reason == "length":
                            stop_reason = "max_tokens"
                        elif finish_reason == "stop":
                            stop_reason = "stop_sequence"
                        
                        # Close content block
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
                        
                        # Send message_delta with stop reason
                        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': len(accumulated_content.split())}})}\n\n"
                        
                        # Send message_stop
                        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                        
            except json.JSONDecodeError:
                continue

# Anthropic endpoints
@app.post("/v1/messages")
async def anthropic_messages(request: MessagesRequest, raw_request: Request):
    """Anthropic-compatible messages endpoint"""
    try:
        # Convert to OpenAI format
        openai_request = convert_anthropic_to_openai(request)
        
        # Apply model config
        openai_request = apply_model_config(openai_request, request.model)
        
        # Log request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            request.model,
            request.model,
            len(request.messages),
            num_tools,
            200
        )
        
        if request.stream:
            # Handle streaming
            async def stream_wrapper():
                try:
                    async for chunk in try_providers(request.model, openai_request, is_streaming=True):
                        yield chunk
                except HTTPException as e:
                    # Handle HTTP exceptions gracefully in streaming
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'invalid_request_error', 'message': e.detail}})}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': {'type': 'api_error', 'message': str(e)}})}\n\n"
            
            return StreamingResponse(
                handle_anthropic_streaming(stream_wrapper(), request),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming request - let HTTPException bubble up to be handled by FastAPI
            result = None
            async for response in try_providers(request.model, openai_request):
                result = response
                break
            
            # Convert back to Anthropic format
            anthropic_response = convert_openai_to_anthropic(result, request)
            return anthropic_response
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in Anthropic endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(request: TokenCountRequest):
    """Anthropic-compatible token counting endpoint"""
    # For simplicity, return a rough estimate
    # In production, you'd want to use proper tokenization
    total_chars = 0
    
    # Count system message
    if request.system:
        if isinstance(request.system, str):
            total_chars += len(request.system)
        else:
            for block in request.system:
                if hasattr(block, 'text'):
                    total_chars += len(block.text)
    
    # Count messages
    for msg in request.messages:
        if isinstance(msg.content, str):
            total_chars += len(msg.content)
        else:
            for block in msg.content:
                if hasattr(block, 'text'):
                    total_chars += len(block.text)
    
    # Rough estimate: ~4 chars per token
    estimated_tokens = total_chars // 4
    
    return TokenCountResponse(input_tokens=estimated_tokens)

# OpenAI endpoints
@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIRequest, raw_request: Request):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Convert to dict for processing
        openai_request = request.dict(exclude_unset=True)
        
        # Apply model config
        openai_request = apply_model_config(openai_request, request.model)
        
        # Log request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            request.model,
            request.model,
            len(request.messages),
            num_tools,
            200
        )
        
        if request.stream:
            # Handle streaming
            async def stream_wrapper():
                try:
                    async for chunk in try_providers(request.model, openai_request, is_streaming=True):
                        yield chunk
                except HTTPException as e:
                    yield f"data: {json.dumps({'error': {'message': e.detail, 'type': 'invalid_request_error'}})}\n\n"
                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'api_error'}})}\n\n"
            
            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming request
            result = None
            async for response in try_providers(request.model, openai_request):
                result = response
                break
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    models = set()
    for provider in config['providers']:
        models.update(provider['models'].keys())
    
    return {
        "object": "list",
        "data": [
            {
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "loadbalancer"
            }
            for model in sorted(models)
        ]
    }

@app.get("/")
async def root():
    return {
        "message": "Load Balancer API Server",
        "mode": config['mode'],
        "providers": [p['name'] for p in config['providers']],
        "models": list(set(model for p in config['providers'] for model in p['models'].keys()))
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "mode": config['mode']}

if __name__ == "__main__":
    port = args.port or int(os.environ.get("PORT", 11434))
    logger.info(f"Starting server with config: {CONFIG_FILE} on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

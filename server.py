from fastapi import FastAPI, Request, HTTPException
import uvicorn
import logging
import json
import os
import time
import itertools
import threading
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
import httpx

# Import the existing functionality from production code
from openai_anthropic import (
    MessagesRequest, 
    TokenCountRequest,
    MessagesResponse,
    TokenCountResponse,
    convert_anthropic_to_litellm,
    convert_litellm_to_anthropic,
    handle_streaming,
    log_request_beautifully,
    app as original_app,
    logger
)
import litellm
from fastapi.responses import StreamingResponse

# Load environment variables
load_dotenv()

# Load balancing configuration
class LoadBalancerConfig:
    def __init__(self):
        # Parse multiple OpenAI API keys (comma-separated)
        openai_keys_str = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", ""))
        self.openai_keys = [key.strip() for key in openai_keys_str.split(",") if key.strip()]
        
        # Parse multiple Anthropic API keys (comma-separated) 
        anthropic_keys_str = os.environ.get("ANTHROPIC_API_KEYS", os.environ.get("ANTHROPIC_API_KEY", ""))
        self.anthropic_keys = [key.strip() for key in anthropic_keys_str.split(",") if key.strip()]
        
        # Parse service endpoints (JSON format)
        endpoints_str = os.environ.get("SERVICE_ENDPOINTS", '{}')
        try:
            self.service_endpoints = json.loads(endpoints_str)
        except json.JSONDecodeError:
            self.service_endpoints = {}
            
        # Default endpoints if not specified
        if not self.service_endpoints:
            self.service_endpoints = {
                "openai": ["https://api.openai.com/v1"],
                "anthropic": ["https://api.anthropic.com/v1"]
            }
        
        # Initialize round-robin iterators
        self.openai_key_cycle = itertools.cycle(self.openai_keys) if self.openai_keys else None
        self.anthropic_key_cycle = itertools.cycle(self.anthropic_keys) if self.anthropic_keys else None
        
        # Initialize endpoint cycles for each service
        self.endpoint_cycles = {}
        for service, endpoints in self.service_endpoints.items():
            if endpoints:
                self.endpoint_cycles[service] = itertools.cycle(endpoints)
        
        # Thread locks for round-robin safety
        self.openai_lock = threading.Lock()
        self.anthropic_lock = threading.Lock()
        self.endpoint_locks = {service: threading.Lock() for service in self.service_endpoints.keys()}
        
        logger.info(f"🔄 Load Balancer Initialized:")
        logger.info(f"   📊 OpenAI Keys: {len(self.openai_keys)} configured")
        logger.info(f"   📊 Anthropic Keys: {len(self.anthropic_keys)} configured") 
        logger.info(f"   🌐 Service Endpoints: {json.dumps(self.service_endpoints, indent=2)}")

    def get_next_openai_key(self) -> Optional[str]:
        """Get the next OpenAI API key using round-robin."""
        if not self.openai_key_cycle:
            return None
        with self.openai_lock:
            return next(self.openai_key_cycle)
    
    def get_next_anthropic_key(self) -> Optional[str]:
        """Get the next Anthropic API key using round-robin."""
        if not self.anthropic_key_cycle:
            return None
        with self.anthropic_lock:
            return next(self.anthropic_key_cycle)
    
    def get_next_endpoint(self, service: str) -> Optional[str]:
        """Get the next endpoint for a service using round-robin."""
        if service not in self.endpoint_cycles:
            return None
        with self.endpoint_locks[service]:
            return next(self.endpoint_cycles[service])

# Global load balancer instance
lb_config = LoadBalancerConfig()

# Create new FastAPI app for load balancing
app = FastAPI(title="LM Proxy Load Balancer", description="Load balancing proxy for OpenAI and Anthropic APIs")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests with load balancing info."""
    method = request.method
    path = request.url.path
    
    logger.debug(f"🔄 Load Balanced Request: {method} {path}")
    
    response = await call_next(request)
    return response

def enhance_litellm_request_with_load_balancing(litellm_request: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Enhance the LiteLLM request with load-balanced API keys and endpoints."""
    
    # Determine service type based on model
    if model.startswith("openai/"):
        service = "openai"
        api_key = lb_config.get_next_openai_key()
        base_url = lb_config.get_next_endpoint("openai")
        
        if api_key:
            litellm_request["api_key"] = api_key
            logger.debug(f"🔑 Using OpenAI key: ...{api_key[-8:] if len(api_key) > 8 else api_key}")
        
        if base_url:
            # Ensure the base URL is properly formatted for LiteLLM
            if not base_url.endswith('/'):
                base_url += '/'
            litellm_request["api_base"] = base_url
            logger.debug(f"🌐 Using OpenAI endpoint: {base_url}")
            
    elif model.startswith("anthropic/") or not model.startswith(("openai/", "gemini/")):
        service = "anthropic"
        api_key = lb_config.get_next_anthropic_key()
        base_url = lb_config.get_next_endpoint("anthropic")
        
        if api_key:
            litellm_request["api_key"] = api_key
            logger.debug(f"🔑 Using Anthropic key: ...{api_key[-8:] if len(api_key) > 8 else api_key}")
            
        if base_url:
            if not base_url.endswith('/'):
                base_url += '/'
            litellm_request["api_base"] = base_url
            logger.debug(f"🌐 Using Anthropic endpoint: {base_url}")
    
    return litellm_request

@app.post("/v1/messages")
async def create_message_load_balanced(
    request: MessagesRequest,
    raw_request: Request
):
    """Load-balanced version of the messages endpoint."""
    try:
        # Get request body for logging
        body = await raw_request.body()
        body_json = json.loads(body.decode('utf-8'))
        original_model = body_json.get("model", "unknown")
        
        # Get display name for logging
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        logger.debug(f"🔄 LOAD BALANCED REQUEST: Model={request.model}, Stream={request.stream}")
        
        # Convert Anthropic request to LiteLLM format (using existing function)
        litellm_request = convert_anthropic_to_litellm(request)
        
        # Apply load balancing
        litellm_request = enhance_litellm_request_with_load_balancing(litellm_request, request.model)
        
        # Enhanced request processing for OpenAI models (from existing code)
        if "openai" in litellm_request["model"] and "messages" in litellm_request:
            logger.debug(f"🔧 Processing OpenAI model request: {litellm_request['model']}")
            
            for i, msg in enumerate(litellm_request["messages"]):
                if "content" in msg and isinstance(msg["content"], list):
                    is_only_tool_result = True
                    for block in msg["content"]:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            is_only_tool_result = False
                            break
                    
                    if is_only_tool_result and len(msg["content"]) > 0:
                        logger.debug(f"🛠️ Converting tool_result to text format")
                        all_text = ""
                        for block in msg["content"]:
                            all_text += "Tool Result:\n"
                            result_content = block.get("content", [])
                            
                            if isinstance(result_content, list):
                                for item in result_content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        all_text += item.get("text", "") + "\n"
                                    elif isinstance(item, dict):
                                        try:
                                            item_text = item.get("text", json.dumps(item))
                                            all_text += item_text + "\n"
                                        except:
                                            all_text += str(item) + "\n"
                            elif isinstance(result_content, str):
                                all_text += result_content + "\n"
                            else:
                                try:
                                    all_text += json.dumps(result_content) + "\n"
                                except:
                                    all_text += str(result_content) + "\n"
                        
                        litellm_request["messages"][i]["content"] = all_text.strip() or "..."
                        continue
                
                if "content" in msg and isinstance(msg["content"], list):
                    text_content = ""
                    for block in msg["content"]:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                text_content += block.get("text", "") + "\n"
                            elif block.get("type") == "tool_result":
                                tool_id = block.get("tool_use_id", "unknown")
                                text_content += f"[Tool Result ID: {tool_id}]\n"
                                
                                result_content = block.get("content", [])
                                if isinstance(result_content, list):
                                    for item in result_content:
                                        if isinstance(item, dict) and item.get("type") == "text":
                                            text_content += item.get("text", "") + "\n"
                                        elif isinstance(item, dict):
                                            if "text" in item:
                                                text_content += item.get("text", "") + "\n"
                                            else:
                                                try:
                                                    text_content += json.dumps(item) + "\n"
                                                except:
                                                    text_content += str(item) + "\n"
                                elif isinstance(result_content, str):
                                    text_content += result_content + "\n"
                                else:
                                    try:
                                        text_content += json.dumps(result_content) + "\n"
                                    except:
                                        text_content += str(result_content) + "\n"
                            elif block.get("type") == "tool_use":
                                tool_name = block.get("name", "unknown")
                                tool_id = block.get("id", "unknown")
                                tool_input = json.dumps(block.get("input", {}))
                                text_content += f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}\n\n"
                            elif block.get("type") == "image":
                                text_content += "[Image content - not displayed in text format]\n"
                    
                    if not text_content.strip():
                        text_content = "..."
                    
                    litellm_request["messages"][i]["content"] = text_content.strip()
                elif msg.get("content") is None:
                    litellm_request["messages"][i]["content"] = "..."
                
                # Remove unsupported fields
                for key in list(msg.keys()):
                    if key not in ["role", "content", "name", "tool_call_id", "tool_calls"]:
                        del msg[key]
            
            # Final validation
            for i, msg in enumerate(litellm_request["messages"]):
                if isinstance(msg.get("content"), list):
                    logger.warning(f"⚠️ Message {i} still has list content after processing")
                    litellm_request["messages"][i]["content"] = f"Content as JSON: {json.dumps(msg.get('content'))}"
                elif msg.get("content") is None:
                    litellm_request["messages"][i]["content"] = "..."
        
        # Log the request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST", 
            raw_request.url.path, 
            display_model, 
            litellm_request.get('model'),
            len(litellm_request['messages']),
            num_tools,
            200
        )
        
        # Handle streaming vs non-streaming
        if request.stream:
            logger.debug(f"🌊 Starting streaming response")
            response_generator = await litellm.acompletion(**litellm_request)
            
            return StreamingResponse(
                handle_streaming(response_generator, request),
                media_type="text/event-stream"
            )
        else:
            logger.debug(f"💬 Starting non-streaming response")
            start_time = time.time()
            litellm_response = litellm.completion(**litellm_request)
            logger.debug(f"✅ Response received in {time.time() - start_time:.2f}s")
            
            # Convert response using existing function
            anthropic_response = convert_litellm_to_anthropic(litellm_response, request)
            return anthropic_response
                
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        
        # Enhanced error details
        error_details = {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": error_traceback
        }
        
        # Check for LiteLLM-specific attributes
        for attr in ['message', 'status_code', 'response', 'llm_provider', 'model']:
            if hasattr(e, attr):
                error_details[attr] = getattr(e, attr)
        
        logger.error(f"💥 Load balancer error: {json.dumps(error_details, indent=2)}")
        
        # Format error response
        error_message = f"Load Balancer Error: {str(e)}"
        if 'message' in error_details and error_details['message']:
            error_message += f"\nMessage: {error_details['message']}"
        if 'response' in error_details and error_details['response']:
            error_message += f"\nResponse: {error_details['response']}"
        
        status_code = error_details.get('status_code', 500)
        raise HTTPException(status_code=status_code, detail=error_message)

@app.post("/v1/messages/count_tokens")
async def count_tokens_load_balanced(
    request: TokenCountRequest,
    raw_request: Request
):
    """Load-balanced version of the token counting endpoint."""
    try:
        original_model = request.original_model or request.model
        
        # Get display name for logging
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]
        
        logger.debug(f"🔢 LOAD BALANCED TOKEN COUNT: Model={request.model}")
        
        # Convert the messages using existing function
        from openai_anthropic import MessagesRequest
        converted_request = convert_anthropic_to_litellm(
            MessagesRequest(
                model=request.model,
                max_tokens=100,
                messages=request.messages,
                system=request.system,
                tools=request.tools,
                tool_choice=request.tool_choice,
                thinking=request.thinking
            )
        )
        
        # Apply load balancing
        converted_request = enhance_litellm_request_with_load_balancing(converted_request, request.model)
        
        # Log the request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            converted_request.get('model'),
            len(converted_request['messages']),
            num_tools,
            200
        )
        
        # Count tokens using LiteLLM
        try:
            from litellm import token_counter
            
            token_count = token_counter(
                model=converted_request["model"],
                messages=converted_request["messages"],
            )
            
            return TokenCountResponse(input_tokens=token_count)
            
        except ImportError:
            logger.error("❌ Could not import token_counter from litellm")
            return TokenCountResponse(input_tokens=1000)
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"💥 Token count error: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with load balancer status."""
    return {
        "message": "LM Proxy Load Balancer",
        "status": "running",
        "load_balancer": {
            "openai_keys": len(lb_config.openai_keys),
            "anthropic_keys": len(lb_config.anthropic_keys),
            "service_endpoints": lb_config.service_endpoints
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "load_balancer": {
            "openai_keys_configured": len(lb_config.openai_keys),
            "anthropic_keys_configured": len(lb_config.anthropic_keys),
            "endpoints": lb_config.service_endpoints
        }
    }

@app.get("/stats")
async def get_stats():
    """Get load balancer statistics."""
    return {
        "load_balancer_stats": {
            "openai": {
                "keys_count": len(lb_config.openai_keys),
                "endpoints": lb_config.service_endpoints.get("openai", [])
            },
            "anthropic": {
                "keys_count": len(lb_config.anthropic_keys),
                "endpoints": lb_config.service_endpoints.get("anthropic", [])
            }
        }
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Load Balancing LM Proxy Server")
        print("===============================")
        print()
        print("Environment Variables:")
        print("  OPENAI_API_KEYS     - Comma-separated list of OpenAI API keys")
        print("  ANTHROPIC_API_KEYS  - Comma-separated list of Anthropic API keys")
        print("  SERVICE_ENDPOINTS   - JSON object mapping services to endpoint URLs")
        print()
        print("Example:")
        print('  OPENAI_API_KEYS="key1,key2,key3"')
        print('  ANTHROPIC_API_KEYS="key1,key2"')
        print('  SERVICE_ENDPOINTS=\'{"openai": ["https://api.openai.com/v1", "https://api2.openai.com/v1"], "anthropic": ["https://api.anthropic.com/v1"]}\'')
        print()
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)
    
    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")

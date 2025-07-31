#!/usr/bin/env python3
"""
Test script for the Load Balancing LM Proxy Server.

This script tests various load balancing scenarios including:
- Multiple API key rotation
- Endpoint cycling
- Health checks
- Statistics endpoints
- Error handling

Usage:
  python test_loadbalancer.py --help
  python test_loadbalancer.py --test-keys
  python test_loadbalancer.py --test-endpoints
  python test_loadbalancer.py --test-health
"""

import os
import json
import time
import httpx
import argparse
import asyncio
from typing import Dict, List, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SERVER_URL = "http://localhost:8082"
TIMEOUT = 30.0

def test_health_endpoints():
    """Test the health check and stats endpoints."""
    print("🏥 Testing Health Endpoints...")
    
    try:
        # Test root endpoint
        response = httpx.get(f"{SERVER_URL}/", timeout=TIMEOUT)
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data.get('message')}")
            print(f"   Load Balancer: {data.get('load_balancer')}")
        
        # Test health endpoint
        response = httpx.get(f"{SERVER_URL}/health", timeout=TIMEOUT)
        print(f"✅ Health endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            lb_info = data.get('load_balancer', {})
            print(f"   OpenAI Keys: {lb_info.get('openai_keys_configured', 0)}")
            print(f"   Anthropic Keys: {lb_info.get('anthropic_keys_configured', 0)}")
        
        # Test stats endpoint
        response = httpx.get(f"{SERVER_URL}/stats", timeout=TIMEOUT)
        print(f"✅ Stats endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('load_balancer_stats', {})
            print(f"   OpenAI: {stats.get('openai', {})}")
            print(f"   Anthropic: {stats.get('anthropic', {})}")
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    return True

def test_load_balancing_with_requests():
    """Test load balancing by making multiple requests."""
    print("⚖️ Testing Load Balancing with Multiple Requests...")
    
    # Test message for API calls
    test_request = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Say 'Hello from load balancer test!'"}
        ]
    }
    
    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    success_count = 0
    total_requests = 5
    
    for i in range(total_requests):
        try:
            print(f"📤 Request {i+1}/{total_requests}...")
            
            response = httpx.post(
                f"{SERVER_URL}/v1/messages",
                headers=headers,
                json=test_request,
                timeout=TIMEOUT
            )
            
            if response.status_code == 200:
                print(f"✅ Request {i+1} successful")
                success_count += 1
                
                # Try to parse response
                try:
                    data = response.json()
                    if 'content' in data and data['content']:
                        content = data['content'][0].get('text', 'No text')[:100]
                        print(f"   Response: {content}...")
                except:
                    print("   Response: (parsing failed)")
            else:
                print(f"❌ Request {i+1} failed: {response.status_code}")
                if response.text:
                    print(f"   Error: {response.text[:200]}...")
                    
        except Exception as e:
            print(f"❌ Request {i+1} exception: {e}")
        
        # Small delay between requests
        time.sleep(1)
    
    print(f"📊 Success Rate: {success_count}/{total_requests}")
    return success_count > 0

def test_token_counting():
    """Test the token counting endpoint."""
    print("🔢 Testing Token Counting...")
    
    test_request = {
        "model": "claude-3-haiku-20240307",
        "messages": [
            {"role": "user", "content": "Count the tokens in this message."}
        ]
    }
    
    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        response = httpx.post(
            f"{SERVER_URL}/v1/messages/count_tokens",
            headers=headers,
            json=test_request,
            timeout=TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            token_count = data.get('input_tokens', 0)
            print(f"✅ Token counting successful: {token_count} tokens")
            return True
        else:
            print(f"❌ Token counting failed: {response.status_code}")
            if response.text:
                print(f"   Error: {response.text[:200]}...")
                
    except Exception as e:
        print(f"❌ Token counting exception: {e}")
    
    return False

def test_streaming():
    """Test streaming responses."""
    print("🌊 Testing Streaming...")
    
    test_request = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 50,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Stream a short hello message."}
        ]
    }
    
    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    try:
        with httpx.stream(
            "POST",
            f"{SERVER_URL}/v1/messages",
            headers=headers,
            json=test_request,
            timeout=TIMEOUT
        ) as response:
            
            if response.status_code == 200:
                print("✅ Streaming started...")
                chunk_count = 0
                
                for chunk in response.iter_text():
                    if chunk.strip():
                        chunk_count += 1
                        if chunk_count <= 3:  # Show first few chunks
                            print(f"   Chunk {chunk_count}: {chunk[:100]}...")
                
                print(f"✅ Streaming completed: {chunk_count} chunks received")
                return True
            else:
                print(f"❌ Streaming failed: {response.status_code}")
                
    except Exception as e:
        print(f"❌ Streaming exception: {e}")
    
    return False

def run_configuration_test():
    """Test configuration parsing and validation."""
    print("⚙️ Testing Configuration...")
    
    # Test environment variables
    openai_keys = os.environ.get("OPENAI_API_KEYS", os.environ.get("OPENAI_API_KEY", ""))
    anthropic_keys = os.environ.get("ANTHROPIC_API_KEYS", os.environ.get("ANTHROPIC_API_KEY", ""))
    service_endpoints = os.environ.get("SERVICE_ENDPOINTS", "{}")
    
    print(f"📋 Configuration Status:")
    print(f"   OpenAI Keys: {'✅ Configured' if openai_keys else '❌ Not configured'}")
    print(f"   Anthropic Keys: {'✅ Configured' if anthropic_keys else '❌ Not configured'}")
    
    # Try to parse service endpoints
    try:
        endpoints = json.loads(service_endpoints)
        print(f"   Service Endpoints: ✅ Valid JSON")
        for service, urls in endpoints.items():
            print(f"     {service}: {len(urls) if isinstance(urls, list) else 0} endpoints")
    except json.JSONDecodeError:
        print(f"   Service Endpoints: ❌ Invalid JSON")
    
    return True

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Test Load Balancing LM Proxy Server")
    parser.add_argument("--test-health", action="store_true", help="Test health endpoints only")
    parser.add_argument("--test-config", action="store_true", help="Test configuration only")
    parser.add_argument("--test-streaming", action="store_true", help="Test streaming only")
    parser.add_argument("--test-tokens", action="store_true", help="Test token counting only")
    parser.add_argument("--test-requests", action="store_true", help="Test request load balancing only")
    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    # If no specific test is selected, run all
    if not any([args.test_health, args.test_config, args.test_streaming, 
                args.test_tokens, args.test_requests]):
        args.all = True
    
    print("🚀 Load Balancer Test Suite")
    print("=" * 50)
    
    results = []
    
    # Configuration test
    if args.all or args.test_config:
        results.append(("Configuration", run_configuration_test()))
    
    # Health endpoints test
    if args.all or args.test_health:
        results.append(("Health Endpoints", test_health_endpoints()))
    
    # Request load balancing test
    if args.all or args.test_requests:
        results.append(("Request Load Balancing", test_load_balancing_with_requests()))
    
    # Token counting test
    if args.all or args.test_tokens:
        results.append(("Token Counting", test_token_counting()))
    
    # Streaming test
    if args.all or args.test_streaming:
        results.append(("Streaming", test_streaming()))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Load balancer is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check configuration and server status.")
        return 1

if __name__ == "__main__":
    exit(main()) 

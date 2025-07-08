#!/usr/bin/env python3
"""
Test script to verify LLM configuration and API connectivity
"""

import os
import sys
import asyncio
import httpx

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import src package to set environment variables
import src
from utils.config import get_settings

async def test_nebius_api():
    """Test Nebius AI API connectivity"""
    print("🔍 Testing Nebius AI API...")
    
    settings = get_settings()
    
    if not settings.NEBUIS_API_KEY:
        print("❌ Nebius API key not configured")
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {settings.NEBUIS_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": settings.DEFAULT_LLM_MODEL,
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "temperature": 0.7,
            "max_tokens": 50
        }
        
        print(f"  - API URL: {settings.NEBUIS_API_URL}")
        print(f"  - Model: {settings.DEFAULT_LLM_MODEL}")
        print(f"  - API Key: {settings.NEBUIS_API_KEY[:20]}...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                settings.NEBUIS_API_URL,
                headers=headers,
                json=payload
            )
            
            print(f"  - Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                print(f"  - Response Content: {content[:100]}...")
                print("✅ Nebius AI API test passed!")
                return True
            else:
                print(f"  - Error Response: {response.text}")
                print("❌ Nebius AI API test failed!")
                return False
                
    except Exception as e:
        print(f"  - Exception: {str(e)}")
        print("❌ Nebius AI API test failed!")
        return False

async def test_ollama_api():
    """Test Ollama API connectivity"""
    print("\n🔍 Testing Ollama API...")
    
    settings = get_settings()
    
    try:
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": "Hello, this is a test message.",
            "options": {
                "num_predict": 50,
                "temperature": 0.7,
            },
            "stream": False
        }
        
        print(f"  - API URL: {settings.OLLAMA_BASE_URL}")
        print(f"  - Model: {settings.OLLAMA_MODEL}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # First check if Ollama is running
            try:
                health_response = await client.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
                print(f"  - Ollama Status: {health_response.status_code}")
                
                if health_response.status_code == 200:
                    models = health_response.json().get("models", [])
                    model_names = [model["name"] for model in models]
                    print(f"  - Available Models: {model_names}")
                    
                    if settings.OLLAMA_MODEL in model_names:
                        print(f"  - Model '{settings.OLLAMA_MODEL}' is available")
                    else:
                        print(f"  - Model '{settings.OLLAMA_MODEL}' is NOT available")
                        print("  - You may need to run: ollama pull llama2")
                        return False
                
            except Exception as e:
                print(f"  - Cannot connect to Ollama: {str(e)}")
                print("  - Please ensure Ollama is running: ollama serve")
                return False
            
            # Test generation
            response = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/generate",
                json=payload
            )
            
            print(f"  - Generation Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print(f"  - Response Content: {response_text[:100]}...")
                print("✅ Ollama API test passed!")
                return True
            else:
                print(f"  - Error Response: {response.text}")
                print("❌ Ollama API test failed!")
                return False
                
    except Exception as e:
        print(f"  - Exception: {str(e)}")
        print("❌ Ollama API test failed!")
        return False

async def main():
    """Run all tests"""
    print("🔧 Testing LLM API Configuration...\n")
    
    settings = get_settings()
    print("Configuration:")
    print(f"  LLM_PROVIDER: {settings.LLM_PROVIDER}")
    print(f"  DEFAULT_LLM_MODEL: {settings.DEFAULT_LLM_MODEL}")
    print(f"  OLLAMA_MODEL: {settings.OLLAMA_MODEL}")
    print()
    
    nebius_ok = await test_nebius_api()
    ollama_ok = await test_ollama_api()
    
    print(f"\n📊 Test Results:")
    print(f"  Nebius AI: {'✅ PASS' if nebius_ok else '❌ FAIL'}")
    print(f"  Ollama: {'✅ PASS' if ollama_ok else '❌ FAIL'}")
    
    if settings.LLM_PROVIDER == "nebius" and not nebius_ok:
        print("\n⚠️  Primary LLM provider (Nebius) is not working!")
        if ollama_ok:
            print("   Consider switching to Ollama or fixing Nebius configuration.")
        else:
            print("   Both LLM providers are failing!")
    
    return nebius_ok or ollama_ok

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
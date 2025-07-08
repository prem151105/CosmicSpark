#!/usr/bin/env python3
"""
Test script to verify model configuration is correct
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import src package to set environment variables
import src
from utils.config import get_settings
from src.models.llm_handler import ContextAwareGenerator, ConversationContext, GenerationConfig

async def test_model_configuration():
    """Test that models are configured correctly"""
    print("🔧 Testing Model Configuration...\n")
    
    settings = get_settings()
    
    print("Settings from config:")
    print(f"  LLM_PROVIDER: {settings.LLM_PROVIDER}")
    print(f"  DEFAULT_LLM_MODEL: {settings.DEFAULT_LLM_MODEL}")
    print(f"  OLLAMA_MODEL: {getattr(settings, 'OLLAMA_MODEL', 'NOT SET')}")
    print(f"  NEBIUS_API_URL: {settings.NEBUIS_API_URL}")
    print()
    
    # Test LLM handler initialization
    generator = ContextAwareGenerator()
    print(f"Generator Ollama model: {generator.ollama_model_name}")
    
    # Test a simple generation
    context = ConversationContext(
        history=[],
        user_profile={},
        session_id="test",
        domain_context="general_inquiry",
        timestamp=None
    )
    
    config = GenerationConfig(max_tokens=100, temperature=0.7)
    
    print("\n🧪 Testing response generation...")
    
    try:
        response = await generator.generate_response(
            query="What is INSAT?",
            retrieval_results=[],
            context=context,
            config=config
        )
        
        print("✅ Response generated successfully!")
        print(f"Response: {response.get('response', 'No response')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Response generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_model_configuration())
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test script to verify TensorFlow Lite delegate fixes
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import src package to set environment variables
import src

def test_sentence_transformer():
    """Test SentenceTransformer initialization without TFLite errors"""
    try:
        from sentence_transformers import SentenceTransformer
        print("Testing SentenceTransformer initialization...")
        
        # Test with CPU device specification
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Test encoding
        test_text = "INSAT is a series of multipurpose geostationary satellites"
        embedding = model.encode([test_text])
        
        print(f"✅ SentenceTransformer test passed!")
        print(f"   - Model loaded successfully")
        print(f"   - Embedding shape: {embedding.shape}")
        print(f"   - Device: {model.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ SentenceTransformer test failed: {str(e)}")
        return False

def test_rag_components():
    """Test RAG components initialization"""
    try:
        from src.rag.retriever import ContextualQueryProcessor
        print("\nTesting RAG components...")
        
        # Test query processor
        processor = ContextualQueryProcessor()
        print("✅ ContextualQueryProcessor initialized successfully")
        
        # Test entity extraction
        entities = processor.extract_entities("What is INSAT satellite data?")
        print(f"✅ Entity extraction works: {len(entities)} entities found")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG components test failed: {str(e)}")
        return False

def test_llm_handler():
    """Test LLM handler initialization"""
    try:
        from src.models.llm_handler import ContextAwareGenerator
        print("\nTesting LLM handler...")
        
        generator = ContextAwareGenerator()
        print("✅ ContextAwareGenerator initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM handler test failed: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🔧 Testing TensorFlow Lite delegate fixes...\n")
    
    # Check environment variables
    print("Environment variables:")
    print(f"  TOKENIZERS_PARALLELISM: {os.environ.get('TOKENIZERS_PARALLELISM', 'not set')}")
    print(f"  TF_CPP_MIN_LOG_LEVEL: {os.environ.get('TF_CPP_MIN_LOG_LEVEL', 'not set')}")
    print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print()
    
    tests = [
        test_sentence_transformer,
        test_rag_components,
        test_llm_handler
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The TensorFlow Lite delegate fixes are working.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
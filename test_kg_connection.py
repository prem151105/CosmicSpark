#!/usr/bin/env python3
"""
Test Knowledge Graph connection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import get_settings
from src.knowledge_graph.kg_builder import DynamicKnowledgeGraph

def test_kg_connection():
    settings = get_settings()
    
    try:
        print("Testing Knowledge Graph connection...")
        kg = DynamicKnowledgeGraph(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )
        
        print("Knowledge Graph initialized successfully!")
        
        # Test with a simple document
        test_docs = ["INSAT-3DR satellite observes weather patterns over India"]
        result = kg.incremental_update(test_docs)
        
        print(f"Test update completed: {result}")
        
        kg.close()
        return True
        
    except Exception as e:
        print(f"Knowledge Graph test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_kg_connection()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test Neo4j connection
"""

from neo4j import GraphDatabase
import sys

def test_connection():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        with driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!' as message")
            record = result.single()
            print(f"Connection successful: {record['message']}")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
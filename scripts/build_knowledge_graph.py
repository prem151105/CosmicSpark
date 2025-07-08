#!/usr/bin/env python3
"""
Knowledge Graph Builder Script
Builds and updates the MOSDAC knowledge graph from various data sources
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.knowledge_graph.kg_builder import DynamicKnowledgeGraph
from src.knowledge_graph.kg_builder import LLMEntityExtractor
from src.knowledge_graph.kg_builder import RelationshipMapper
from utils.config import get_settings, ensure_directories
from utils.logger import setup_logger
from loguru import logger

def load_documents_from_file(file_path: str) -> List[str]:
    """Load documents from various file formats"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return []
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Handle crawled data format
            if 'documents' in data:
                return [doc['content'] for doc in data['documents']]
            # Handle simple list format
            elif isinstance(data, list):
                return [str(item) for item in data]
            else:
                return [str(data)]
    
    elif file_path.suffix == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Split by double newlines to get documents
            return [doc.strip() for doc in content.split('\n\n') if doc.strip()]
    
    else:
        logger.error(f"Unsupported file format: {file_path.suffix}")
        return []

def main():
    """Main knowledge graph building function"""
    parser = argparse.ArgumentParser(description="MOSDAC Knowledge Graph Builder")
    parser.add_argument("--input", required=True, help="Input file with documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild entire knowledge graph")
    parser.add_argument("--incremental", action="store_true", help="Incremental update")
    parser.add_argument("--schema-file", help="Custom schema file")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--dry-run", action="store_true", help="Dry run without updating database")
    
    args = parser.parse_args()
    
    # Setup
    setup_logger()
    ensure_directories()
    settings = get_settings()
    
    logger.info("Starting Knowledge Graph Builder...")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Mode: {'Rebuild' if args.rebuild else 'Incremental' if args.incremental else 'Default'}")
    
    # Load documents
    logger.info("Loading documents...")
    documents = load_documents_from_file(args.input)
    
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Initialize components
    try:
        # Knowledge Graph
        if not args.dry_run:
            kg = DynamicKnowledgeGraph(
                neo4j_uri=settings.NEO4J_URI,
                neo4j_user=settings.NEO4J_USER,
                neo4j_password=settings.NEO4J_PASSWORD
            )
        else:
            kg = None
            logger.info("Dry run mode - no database connection")
        
        # Entity Extractor
        entity_extractor = LLMEntityExtractor()
        
        # Relationship Mapper
        relationship_mapper = RelationshipMapper()
        
        # Load custom schema if provided
        if args.schema_file:
            logger.info(f"Loading custom schema from: {args.schema_file}")
            with open(args.schema_file, 'r') as f:
                custom_schema = json.load(f)
                entity_extractor.update_schema(custom_schema)
        
        # Process documents in batches
        total_entities = 0
        total_relationships = 0
        
        for i in range(0, len(documents), args.batch_size):
            batch = documents[i:i + args.batch_size]
            batch_num = i // args.batch_size + 1
            total_batches = (len(documents) + args.batch_size - 1) // args.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} documents)")
            
            # Extract entities from batch
            batch_entities = []
            batch_relationships = []
            
            for doc_idx, document in enumerate(batch):
                try:
                    # Extract entities
                    entities = entity_extractor.extract_entities_with_llm(document)
                    batch_entities.extend(entities)
                    
                    # Extract relationships
                    relationships = relationship_mapper.extract_relationships(document, entities)
                    batch_relationships.extend(relationships)
                    
                    if (doc_idx + 1) % 10 == 0:
                        logger.debug(f"Processed {doc_idx + 1}/{len(batch)} documents in batch")
                
                except Exception as e:
                    logger.error(f"Error processing document {i + doc_idx}: {e}")
                    continue
            
            logger.info(f"Batch {batch_num}: {len(batch_entities)} entities, {len(batch_relationships)} relationships")
            
            # Update knowledge graph
            if kg and not args.dry_run:
                if args.rebuild and batch_num == 1:
                    logger.info("Rebuilding knowledge graph...")
                    # Clear existing data by creating a new graph
                    kg = DynamicKnowledgeGraph(
                        neo4j_uri=settings.NEO4J_URI,
                        neo4j_user=settings.NEO4J_USER,
                        neo4j_password=settings.NEO4J_PASSWORD
                    )
                
                # Process the batch of documents
                if args.incremental:
                    kg.incremental_update(batch)
                else:
                    kg.build_knowledge_graph(batch)
            
            total_entities += len(batch_entities)
            total_relationships += len(batch_relationships)
        
        # Final statistics
        logger.info("Knowledge Graph Building Completed!")
        logger.info(f"Total entities processed: {total_entities}")
        logger.info(f"Total relationships processed: {total_relationships}")
        
        if kg and not args.dry_run:
            # Get final graph statistics
            final_stats = kg.get_graph_statistics()
            logger.info("Final Knowledge Graph Statistics:")
            logger.info(f"  Nodes: {final_stats.get('total_nodes', 0)}")
            logger.info(f"  Relationships: {final_stats.get('total_relationships', 0)}")
            logger.info(f"  Node Types: {final_stats.get('node_types', 0)}")
            logger.info(f"  Relationship Types: {final_stats.get('relationship_types', 0)}")
            
            # Save graph statistics
            stats_file = Path(settings.KG_DATA_DIR) / "graph_statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            logger.info(f"Statistics saved to: {stats_file}")
            
            kg.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during knowledge graph building: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
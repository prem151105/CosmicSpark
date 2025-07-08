#!/usr/bin/env python3
"""
MOSDAC Web Crawler Script
Crawls MOSDAC website and processes content for the knowledge base
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.crawler.web_crawler import MOSDACCrawler, CrawlConfig
from src.rag.vector_store import AdvancedVectorStore, Document
from src.knowledge_graph.kg_builder import DynamicKnowledgeGraph
from utils.config import get_settings, ensure_directories
from utils.logger import setup_logger
from loguru import logger

async def main():
    """Main crawler function"""
    parser = argparse.ArgumentParser(description="MOSDAC Web Crawler")
    parser.add_argument("--url", default="https://www.mosdac.gov.in", help="Starting URL to crawl")
    parser.add_argument("--max-pages", type=int, default=100, help="Maximum pages to crawl")
    parser.add_argument("--max-depth", type=int, default=3, help="Maximum crawl depth")
    parser.add_argument("--output", default="./data/crawled_data.json", help="Output file path")
    parser.add_argument("--update-kg", action="store_true", help="Update knowledge graph")
    parser.add_argument("--update-vs", action="store_true", help="Update vector store")
    
    args = parser.parse_args()
    
    # Setup
    setup_logger()
    ensure_directories()
    settings = get_settings()
    
    logger.info("Starting MOSDAC web crawler...")
    logger.info(f"Target URL: {args.url}")
    logger.info(f"Max pages: {args.max_pages}")
    logger.info(f"Max depth: {args.max_depth}")
    
    # Initialize crawler
    config = CrawlConfig(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        allowed_domains=['mosdac.gov.in', 'www.mosdac.gov.in'],
        excluded_patterns=[
            r'/login', r'/admin', r'\.css$', r'\.js$',
            r'/images/', r'/css/', r'/js/'
        ]
    )
    
    crawler = MOSDACCrawler()
    crawler.config = config
    
    try:
        # Start crawling
        logger.info("Starting crawl process...")
        crawled_documents = await crawler.crawl_website([args.url])
        
        # Save results
        crawler.save_crawl_results(args.output)
        
        # Display statistics
        stats = crawler.get_crawl_statistics()
        logger.info("Crawling completed!")
        logger.info(f"Total documents: {stats['total_documents']}")
        logger.info(f"Total size: {stats['total_size_mb']} MB")
        logger.info(f"Unique domains: {stats['unique_domains']}")
        
        # Update vector store if requested
        if args.update_vs:
            logger.info("Updating vector store...")
            vector_store = AdvancedVectorStore(
                persist_directory=settings.CHROMADB_PATH,
                collection_name="mosdac_documents"
            )
            
            # Convert crawled documents to vector store format
            documents = []
            for crawled_doc in crawled_documents:
                # Generate a unique ID using both URL and content hash
                import hashlib
                unique_id = hashlib.md5(
                    (crawled_doc.url + crawled_doc.content_hash).encode()
                ).hexdigest()
                
                doc = Document(
                    id=f"crawl_{unique_id}",
                    content=crawled_doc.content,
                    metadata={
                        **crawled_doc.metadata,
                        "url": crawled_doc.url,
                        "title": crawled_doc.title,
                        "crawl_timestamp": crawled_doc.crawl_timestamp.isoformat(),
                        "content_hash": crawled_doc.content_hash  # Store original hash in metadata
                    },
                    modality="text",
                    source="web_crawl",
                    timestamp=crawled_doc.crawl_timestamp.isoformat()
                )
                documents.append(doc)
            
            vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        
        # Update knowledge graph if requested
        if args.update_kg:
            logger.info("Updating knowledge graph...")
            kg = DynamicKnowledgeGraph(
                neo4j_uri=settings.NEO4J_URI,
                neo4j_user=settings.NEO4J_USER,
                neo4j_password=settings.NEO4J_PASSWORD
            )
            
            # Extract text content for KG update
            document_texts = [doc.content for doc in crawled_documents]
            kg.incremental_update(document_texts)
            
            kg_stats = kg.get_graph_statistics()
            logger.info(f"Knowledge graph updated: {kg_stats['total_nodes']} nodes, {kg_stats['total_relationships']} relationships")
            
            kg.close()
        
        logger.info(f"Results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during crawling: {e}")
        return 1
    
    finally:
        crawler.close()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
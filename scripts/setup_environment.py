#!/usr/bin/env python3
"""
Environment Setup Script for MOSDAC AI Help Bot
Sets up the complete development and production environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import get_settings, ensure_directories
from utils.logger import setup_logger
from loguru import logger

def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run shell command with logging"""
    logger.info(f"Running: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.debug(f"Output: {result.stdout}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        raise

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = run_command("docker --version", check=False)
        if result.returncode != 0:
            logger.error("Docker is not installed or not in PATH")
            return False
        
        result = run_command("docker info", check=False)
        if result.returncode != 0:
            logger.error("Docker daemon is not running")
            return False
        
        logger.info("Docker is available and running")
        return True
    except Exception as e:
        logger.error(f"Error checking Docker: {e}")
        return False

def setup_ollama():
    """Setup Ollama and download required models"""
    logger.info("Setting up Ollama...")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info("Ollama is already running")
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            # Download required models if not present
            required_models = ["llama2:7b-chat", "mistral:7b-instruct"]
            
            for model in required_models:
                if not any(model in name for name in model_names):
                    logger.info(f"Downloading model: {model}")
                    run_command(f"ollama pull {model}")
                else:
                    logger.info(f"Model {model} already available")
        else:
            logger.warning("Ollama API not responding")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        logger.info("Please ensure Ollama is installed and running")

def setup_neo4j():
    """Setup Neo4j database"""
    logger.info("Setting up Neo4j...")
    
    # Check if Neo4j is running via Docker
    try:
        result = run_command("docker ps --filter name=mosdac-neo4j --format '{{.Names}}'", check=False)
        if "mosdac-neo4j" in result.stdout:
            logger.info("Neo4j container is already running")
        else:
            logger.info("Starting Neo4j container...")
            run_command("docker-compose up -d neo4j")
            
            # Wait for Neo4j to be ready
            logger.info("Waiting for Neo4j to be ready...")
            time.sleep(30)
            
    except Exception as e:
        logger.error(f"Error setting up Neo4j: {e}")

def setup_python_environment():
    """Setup Python environment and install dependencies"""
    logger.info("Setting up Python environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major != 3 or python_version.minor < 8:
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Install requirements
    logger.info("Installing Python dependencies...")
    run_command("pip install -r requirements.txt")
    
    # Download spaCy model
    logger.info("Downloading spaCy model...")
    run_command("python -m spacy download en_core_web_sm")
    
    return True

def setup_directories():
    """Setup required directories"""
    logger.info("Setting up directories...")
    ensure_directories()
    
    # Create additional directories
    additional_dirs = [
        "data/raw",
        "data/processed", 
        "data/knowledge_graph",
        "data/chromadb",
        "logs",
        "monitoring",
        "ssl"
    ]
    
    for dir_path in additional_dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")

def create_env_file():
    """Create .env file with default settings"""
    env_file = Path(".env")
    
    if env_file.exists():
        logger.info(".env file already exists")
        return
    
    logger.info("Creating .env file...")
    
    env_content = """# MOSDAC AI Help Bot Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Database Configuration
CHROMADB_PATH=./data/chromadb
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_LLM_MODEL=llama2:7b-chat
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Crawler Configuration
CRAWLER_USER_AGENT=MOSDAC-Crawler/1.0
CRAWLER_DELAY=1.0
CRAWLER_MAX_PAGES=1000

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Data Paths
DATA_DIR=./data
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
KG_DATA_DIR=./data/knowledge_graph
"""
    
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    logger.info("Created .env file with default settings")

def run_initial_tests():
    """Run initial tests to verify setup"""
    logger.info("Running initial tests...")
    
    try:
        # Test imports
        logger.info("Testing imports...")
        from src.rag.vector_store import AdvancedVectorStore
        from src.knowledge_graph.kg_builder import DynamicKnowledgeGraph
        from src.models.llm_handler import ContextAwareGenerator
        logger.info("All imports successful")
        
        # Test vector store
        logger.info("Testing vector store...")
        vs = AdvancedVectorStore(persist_directory="./data/test_chromadb")
        logger.info("Vector store test successful")
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="MOSDAC Environment Setup")
    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-tests", action="store_true", help="Skip initial tests")
    parser.add_argument("--production", action="store_true", help="Production setup")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    logger.info("🚀 Starting MOSDAC AI Help Bot Environment Setup")
    logger.info("=" * 60)
    
    try:
        # Step 1: Setup directories
        setup_directories()
        
        # Step 2: Create environment file
        create_env_file()
        
        # Step 3: Setup Python environment
        if not setup_python_environment():
            logger.error("Python environment setup failed")
            return 1
        
        # Step 4: Docker setup
        if not args.skip_docker:
            if check_docker():
                setup_neo4j()
            else:
                logger.warning("Skipping Docker setup - Docker not available")
        
        # Step 5: Setup Ollama
        if not args.skip_models:
            setup_ollama()
        
        # Step 6: Run tests
        if not args.skip_tests:
            if not run_initial_tests():
                logger.warning("Some tests failed - please check the setup")
        
        # Final message
        logger.info("=" * 60)
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the services: docker-compose up -d")
        logger.info("2. Run the crawler: python scripts/run_crawler.py --url https://www.mosdac.gov.in")
        logger.info("3. Build knowledge graph: python scripts/build_knowledge_graph.py --input data/crawled_data.json")
        logger.info("4. Start the API: python -m src.api.main")
        logger.info("5. Start the frontend: streamlit run frontend/streamlit_app.py")
        logger.info("")
        logger.info("Access points:")
        logger.info("- Frontend: http://localhost:8501")
        logger.info("- API: http://localhost:8000")
        logger.info("- Neo4j Browser: http://localhost:7474")
        logger.info("- Grafana: http://localhost:3000")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
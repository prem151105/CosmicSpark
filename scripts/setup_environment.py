#!/usr/bin/env python3
"""
Environment Setup Script for CosmicSpark
Sets up the complete development and production environment
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import json
import time
import importlib.util

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

def check_requirements():
    """Check if required tools are installed"""
    required_tools = {
        "python": "Python 3.8+",
        "pip": "pip package manager",
        "git": "Git version control"
    }
    
    all_available = True
    for tool, description in required_tools.items():
        result = run_command(f"{tool} --version", check=False)
        if result.returncode != 0:
            logger.error(f"{description} is not installed or not in PATH")
            all_available = False
        else:
            logger.info(f"✓ {description} is available")
    
    return all_available

def setup_ollama():
    """Setup Ollama and download required models"""
    logger.info("Setting up Ollama...")
    
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
            logger.warning("Ollama API not responding. Make sure Ollama is installed and running.")
            logger.info("Download from: https://ollama.ai/")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}")
        logger.info("Ollama is optional but recommended for local LLM inference")

def setup_neo4j():
    """Setup Neo4j database (optional)"""
    logger.info("Setting up Neo4j...")
    
    try:
        # Check if neo4j package is installed
        neo4j_spec = importlib.util.find_spec("neo4j")
        if neo4j_spec is None:
            logger.warning("Neo4j Python driver not installed. Some features may be limited.")
            logger.info("To enable Neo4j features, install it with: pip install neo4j")
            return False
            
        # Try to connect to Neo4j if configured
        from neo4j import GraphDatabase
        from neo4j.exceptions import ServiceUnavailable
        
        try:
            uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            auth = (
                os.getenv("NEO4J_USER", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "password")
            )
            
            driver = GraphDatabase.driver(uri, auth=auth)
            with driver.session() as session:
                session.run("RETURN 1")
                
            logger.info("✓ Successfully connected to Neo4j database")
            return True
            
        except ServiceUnavailable:
            logger.warning("Could not connect to Neo4j. Some features may be limited.")
            logger.info("To use Neo4j, make sure it's running and update the .env file with the correct credentials")
            return False
            
    except Exception as e:
        logger.warning(f"Neo4j setup warning: {e}")
        logger.info("The application can run without Neo4j, but some features will be limited.")
        return False

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
    run_command("pip install -r requirements_clean.txt")
    
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
        "logs"
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
    
    env_content = """# CosmicSpark Configuration

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

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log
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
    parser = argparse.ArgumentParser(description="CosmicSpark Environment Setup")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--skip-tests", action="store_true", help="Skip initial tests")
    parser.add_argument("--production", action="store_true", help="Production setup")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    
    logger.info("🚀 Starting CosmicSpark Environment Setup")
    logger.info("=" * 60)
    
    try:
        # Step 1: Check requirements
        if not check_requirements():
            logger.error("Some required tools are missing. Please install them and try again.")
            return 1
        
        # Step 2: Setup directories
        setup_directories()
        
        # Step 3: Create environment file if it doesn't exist
        create_env_file()
        
        # Step 4: Setup Python environment
        if not setup_python_environment():
            logger.error("Python environment setup failed")
            return 1
        
        # Step 5: Setup Neo4j (optional)
        setup_neo4j()
        
        # Step 6: Setup Ollama (optional)
        if not args.skip_models:
            setup_ollama()
        
        # Step 7: Run tests
        if not args.skip_tests:
            if not run_initial_tests():
                logger.warning("Some tests failed - please check the setup")
        
        # Final message
        logger.info("=" * 60)
        logger.info("🎉 Environment setup completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Start the backend: python -m src.api.main")
        logger.info("2. In a new terminal, start the frontend: streamlit run frontend/streamlit_app.py")
        logger.info("")
        logger.info("Access points:")
        logger.info("- Frontend: http://localhost:8501")
        logger.info("- API: http://localhost:8000")
        logger.info("- API Documentation: http://localhost:8000/docs")
        
        return 0
        
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

"""
Configuration management for MOSDAC AI Help Bot
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = True
    
    # Database Configuration
    CHROMADB_PATH: str = "./data/chromadb"
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    
    # LLM Configuration
    LLM_PROVIDER: str = "nebius"  # 'ollama' or 'nebius'
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama2:latest"  # Ollama model name
    NEBUIS_API_KEY: str = ""  # Your Nebius AI API key
    NEBUIS_API_URL: str = "https://api.studio.nebius.ai/v1/chat/completions"
    DEFAULT_LLM_MODEL: str = "meta-llama/Llama-3.3-70B-Instruct"  # Nebius model name
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Crawler Configuration
    CRAWLER_USER_AGENT: str = "MOSDAC-Crawler/1.0"
    CRAWLER_DELAY: float = 1.0
    CRAWLER_MAX_PAGES: int = 1000
    MOSDAC_BASE_URL: str = "https://www.mosdac.gov.in"
    CRAWL_DELAY: float = 1.0
    MAX_PAGES: int = 1000
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # Data Paths
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    KG_DATA_DIR: str = "./data/knowledge_graph"
    
    # Frontend Configuration
    STREAMLIT_PORT: int = 8501
    STREAMLIT_HOST: str = "0.0.0.0"
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"  # Allow extra fields in .env file
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Environment-specific configurations
def get_database_url() -> str:
    """Get database URL based on environment"""
    settings = get_settings()
    return f"neo4j://{settings.NEO4J_USER}:{settings.NEO4J_PASSWORD}@{settings.NEO4J_URI.replace('bolt://', '')}"

def ensure_directories():
    """Ensure all required directories exist"""
    settings = get_settings()
    
    directories = [
        settings.DATA_DIR,
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.KG_DATA_DIR,
        settings.CHROMADB_PATH,
        "./logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
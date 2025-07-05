# MOSDAC AI Help Bot

This project is an AI-based Help Bot for information retrieval from the MOSDAC (www.mosdac.gov.in) portal. It utilizes a Retrieval-Augmented Generation (RAG) system with knowledge graph integration to answer user queries.

## Project Overview

The main goal is to extract content from the MOSDAC portal, build a dynamic knowledge graph, and implement a RAG pipeline using open-source LLMs. A Streamlit-based conversational interface will serve as the frontend. The system is designed for modularity to allow potential deployment across other government portals.

## Key Features

- Web crawling of MOSDAC portal content (web pages, PDFs, documents).
- Dynamic knowledge graph construction (entities and relationships).
- RAG pipeline with vector similarity search and graph-based context retrieval.
- Integration with open-source LLMs (e.g., Llama-2, Mistral) via Ollama.
- Conversational user interface using Streamlit.
- Dockerized components for deployment.

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **RAG Pipeline**: Langchain, Sentence-Transformers
- **Knowledge Graph**: Neo4j
- **Vector Database**: ChromaDB
- **LLM**: Ollama (Llama-2, Mistral)
- **Web Crawling**: Scrapy, BeautifulSoup4, Selenium
- **NLP**: spaCy, Transformers
- **Deployment**: Docker

This project is part of a 6-day sprint, aiming for a functional prototype demonstrating all core capabilities.

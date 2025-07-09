# CosmicSpark: KnowledgeForge AI

CosmicSpark is an advanced knowledge management and retrieval system that combines state-of-the-art NLP, graph-based knowledge representation, and interactive visualization to create a powerful platform for information extraction, processing, and exploration.

## 🚀 Key Innovations

1. **Hybrid Knowledge Representation**
   - Combines vector embeddings with graph-based relationships
   - Dynamic schema adaptation based on document content
   - Multi-hop reasoning across connected entities

2. **Advanced NLP Pipeline**
   - Custom entity recognition with domain adaptation
   - Contextual relationship mapping
   - Multi-modal processing (text, PDFs, web content)

3. **Intelligent Search & Retrieval**
   - Hybrid search combining BM25 with dense vector search
   - Context-aware result reranking
   - Query understanding and expansion

## 🛠️ Prerequisites

- Python 3.9+
- pip (Python package manager)
- (Optional) Neo4j Database for graph features
- (Recommended) CUDA-enabled GPU for faster inference

## 🚀 Quick Start

1. **Clone and Setup**
   ```bash
   git clone https://github.com/prem151105/CosmicSpark.git
   cd CosmicSpark
   
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   
   # Install dependencies
   pip install -r requirements_clean.txt
   ```

2. **Configure Environment**
   Copy `.env.example` to `.env` and update the settings:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` to configure:
   - API settings
   - Database connections
   - Model parameters

## 🚀 Running the Application

### Backend (FastAPI)
```bash
# Start the API server
python -m src.api.main
```

### Frontend (Streamlit)
```bash
# In a new terminal
streamlit run frontend/streamlit_app.py
```

Access the application at `http://localhost:8501`

## 🏗️ Project Structure

```
CosmicSpark/
├── data/                   # Data storage
│   ├── chromadb/           # Vector embeddings
│   ├── knowledge_graph/    # Graph database files
│   ├── processed/          # Processed documents
│   └── raw/                # Raw input data
│
├── frontend/               # Streamlit UI
│   ├── components/         # Reusable UI components
│   │   ├── chat_interface.py
│   │   └── knowledge_graph_viz.py
│   └── streamlit_app.py    # Main application
│
├── src/                    # Core application code
│   ├── api/                # FastAPI endpoints
│   │   ├── endpoints.py
│   │   └── main.py
│   │
│   ├── crawler/            # Web crawling and processing
│   │   ├── content_extractor.py
│   │   ├── pdf_processor.py
│   │   └── web_crawler.py
│   │
│   ├── knowledge_graph/    # Knowledge graph components
│   │   ├── entity_extractor.py
│   │   ├── kg_builder.py
│   │   └── relationship_mapper.py
│   │
│   ├── models/             # ML models
│   │   ├── embedding_model.py
│   │   └── llm_handler.py
│   │
│   └── rag/                # Retrieval-Augmented Generation
│       ├── generator.py
│       ├── retriever.py
│       └── vector_store.py
│
├── .env.example           # Example environment config
├── requirements_clean.txt # Dependencies
└── README.md             # This file
```

## 🛠️ Configuration

### Environment Variables

```env
# Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Vector Store
VECTOR_STORE_PATH=./data/chromadb
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Knowledge Graph (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
LLM_MODEL=llama3
LLM_TEMPERATURE=0.7
MAX_TOKENS=1024
```

## 🚀 Deployment

### Production Setup
1. Set up a production-grade server (Nginx/Apache)
2. Use Gunicorn/Uvicorn for ASGI server
3. Configure environment variables in production
4. Set up monitoring and logging



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [CosmicSpark](LICENSE) file for details.

## ✨ Key Features in Action

- **Real-time Knowledge Graph Updates**: See relationships form as you add documents
- **Context-Aware Chat**: Get precise answers with source attribution
- **Document Intelligence**: Extract and link entities across multiple documents
- **Scalable Architecture**: Built to handle thousands of documents

## 📚 Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## 📬 Contact

For questions or feedback, please open an issue or contact the maintainers.

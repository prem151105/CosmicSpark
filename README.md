# KnowledgeForge AI

KnowledgeForge AI is a powerful knowledge management and retrieval system that combines web crawling, natural language processing, and graph-based knowledge representation to build and query a comprehensive knowledge base.

## Features

- **Web Crawling**: Extract and process content from various web sources
- **Knowledge Graph**: Store and query relationships between entities
- **Semantic Search**: Find relevant information using natural language
- **Document Processing**: Handle multiple document formats (PDF, DOCX, etc.)
- **Interactive UI**: User-friendly interface for exploring knowledge

## Prerequisites

- Python 3.9+
- Neo4j Database (local or remote)
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd KnowledgeForge-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements_clean.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following content:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

## Running the Application

### 1. Start the Backend (FastAPI)

Open a terminal and run:

```bash
# Activate virtual environment (if not already activated)
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Start the FastAPI backend
python -m src.api.main
```

The backend will start at `http://localhost:8000` by default. Access the API documentation at `http://localhost:8000/docs`.

### 2. Start the Frontend (Streamlit)

Open a new terminal and run:

```bash
# Activate virtual environment in the new terminal
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install streamlit if not already installed
pip install streamlit

# Start the Streamlit frontend
streamlit run frontend/streamlit_app.py
```

The frontend will open automatically in your default web browser at `http://localhost:8501`.

### 3. Using the Application

1. The main interface includes:
   - **Chat Interface**: For asking questions and getting AI responses
   - **Knowledge Graph**: Visual exploration of entities and relationships
   - **Document Upload**: Add new documents to the knowledge base
   - **System Status**: Monitor the health of the application

2. To test the system:
   - Try asking questions in the chat interface
   - Upload documents to expand the knowledge base
   - Explore the knowledge graph visualization

## Development

### Project Structure

```
KnowledgeForge/
├── src/                    # Source code
│   ├── api/                # FastAPI application
│   ├── knowledge_graph/    # Knowledge graph components
│   ├── models/             # ML models and embeddings
│   └── rag/                # Retrieval-Augmented Generation
├── frontend/               # Streamlit frontend
│   ├── components/         # Reusable UI components
│   └── streamlit_app.py    # Main frontend application
├── data/                   # Data storage
├── tests/                  # Test files
├── .env.example            # Example environment variables
└── requirements.txt        # Python dependencies
```

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Neo4j Configuration (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
LLM_MODEL=llama3  # or any other supported model
LLM_TEMPERATURE=0.7

# Vector Store
VECTOR_STORE_PATH=./data/vector_store
```

## Troubleshooting

1. **Port Conflicts**: If ports 8000 (backend) or 8501 (frontend) are in use, update the respective configuration.

2. **Missing Dependencies**: Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. **Neo4j Connection**: The application can run without Neo4j, but some features may be limited. To use Neo4j:
   - Install Neo4j Desktop or Docker
   - Update the `.env` file with your Neo4j credentials
   - Ensure the database is running before starting the backend

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
cd src/api
uvicorn main:app --reload
```

### Frontend (Streamlit)
```bash
streamlit run frontend/streamlit_app.py
```

## Project Structure

```
KnowledgeForge-AI/
├── src/                    # Source code
│   ├── api/                # FastAPI backend
│   ├── crawler/            # Web crawling components
│   ├── models/             # Data models
│   └── utils/              # Utility functions
├── data/                   # Data storage
│   ├── raw/                # Raw crawled data
│   └── processed/          # Processed data
├── requirements_clean.txt   # Project dependencies
└── .env.example           # Example environment variables
```

## Configuration

Update the `.env` file with your Neo4j database credentials and other configuration options.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [CosmicSpark](LICENSE) file for details.

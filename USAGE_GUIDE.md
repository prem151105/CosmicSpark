# CosmicSpark Usage Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Basic Commands](#basic-commands)
3. [Configuration](#configuration)
4. [Troubleshooting](#troubleshooting)
5. [FAQs](#faqs)

## Getting Started

### Prerequisites
- Python 3.9+
- pip (Python package manager)
- (Optional) Neo4j for graph features

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prem151105/CosmicSpark.git
   cd CosmicSpark
   ```

2. **Set up the environment**
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # Linux/Mac
   
   # Install dependencies
   pip install -r requirements_clean.txt
   
   # Set up the environment
   python run.py setup
   ```

## Basic Commands

### Start Development Server
```bash
# Start both backend and frontend
python run.py dev
```

### Individual Components

#### Backend API
```bash
# Start backend only
python run.py backend

# Access API documentation
# http://localhost:8000/docs
```

#### Frontend
```bash
# Start frontend only
python run.py frontend

# Access at http://localhost:8501
```

### Data Management
```bash
# Clean data directories
python run.py clean

# View logs
python run.py logs
```

## Configuration

### Environment Variables
Create a `.env` file in the root directory with the following variables:

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
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Find and kill the process
   lsof -i :8000  # For port 8000
   kill -9 <PID>  # Replace <PID> with the process ID
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements_clean.txt
   ```

3. **Neo4j Connection Issues**
   - Ensure Neo4j is running
   - Verify credentials in `.env`
   - Check if the port (default: 7687) is accessible

## FAQs

### Q: How do I reset the database?
A: Run `python run.py clean` to reset the data directories.

### Q: How do I update the models?
A: Update the model names in the `.env` file and restart the application.

### Q: How can I contribute?
A: Fork the repository, create a feature branch, and submit a pull request.

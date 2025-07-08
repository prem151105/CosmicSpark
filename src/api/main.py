"""
Advanced MOSDAC AI Help Bot API
Main FastAPI application with GraphRAG and multimodal capabilities
"""

# Import src package first to set environment variables
import src

import asyncio
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import networkx as nx
from loguru import logger

# Import our advanced components
from ..rag.retriever import AdvancedRAGPipeline, RetrievalResult
from ..rag.vector_store import AdvancedVectorStore, Document, MultimodalQuery
from ..knowledge_graph.kg_builder import DynamicKnowledgeGraph
from ..models.llm_handler import ContextAwareGenerator, ConversationContext, GenerationConfig
from ..crawler.web_crawler import MOSDACCrawler
from utils.config import get_settings
from utils.logger import setup_logger

# Global variables for components
rag_pipeline: Optional[AdvancedRAGPipeline] = None
vector_store: Optional[AdvancedVectorStore] = None
knowledge_graph: Optional[DynamicKnowledgeGraph] = None
llm_generator: Optional[ContextAwareGenerator] = None
crawler: Optional[MOSDACCrawler] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    await initialize_components()
    yield
    # Shutdown
    await cleanup_components()

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="User query")
    session_id: str = Field(default="default", description="Session ID for conversation tracking")
    user_profile: Dict[str, Any] = Field(default_factory=dict, description="User profile information")
    stream: bool = Field(default=False, description="Enable streaming response")
    max_tokens: int = Field(default=1024, description="Maximum tokens in response")
    temperature: float = Field(default=0.7, description="Generation temperature")

class QueryResponse(BaseModel):
    response: str
    confidence_score: float
    intent: str
    entities: List[Dict[str, Any]]
    retrieval_results: List[Dict[str, Any]]
    reasoning_path: List[str]
    session_id: str
    timestamp: str

class DocumentUpload(BaseModel):
    content: str
    title: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    modality: str = Field(default="text", description="Document modality: text, image, table")

class CrawlRequest(BaseModel):
    urls: List[str] = Field(..., description="URLs to crawl")
    max_pages: int = Field(default=100, description="Maximum pages to crawl")
    max_depth: int = Field(default=2, description="Maximum crawl depth")

class SystemStatus(BaseModel):
    status: str
    components: Dict[str, bool]
    statistics: Dict[str, Any]
    timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="MOSDAC AI Help Bot",
    description="Advanced AI assistant for meteorological and oceanographic data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_components():
    """Initialize all system components"""
    global rag_pipeline, vector_store, knowledge_graph, llm_generator, crawler
    
    logger.info("Initializing MOSDAC AI Help Bot components...")
    
    try:
        # Get configuration
        settings = get_settings()
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = AdvancedVectorStore(
            persist_directory=settings.CHROMADB_PATH,
            collection_name="mosdac_documents"
        )
        
        # Initialize knowledge graph
        logger.info("Initializing knowledge graph...")
        knowledge_graph = DynamicKnowledgeGraph(
            neo4j_uri=settings.NEO4J_URI,
            neo4j_user=settings.NEO4J_USER,
            neo4j_password=settings.NEO4J_PASSWORD
        )
        
        # Initialize LLM generator
        logger.info("Initializing LLM generator...")
        llm_generator = ContextAwareGenerator(
            model_name=getattr(settings, 'OLLAMA_MODEL', 'llama2:latest'),
            ollama_base_url=settings.OLLAMA_BASE_URL
        )
        
        # Initialize crawler
        logger.info("Initializing web crawler...")
        crawler = MOSDACCrawler()
        
        # Initialize RAG pipeline with comprehensive MOSDAC data
        logger.info("Initializing RAG pipeline...")
        
        # Use comprehensive MOSDAC documents instead of minimal sample
        mosdac_documents = [
            "INSAT (Indian National Satellite System) is a series of multipurpose geostationary satellites launched by ISRO to satisfy telecommunications, broadcasting, meteorology, and search and rescue operations.",
            "MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) is a facility established by ISRO to provide satellite-based meteorological and oceanographic data and products to users.",
            "SCATSAT-1 is an Indian satellite mission for ocean and weather related studies. It carries a Ku-band scatterometer for wind vector data products over global oceans.",
            "OCEANSAT-2 carried Ocean Color Monitor (OCM) and Ku-band pencil beam scatterometer for ocean color and wind vector measurements respectively.",
            "Indian Meteorological Department (IMD) uses satellite data from INSAT and other satellites for weather forecasting and monitoring including cloud imagery analysis, temperature and humidity profiles, and cyclone monitoring.",
            "CARTOSAT series are Indian remote sensing satellites designed for cartographic applications. CARTOSAT-1 provides stereo imagery for Digital Elevation Model generation while CARTOSAT-2 series provides high resolution imagery.",
            "MEGHA-TROPIQUES is a joint Indo-French satellite mission for studying the water cycle and energy exchanges in the tropical atmosphere with payloads like MADRAS, SAPHIR, and ScaRaB.",
            "MOSDAC archives and disseminates data from various Indian and foreign satellites including Sea Surface Temperature, Ocean Color data, Wind Vector data, Precipitation data, and Cyclone tracking information."
        ]
        
        # Create comprehensive knowledge graph
        mosdac_graph = nx.Graph()
        
        # Organizations
        mosdac_graph.add_node("MOSDAC", type="organization")
        mosdac_graph.add_node("ISRO", type="organization")
        mosdac_graph.add_node("IMD", type="organization")
        
        # Satellites
        mosdac_graph.add_node("INSAT", type="satellite_system")
        mosdac_graph.add_node("SCATSAT-1", type="satellite")
        mosdac_graph.add_node("OCEANSAT-2", type="satellite")
        mosdac_graph.add_node("CARTOSAT-1", type="satellite")
        mosdac_graph.add_node("CARTOSAT-2", type="satellite")
        mosdac_graph.add_node("MEGHA-TROPIQUES", type="satellite")
        
        # Data types
        mosdac_graph.add_node("Weather Data", type="data_type")
        mosdac_graph.add_node("Ocean Data", type="data_type")
        mosdac_graph.add_node("Wind Data", type="data_type")
        mosdac_graph.add_node("Satellite Imagery", type="data_type")
        
        # Relationships
        mosdac_graph.add_edge("MOSDAC", "ISRO", relation="operated_by")
        mosdac_graph.add_edge("INSAT", "ISRO", relation="developed_by")
        mosdac_graph.add_edge("SCATSAT-1", "ISRO", relation="developed_by")
        mosdac_graph.add_edge("OCEANSAT-2", "ISRO", relation="developed_by")
        mosdac_graph.add_edge("CARTOSAT-1", "ISRO", relation="developed_by")
        mosdac_graph.add_edge("CARTOSAT-2", "ISRO", relation="developed_by")
        mosdac_graph.add_edge("MEGHA-TROPIQUES", "ISRO", relation="joint_mission")
        mosdac_graph.add_edge("INSAT", "Weather Data", relation="provides")
        mosdac_graph.add_edge("OCEANSAT-2", "Ocean Data", relation="provides")
        mosdac_graph.add_edge("SCATSAT-1", "Wind Data", relation="provides")
        mosdac_graph.add_edge("CARTOSAT-1", "Satellite Imagery", relation="provides")
        mosdac_graph.add_edge("CARTOSAT-2", "Satellite Imagery", relation="provides")
        mosdac_graph.add_edge("MOSDAC", "Weather Data", relation="archives")
        mosdac_graph.add_edge("MOSDAC", "Ocean Data", relation="archives")
        mosdac_graph.add_edge("MOSDAC", "Wind Data", relation="archives")
        mosdac_graph.add_edge("IMD", "Weather Data", relation="uses")
        
        logger.info(f"Created knowledge graph with {mosdac_graph.number_of_nodes()} nodes and {mosdac_graph.number_of_edges()} edges")
        
        rag_pipeline = AdvancedRAGPipeline(mosdac_graph, mosdac_documents)
        
        logger.info("All components initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

async def cleanup_components():
    """Cleanup components on shutdown"""
    global knowledge_graph, crawler
    
    logger.info("Cleaning up components...")
    
    if knowledge_graph:
        knowledge_graph.close()
    
    if crawler:
        crawler.close()
    
    logger.info("Cleanup completed")

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "MOSDAC AI Help Bot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Health check endpoint"""
    
    components_status = {
        "rag_pipeline": rag_pipeline is not None,
        "vector_store": vector_store is not None,
        "knowledge_graph": knowledge_graph is not None,
        "llm_generator": llm_generator is not None,
        "crawler": crawler is not None
    }
    
    all_healthy = all(components_status.values())
    
    # Get statistics
    statistics = {}
    if vector_store:
        statistics["vector_store"] = vector_store.get_collection_stats()
    if knowledge_graph:
        statistics["knowledge_graph"] = knowledge_graph.get_graph_statistics()
    
    return SystemStatus(
        status="healthy" if all_healthy else "degraded",
        components=components_status,
        statistics=statistics,
        timestamp=datetime.now().isoformat()
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process user query with advanced RAG"""
    
    if not rag_pipeline or not llm_generator:
        raise HTTPException(status_code=503, detail="System components not initialized")
    
    try:
        # Create conversation context
        context = ConversationContext(
            history=llm_generator.get_conversation_history(request.session_id),
            user_profile=request.user_profile,
            session_id=request.session_id,
            domain_context="general_inquiry",  # Will be determined by intent classification
            timestamp=datetime.now()
        )
        
        # Process query through RAG pipeline
        rag_result = await rag_pipeline.process_query(
            request.query,
            user_context={"history": context.history}
        )
        
        # Update domain context based on intent
        context.domain_context = rag_result.get("intent", "general_inquiry")
        
        # Generate response
        generation_config = GenerationConfig(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=request.stream
        )
        
        llm_result = await llm_generator.generate_response(
            request.query,
            rag_result.get("retrieval_results", []),
            context,
            generation_config
        )
        
        # Update conversation memory
        llm_generator.update_conversation_memory(
            request.session_id,
            request.query,
            llm_result.get("response", "")
        )
        
        # Format retrieval results for response
        formatted_results = []
        for result in rag_result.get("retrieval_results", []):
            if hasattr(result, 'content'):
                formatted_results.append({
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "score": result.score,
                    "source": result.source
                })
        
        return QueryResponse(
            response=llm_result.get("response", ""),
            confidence_score=rag_result.get("confidence_score", 0.0),
            intent=rag_result.get("intent", "general_inquiry"),
            entities=rag_result.get("entities", []),
            retrieval_results=formatted_results,
            reasoning_path=rag_result.get("reasoning_path", []),
            session_id=request.session_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/query/stream")
async def stream_query(request: QueryRequest):
    """Stream query response"""
    
    if not rag_pipeline or not llm_generator:
        raise HTTPException(status_code=503, detail="System components not initialized")
    
    async def generate_stream():
        try:
            # Create conversation context
            context = ConversationContext(
                history=llm_generator.get_conversation_history(request.session_id),
                user_profile=request.user_profile,
                session_id=request.session_id,
                domain_context="general_inquiry",
                timestamp=datetime.now()
            )
            
            # Process query through RAG pipeline
            rag_result = await rag_pipeline.process_query(
                request.query,
                user_context={"history": context.history}
            )
            
            # Update domain context
            context.domain_context = rag_result.get("intent", "general_inquiry")
            
            # Generate streaming response
            generation_config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True
            )
            
            llm_result = await llm_generator.generate_response(
                request.query,
                rag_result.get("retrieval_results", []),
                context,
                generation_config
            )
            
            # Stream the response
            async for chunk in llm_result.get("response_stream", []):
                yield f"data: {chunk}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(generate_stream(), media_type="text/plain")

@app.post("/documents/upload")
async def upload_document(document: DocumentUpload):
    """Upload document to vector store"""
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Create document object
        doc = Document(
            id=f"upload_{datetime.now().timestamp()}",
            content=document.content,
            metadata={
                **document.metadata,
                "upload_timestamp": datetime.now().isoformat(),
                "source": "user_upload"
            },
            modality=document.modality,
            source="user_upload",
            timestamp=datetime.now().isoformat()
        )
        
        # Add to vector store
        vector_store.add_documents([doc])
        
        return {
            "message": "Document uploaded successfully",
            "document_id": doc.id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/documents/upload/file")
async def upload_file(file: UploadFile = File(...)):
    """Upload file and extract content"""
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        # Read file content
        content = await file.read()
        
        # Determine file type and extract content
        if file.filename.endswith('.pdf'):
            from ..crawler.pdf_processor import PDFProcessor
            processor = PDFProcessor()
            # Save temporarily and process
            temp_path = f"temp_{file.filename}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            text_content = processor.extract_text(temp_path)
            os.unlink(temp_path)
        elif file.filename.endswith(('.txt', '.md')):
            text_content = content.decode('utf-8')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Create document
        doc = Document(
            id=f"file_{datetime.now().timestamp()}",
            content=text_content,
            metadata={
                "filename": file.filename,
                "file_size": len(content),
                "upload_timestamp": datetime.now().isoformat(),
                "source": "file_upload"
            },
            modality="text",
            source="file_upload",
            timestamp=datetime.now().isoformat()
        )
        
        # Add to vector store
        vector_store.add_documents([doc])
        
        return {
            "message": "File uploaded and processed successfully",
            "document_id": doc.id,
            "filename": file.filename,
            "content_length": len(text_content),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.post("/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start web crawling process"""
    
    if not crawler:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        # Start crawling in background
        background_tasks.add_task(
            crawl_and_process,
            request.urls,
            request.max_pages,
            request.max_depth
        )
        
        return {
            "message": "Crawling started",
            "urls": request.urls,
            "max_pages": request.max_pages,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting crawl: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting crawl: {str(e)}")

async def crawl_and_process(urls: List[str], max_pages: int, max_depth: int):
    """Background task for crawling and processing"""
    
    try:
        logger.info(f"Starting crawl for {len(urls)} URLs")
        
        # Update crawler config
        crawler.config.max_pages = max_pages
        crawler.config.max_depth = max_depth
        
        # Crawl websites
        crawled_docs = await crawler.crawl_website(urls)
        
        # Convert to vector store documents
        documents = []
        for crawled_doc in crawled_docs:
            doc = Document(
                id=f"crawl_{crawled_doc.content_hash}",
                content=crawled_doc.content,
                metadata={
                    **crawled_doc.metadata,
                    "url": crawled_doc.url,
                    "title": crawled_doc.title,
                    "crawl_timestamp": crawled_doc.crawl_timestamp.isoformat()
                },
                modality="text",
                source="web_crawl",
                timestamp=crawled_doc.crawl_timestamp.isoformat()
            )
            documents.append(doc)
        
        # Add to vector store
        if documents and vector_store:
            vector_store.add_documents(documents)
        
        # Update knowledge graph
        if documents and knowledge_graph:
            document_texts = [doc.content for doc in documents]
            knowledge_graph.incremental_update(document_texts)
        
        logger.info(f"Crawling completed. Processed {len(documents)} documents")
        
    except Exception as e:
        logger.error(f"Error in crawl and process: {e}")

@app.get("/knowledge-graph/stats")
async def get_kg_stats():
    """Get knowledge graph statistics"""
    
    if not knowledge_graph:
        raise HTTPException(status_code=503, detail="Knowledge graph not initialized")
    
    try:
        stats = knowledge_graph.get_graph_statistics()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting KG stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting KG stats: {str(e)}")

@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics"""
    
    if not vector_store:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    
    try:
        stats = vector_store.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting vector store stats: {str(e)}")

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation session"""
    
    if not llm_generator:
        raise HTTPException(status_code=503, detail="LLM generator not initialized")
    
    try:
        llm_generator.clear_conversation_memory(session_id)
        
        return {
            "message": f"Session {session_id} cleared",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    # Get settings
    settings = get_settings()
    
    # Run the application
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
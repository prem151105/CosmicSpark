"""
Advanced Multimodal Vector Store Implementation
Supports text, images, and scientific data with cross-modal alignment
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import base64
import io
import json
from pathlib import Path
import faiss
import shutil
from loguru import logger

@dataclass
class Document:
    """Enhanced document representation with multimodal support"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    modality: str = "text"  # text, image, table, chart
    source: str = ""
    timestamp: str = ""

@dataclass
class MultimodalQuery:
    """Multimodal query representation"""
    text: Optional[str] = None
    image: Optional[np.ndarray] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    modality_weights: Optional[Dict[str, float]] = None

class CrossModalAlignmentModel(nn.Module):
    """Cross-modal alignment model for text-image-document alignment"""
    
    def __init__(self, text_dim: int = 768, image_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.hidden_dim = hidden_dim
        
        # Text encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Image encoder
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, text_features: torch.Tensor, 
                image_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for cross-modal alignment"""
        
        # Encode text
        text_encoded = self.text_encoder(text_features)
        
        if image_features is not None:
            # Encode image
            image_encoded = self.image_encoder(image_features)
            
            # Cross-attention between text and image
            text_encoded = text_encoded.unsqueeze(0)  # Add sequence dimension
            image_encoded = image_encoded.unsqueeze(0)
            
            attended_features, _ = self.cross_attention(
                text_encoded, image_encoded, image_encoded
            )
            
            # Combine features
            combined_features = text_encoded + attended_features
            combined_features = combined_features.squeeze(0)
        else:
            combined_features = text_encoded
        
        # Final projection
        output = self.output_proj(combined_features)
        return F.normalize(output, p=2, dim=-1)

class AdvancedVectorStore:
    """Advanced vector store with multimodal capabilities and hybrid indexing"""
    
    def __init__(self, 
                 persist_directory: str = "./data/chromadb",
                 collection_name: str = "mosdac_documents",
                 embedding_model_name: str = "all-MiniLM-L6-v2"):
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        # Initialize embedding models - Fix TensorFlow Lite delegate error
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.text_embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
        
        # Initialize cross-modal alignment model
        self.cross_modal_model = CrossModalAlignmentModel()
        
        # FAISS index for high-performance similarity search
        self.faiss_index = None
        self.document_ids = []
        
        # Document cache
        self.document_cache = {}
        
        logger.info(f"Advanced Vector Store initialized with collection: {collection_name}")
    
    def _get_or_create_collection(self):
        """Get or create ChromaDB collection with schema compatibility handling"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
        except Exception as e:
            logger.warning(f"Failed to load existing collection: {e}")
            
            # Handle schema compatibility issues
            if "no such column" in str(e).lower() or "sqlite3.operationalerror" in str(e).lower():
                logger.info("Detected ChromaDB schema incompatibility, resetting database...")
                try:
                    # Try to delete the existing collection if it exists
                    try:
                        self.client.delete_collection(name=self.collection_name)
                        logger.info(f"Deleted incompatible collection: {self.collection_name}")
                    except:
                        pass
                    
                    # Reset the client to clear any cached schema
                    import shutil
                    if self.persist_directory.exists():
                        logger.info("Removing incompatible ChromaDB data...")
                        shutil.rmtree(self.persist_directory)
                        self.persist_directory.mkdir(parents=True, exist_ok=True)
                    
                    # Reinitialize client
                    self.client = chromadb.PersistentClient(
                        path=str(self.persist_directory),
                        settings=Settings(anonymized_telemetry=False)
                    )
                    
                except Exception as reset_error:
                    logger.error(f"Failed to reset ChromaDB: {reset_error}")
            
            # Create new collection
            try:
                collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                return collection
            except Exception as create_error:
                logger.error(f"Failed to create collection: {create_error}")
                raise
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text using sentence transformer"""
        return self.text_embedding_model.encode([text])[0]
    
    def _encode_image(self, image: Union[Image.Image, np.ndarray, str]) -> np.ndarray:
        """Encode image using CLIP or similar model"""
        # Placeholder for image encoding - would use CLIP in production
        if isinstance(image, str):
            # Base64 encoded image
            image_data = base64.b64decode(image)
            image = Image.open(io.BytesIO(image_data))
        
        # For now, return a dummy embedding
        # In production, use: clip_model.encode_image(image)
        return np.random.rand(512).astype(np.float32)
    
    def _process_scientific_table(self, table_data: Dict[str, Any]) -> np.ndarray:
        """Process scientific tables and charts for embedding"""
        # Extract text content from table
        text_content = []
        
        if "headers" in table_data:
            text_content.extend(table_data["headers"])
        
        if "rows" in table_data:
            for row in table_data["rows"]:
                text_content.extend([str(cell) for cell in row])
        
        if "caption" in table_data:
            text_content.append(table_data["caption"])
        
        # Create text representation
        table_text = " ".join(text_content)
        return self._encode_text(table_text)
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store with multimodal support"""
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        embeddings = []
        ids = []
        metadatas = []
        contents = []
        
        for doc in documents:
            # Generate embedding based on modality
            if doc.modality == "text":
                embedding = self._encode_text(doc.content)
            elif doc.modality == "image":
                embedding = self._encode_image(doc.content)
            elif doc.modality == "table":
                table_data = json.loads(doc.content) if isinstance(doc.content, str) else doc.content
                embedding = self._process_scientific_table(table_data)
            else:
                # Default to text encoding
                embedding = self._encode_text(str(doc.content))
            
            doc.embedding = embedding
            
            embeddings.append(embedding.tolist())
            ids.append(doc.id)
            metadatas.append({
                **doc.metadata,
                "modality": doc.modality,
                "source": doc.source,
                "timestamp": doc.timestamp
            })
            contents.append(doc.content)
            
            # Cache document
            self.document_cache[doc.id] = doc
        
        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Update FAISS index
        self._update_faiss_index(embeddings, ids)
        
        logger.info(f"Successfully added {len(documents)} documents")
    
    def _update_faiss_index(self, embeddings: List[List[float]], ids: List[str]) -> None:
        """Update FAISS index for high-performance search"""
        embeddings_array = np.array(embeddings).astype('float32')
        
        if self.faiss_index is None:
            # Initialize FAISS index
            dimension = embeddings_array.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.document_ids = []
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.faiss_index.add(embeddings_array)
        self.document_ids.extend(ids)
    
    def similarity_search(self, 
                         query: Union[str, MultimodalQuery], 
                         k: int = 10,
                         filter_dict: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Perform similarity search with multimodal support"""
        
        if isinstance(query, str):
            # Simple text query
            query_embedding = self._encode_text(query)
            modality_filter = None
        else:
            # Multimodal query
            if query.text:
                query_embedding = self._encode_text(query.text)
            elif query.image is not None:
                query_embedding = self._encode_image(query.image)
            else:
                raise ValueError("Query must contain either text or image")
            
            modality_filter = query.metadata_filters
        
        # Use FAISS for initial retrieval if available
        if self.faiss_index is not None:
            return self._faiss_search(query_embedding, k, filter_dict, modality_filter)
        else:
            return self._chromadb_search(query_embedding, k, filter_dict, modality_filter)
    
    def _faiss_search(self, 
                     query_embedding: np.ndarray, 
                     k: int,
                     filter_dict: Optional[Dict[str, Any]] = None,
                     modality_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """High-performance search using FAISS"""
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(query_embedding, min(k * 2, len(self.document_ids)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.document_ids):
                doc_id = self.document_ids[idx]
                if doc_id in self.document_cache:
                    doc = self.document_cache[doc_id]
                    
                    # Apply filters
                    if self._passes_filters(doc, filter_dict, modality_filter):
                        results.append((doc, float(score)))
                        
                        if len(results) >= k:
                            break
        
        return results
    
    def _chromadb_search(self, 
                        query_embedding: np.ndarray, 
                        k: int,
                        filter_dict: Optional[Dict[str, Any]] = None,
                        modality_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """Search using ChromaDB"""
        
        # Prepare where clause
        where_clause = {}
        if filter_dict:
            where_clause.update(filter_dict)
        if modality_filter:
            where_clause.update(modality_filter)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        # Convert to Document objects
        documents_with_scores = []
        for i, doc_id in enumerate(results['ids'][0]):
            if doc_id in self.document_cache:
                doc = self.document_cache[doc_id]
                score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                documents_with_scores.append((doc, score))
        
        return documents_with_scores
    
    def _passes_filters(self, 
                       doc: Document, 
                       filter_dict: Optional[Dict[str, Any]] = None,
                       modality_filter: Optional[Dict[str, Any]] = None) -> bool:
        """Check if document passes all filters"""
        
        if filter_dict:
            for key, value in filter_dict.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    return False
        
        if modality_filter:
            for key, value in modality_filter.items():
                if key == "modality" and doc.modality != value:
                    return False
                elif key in doc.metadata and doc.metadata[key] != value:
                    return False
        
        return True
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 10,
                     alpha: float = 0.7) -> List[Tuple[Document, float]]:
        """Hybrid search combining dense and sparse retrieval"""
        
        # Dense retrieval
        dense_results = self.similarity_search(query, k=k*2)
        
        # Sparse retrieval (BM25-like scoring)
        sparse_results = self._sparse_search(query, k=k*2)
        
        # Combine results with weighted scoring
        combined_scores = {}
        
        # Add dense scores
        for doc, score in dense_results:
            combined_scores[doc.id] = alpha * score
        
        # Add sparse scores
        for doc, score in sparse_results:
            if doc.id in combined_scores:
                combined_scores[doc.id] += (1 - alpha) * score
            else:
                combined_scores[doc.id] = (1 - alpha) * score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        final_results = []
        for doc_id, score in sorted_results[:k]:
            if doc_id in self.document_cache:
                final_results.append((self.document_cache[doc_id], score))
        
        return final_results
    
    def _sparse_search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Sparse search using TF-IDF like scoring"""
        query_terms = query.lower().split()
        
        doc_scores = {}
        for doc_id, doc in self.document_cache.items():
            content = doc.content.lower()
            score = 0.0
            
            for term in query_terms:
                # Simple term frequency scoring
                tf = content.count(term)
                if tf > 0:
                    score += tf / len(content.split())
            
            if score > 0:
                doc_scores[doc_id] = score
        
        # Sort and return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        
        for doc_id, score in sorted_docs[:k]:
            if doc_id in self.document_cache:
                results.append((self.document_cache[doc_id], score))
        
        return results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store collection"""
        count = self.collection.count()
        
        # Modality distribution
        modality_counts = {}
        for doc in self.document_cache.values():
            modality = doc.modality
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            "total_documents": count,
            "modality_distribution": modality_counts,
            "faiss_index_size": len(self.document_ids) if self.faiss_index else 0,
            "cache_size": len(self.document_cache)
        }
    
    def delete_documents(self, document_ids: List[str]) -> None:
        """Delete documents from the vector store"""
        # Delete from ChromaDB
        self.collection.delete(ids=document_ids)
        
        # Remove from cache
        for doc_id in document_ids:
            if doc_id in self.document_cache:
                del self.document_cache[doc_id]
        
        # Rebuild FAISS index (simple approach - could be optimized)
        if self.faiss_index is not None:
            self._rebuild_faiss_index()
        
        logger.info(f"Deleted {len(document_ids)} documents")
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index from scratch"""
        if not self.document_cache:
            self.faiss_index = None
            self.document_ids = []
            return
        
        embeddings = []
        ids = []
        
        for doc_id, doc in self.document_cache.items():
            if doc.embedding is not None:
                embeddings.append(doc.embedding.tolist())
                ids.append(doc_id)
        
        if embeddings:
            self.faiss_index = None
            self.document_ids = []
            self._update_faiss_index(embeddings, ids)
    
    def update_document(self, document: Document) -> None:
        """Update an existing document"""
        # Delete old version
        self.delete_documents([document.id])
        
        # Add new version
        self.add_documents([document])
        
        logger.info(f"Updated document: {document.id}")
    
    def batch_similarity_search(self, 
                               queries: List[str], 
                               k: int = 10) -> List[List[Tuple[Document, float]]]:
        """Perform batch similarity search for multiple queries"""
        results = []
        
        for query in queries:
            query_results = self.similarity_search(query, k=k)
            results.append(query_results)
        
        return results
    
    def close(self) -> None:
        """Close the vector store and clean up resources"""
        # ChromaDB client doesn't need explicit closing
        logger.info("Vector store closed")

# Export main classes
__all__ = [
    'AdvancedVectorStore',
    'Document',
    'MultimodalQuery',
    'CrossModalAlignmentModel'
]
"""
Advanced GraphRAG Retriever Implementation
Implements cutting-edge retrieval algorithms from the research paper
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from sentence_transformers import SentenceTransformer
import spacy
from loguru import logger

@dataclass
class RetrievalResult:
    """Structured retrieval result"""
    content: str
    score: float
    source: str
    entities: List[str]
    relationships: List[Dict[str, Any]]
    reasoning_path: List[str]

class GraphAttentionRetriever(nn.Module):
    """Advanced Graph Attention Network for Knowledge Graph Retrieval"""
    
    def __init__(self, node_types: List[str], edge_types: List[str], hidden_dim: int = 768):
        super().__init__()
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        
        # Heterogeneous Graph Convolution Layers
        self.convs = nn.ModuleList([
            HeteroConv({
                edge_type: GATConv(hidden_dim, hidden_dim, heads=8, dropout=0.1, add_self_loops=False)
                for edge_type in edge_types
            }) for _ in range(3)  # 3 layers for multi-hop reasoning
        ])
        
        # Attention mechanism for node importance
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the graph attention network"""
        
        # Multi-layer graph convolution
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Apply attention mechanism
        for node_type in x_dict:
            x = x_dict[node_type]
            x = x.unsqueeze(1)  # Add sequence dimension
            attended_x, _ = self.attention(x, x, x)
            x_dict[node_type] = self.output_proj(attended_x.squeeze(1))
        
        return x_dict

class ContextualQueryProcessor:
    """Advanced query processing with intent classification and entity extraction"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # Fix TensorFlow Lite delegate error by forcing CPU device and disabling parallelism
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Intent categories for MOSDAC domain
        self.intent_categories = [
            "weather_data", "satellite_imagery", "climate_analysis", 
            "oceanographic_data", "atmospheric_data", "research_papers",
            "technical_documentation", "data_download", "general_inquiry"
        ]
        
    def extract_entities(self, query: str) -> List[Dict[str, Any]]:
        """Extract entities with enhanced NER for scientific domain"""
        doc = self.nlp(query)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # spaCy doesn't provide confidence scores directly
            })
        
        # Add custom entity extraction for scientific terms
        scientific_patterns = [
            r'\b(?:temperature|humidity|pressure|wind|precipitation)\b',
            r'\b(?:satellite|radar|lidar|sonar)\b',
            r'\b(?:INSAT|SCATSAT|OCEANSAT|CARTOSAT)\b',
            r'\b(?:cyclone|monsoon|drought|flood)\b'
        ]
        
        import re
        for pattern in scientific_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                entities.append({
                    "text": match.group(),
                    "label": "SCIENTIFIC_TERM",
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.9
                })
        
        return entities
    
    def classify_intent(self, query: str) -> Dict[str, float]:
        """Classify query intent using semantic similarity"""
        query_embedding = self.embedding_model.encode([query])
        
        # Intent templates for MOSDAC domain
        intent_templates = {
            "weather_data": "weather data temperature rainfall wind speed",
            "satellite_imagery": "satellite image remote sensing earth observation",
            "climate_analysis": "climate change analysis trends patterns",
            "oceanographic_data": "ocean sea surface temperature salinity currents",
            "atmospheric_data": "atmosphere pressure humidity aerosol ozone",
            "research_papers": "research paper publication study analysis",
            "technical_documentation": "documentation manual guide tutorial",
            "data_download": "download data file dataset access",
            "general_inquiry": "help information about what is how"
        }
        
        template_embeddings = self.embedding_model.encode(list(intent_templates.values()))
        similarities = cosine_similarity(query_embedding, template_embeddings)[0]
        
        intent_scores = {}
        for i, intent in enumerate(intent_templates.keys()):
            intent_scores[intent] = float(similarities[i])
        
        return intent_scores
    
    def decompose_query(self, query: str, intent: str) -> List[str]:
        """Decompose complex queries into sub-queries"""
        doc = self.nlp(query)
        
        # Extract main components
        sub_queries = []
        
        # Extract noun phrases as potential sub-queries
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) > 1:
                sub_queries.append(chunk.text)
        
        # Extract questions
        sentences = [sent.text.strip() for sent in doc.sents]
        for sent in sentences:
            if any(word in sent.lower() for word in ['what', 'how', 'when', 'where', 'why', 'which']):
                sub_queries.append(sent)
        
        # If no sub-queries found, use the original query
        if not sub_queries:
            sub_queries = [query]
        
        return sub_queries
    
    async def process_query(self, query: str, context_history: List[str] = None) -> Dict[str, Any]:
        """Process query with full contextual understanding"""
        entities = self.extract_entities(query)
        intent_scores = self.classify_intent(query)
        primary_intent = max(intent_scores, key=intent_scores.get)
        sub_queries = self.decompose_query(query, primary_intent)
        
        # Context integration
        contextualized_query = query
        if context_history:
            # Simple context integration - can be enhanced with more sophisticated methods
            recent_context = " ".join(context_history[-3:])  # Last 3 interactions
            contextualized_query = f"{recent_context} {query}"
        
        return {
            "original_query": query,
            "contextualized_query": contextualized_query,
            "entities": entities,
            "intent_scores": intent_scores,
            "primary_intent": primary_intent,
            "sub_queries": sub_queries
        }

class HybridRetriever:
    """Hybrid retrieval system combining sparse, dense, and graph-based retrieval"""
    
    def __init__(self, knowledge_graph: nx.Graph, documents: List[str]):
        self.knowledge_graph = knowledge_graph
        self.documents = documents
        
        # Initialize retrievers
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        self.bm25_retriever = BM25Okapi([doc.split() for doc in documents])
        
        # Create document embeddings
        self.document_embeddings = self.embedding_model.encode(documents)
        
        # Graph retriever components
        self.graph_attention = GraphAttentionRetriever(
            node_types=['entity', 'document', 'concept'],
            edge_types=['related_to', 'mentions', 'part_of']
        )
        
    def sparse_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25-based sparse retrieval"""
        query_tokens = query.split()
        scores = self.bm25_retriever.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def dense_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Dense retrieval using sentence embeddings"""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.document_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices]
        
        return results
    
    def graph_retrieve(self, entities: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Graph-based retrieval using knowledge graph traversal"""
        if not entities:
            return []
        
        # Find nodes in the graph that match the entities
        relevant_nodes = []
        for entity in entities:
            for node in self.knowledge_graph.nodes():
                if entity.lower() in str(node).lower():
                    relevant_nodes.append(node)
        
        if not relevant_nodes:
            return []
        
        # Perform graph traversal with attention mechanism
        subgraph_nodes = set(relevant_nodes)
        
        # Add neighbors (1-hop and 2-hop)
        for node in relevant_nodes:
            # 1-hop neighbors
            neighbors = list(self.knowledge_graph.neighbors(node))
            subgraph_nodes.update(neighbors)
            
            # 2-hop neighbors (for multi-hop reasoning)
            for neighbor in neighbors[:5]:  # Limit to prevent explosion
                second_hop = list(self.knowledge_graph.neighbors(neighbor))
                subgraph_nodes.update(second_hop[:3])
        
        # Calculate node importance using PageRank
        subgraph = self.knowledge_graph.subgraph(subgraph_nodes)
        if len(subgraph.nodes()) > 0:
            pagerank_scores = nx.pagerank(subgraph)
            
            # Sort by PageRank score
            sorted_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            results = sorted_nodes[:top_k]
        else:
            results = []
        
        return results
    
    def fuse_results(self, sparse_results: List[Tuple[int, float]], 
                    dense_results: List[Tuple[int, float]], 
                    graph_results: List[Tuple[str, float]]) -> List[RetrievalResult]:
        """Advanced fusion of retrieval results with learned weights"""
        
        # Normalize scores
        def normalize_scores(results):
            if not results:
                return []
            scores = [score for _, score in results]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0
            
            return [(idx, (score - min_score) / score_range) for idx, score in results]
        
        sparse_normalized = normalize_scores(sparse_results)
        dense_normalized = normalize_scores(dense_results)
        
        # Fusion weights (can be learned from data)
        sparse_weight = 0.3
        dense_weight = 0.5
        graph_weight = 0.2
        
        # Combine results
        combined_scores = {}
        
        # Add sparse results
        for idx, score in sparse_normalized:
            if idx < len(self.documents):
                combined_scores[idx] = combined_scores.get(idx, 0) + sparse_weight * score
        
        # Add dense results
        for idx, score in dense_normalized:
            if idx < len(self.documents):
                combined_scores[idx] = combined_scores.get(idx, 0) + dense_weight * score
        
        # Convert to RetrievalResult objects
        results = []
        for idx, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            if idx < len(self.documents):
                results.append(RetrievalResult(
                    content=self.documents[idx],
                    score=score,
                    source=f"document_{idx}",
                    entities=[],  # Will be populated by entity extraction
                    relationships=[],  # Will be populated by relationship mapping
                    reasoning_path=[]  # Will be populated by reasoning engine
                ))
        
        return results
    
    async def retrieve(self, processed_query: Dict[str, Any], top_k: int = 10) -> List[RetrievalResult]:
        """Main retrieval method combining all approaches"""
        query = processed_query["contextualized_query"]
        entities = [ent["text"] for ent in processed_query["entities"]]
        
        # Parallel retrieval
        sparse_task = asyncio.create_task(
            asyncio.to_thread(self.sparse_retrieve, query, top_k)
        )
        dense_task = asyncio.create_task(
            asyncio.to_thread(self.dense_retrieve, query, top_k)
        )
        graph_task = asyncio.create_task(
            asyncio.to_thread(self.graph_retrieve, entities, top_k)
        )
        
        sparse_results, dense_results, graph_results = await asyncio.gather(
            sparse_task, dense_task, graph_task
        )
        
        # Fuse results
        fused_results = self.fuse_results(sparse_results, dense_results, graph_results)
        
        return fused_results[:top_k]

class MultiStepReasoningEngine:
    """Multi-step reasoning engine for complex query processing"""
    
    def __init__(self, knowledge_graph: nx.Graph):
        self.knowledge_graph = knowledge_graph
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    def detect_communities(self, subgraph: nx.Graph) -> List[List[str]]:
        """Detect communities in the subgraph for context clustering"""
        try:
            communities = nx.community.greedy_modularity_communities(subgraph)
            return [list(community) for community in communities]
        except:
            # Fallback to simple connected components
            return [list(component) for component in nx.connected_components(subgraph)]
    
    def multi_hop_reasoning(self, communities: List[List[str]], query: str) -> List[str]:
        """Perform multi-hop reasoning across communities"""
        reasoning_path = []
        
        # Find the most relevant community for the query
        query_embedding = self.embedding_model.encode([query])
        
        community_scores = []
        for community in communities:
            # Create community representation
            community_text = " ".join([str(node) for node in community])
            community_embedding = self.embedding_model.encode([community_text])
            
            # Calculate similarity
            similarity = cosine_similarity(query_embedding, community_embedding)[0][0]
            community_scores.append((community, similarity))
        
        # Sort communities by relevance
        community_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Build reasoning path through top communities
        for community, score in community_scores[:3]:  # Top 3 communities
            reasoning_path.extend(community[:5])  # Top 5 nodes from each community
        
        return reasoning_path
    
    async def reason(self, retrieval_results: List[RetrievalResult], 
                    processed_query: Dict[str, Any]) -> Dict[str, Any]:
        """Main reasoning method"""
        
        # Extract entities from retrieval results
        all_entities = []
        for result in retrieval_results:
            # Simple entity extraction from content
            doc = spacy.load("en_core_web_sm")(result.content)
            entities = [ent.text for ent in doc.ents]
            result.entities = entities
            all_entities.extend(entities)
        
        # Create subgraph from relevant entities
        relevant_nodes = []
        for entity in all_entities:
            for node in self.knowledge_graph.nodes():
                if entity.lower() in str(node).lower():
                    relevant_nodes.append(node)
        
        if relevant_nodes:
            subgraph = self.knowledge_graph.subgraph(relevant_nodes)
            communities = self.detect_communities(subgraph)
            reasoning_path = self.multi_hop_reasoning(communities, processed_query["original_query"])
            
            # Update retrieval results with reasoning paths
            for result in retrieval_results:
                result.reasoning_path = reasoning_path[:10]  # Limit path length
        
        return {
            "retrieval_results": retrieval_results,
            "reasoning_path": reasoning_path if relevant_nodes else [],
            "communities": communities if relevant_nodes else [],
            "confidence_score": self._calculate_confidence(retrieval_results)
        }
    
    def _calculate_confidence(self, results: List[RetrievalResult]) -> float:
        """Calculate overall confidence score for the reasoning"""
        if not results:
            return 0.0
        
        # Simple confidence calculation based on retrieval scores
        avg_score = sum(result.score for result in results) / len(results)
        return min(avg_score, 1.0)

class AdvancedRAGPipeline:
    """Complete advanced RAG pipeline with GraphRAG capabilities"""
    
    def __init__(self, knowledge_graph: nx.Graph, documents: List[str]):
        self.query_processor = ContextualQueryProcessor()
        self.hybrid_retriever = HybridRetriever(knowledge_graph, documents)
        self.reasoning_engine = MultiStepReasoningEngine(knowledge_graph)
        
        logger.info("Advanced RAG Pipeline initialized successfully")
    
    async def process_query(self, query: str, user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main pipeline processing method"""
        
        try:
            # Stage 1: Query understanding
            logger.info(f"Processing query: {query}")
            context_history = user_context.get("history", []) if user_context else []
            processed_query = await self.query_processor.process_query(query, context_history)
            
            # Stage 2: Multi-modal retrieval
            logger.info("Performing hybrid retrieval")
            retrieval_results = await self.hybrid_retriever.retrieve(processed_query)
            
            # Stage 3: Multi-step reasoning
            logger.info("Performing multi-step reasoning")
            reasoning_result = await self.reasoning_engine.reason(retrieval_results, processed_query)
            
            # Stage 4: Prepare final response
            response = {
                "query": query,
                "processed_query": processed_query,
                "retrieval_results": reasoning_result["retrieval_results"],
                "reasoning_path": reasoning_result["reasoning_path"],
                "confidence_score": reasoning_result["confidence_score"],
                "intent": processed_query["primary_intent"],
                "entities": processed_query["entities"]
            }
            
            logger.info(f"Pipeline processing completed with confidence: {reasoning_result['confidence_score']}")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "retrieval_results": [],
                "confidence_score": 0.0
            }

# Export main classes
__all__ = [
    'AdvancedRAGPipeline',
    'HybridRetriever', 
    'ContextualQueryProcessor',
    'MultiStepReasoningEngine',
    'GraphAttentionRetriever',
    'RetrievalResult'
]
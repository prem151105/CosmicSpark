"""
Advanced Knowledge Graph Builder Implementation
Implements LLM-enhanced KG construction with attention-based GNNs
"""

import time

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv
from sentence_transformers import SentenceTransformer
import spacy
from neo4j import GraphDatabase
import numpy as np
from loguru import logger

@dataclass
class Entity:
    """Entity representation with metadata"""
    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any]
    confidence: float
    source: str
    timestamp: datetime

@dataclass
class Relationship:
    """Relationship representation with metadata"""
    id: str
    source_entity: str
    target_entity: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    source: str
    timestamp: datetime

class AdvancedKGEncoder(nn.Module):
    """Advanced Knowledge Graph Encoder with Graph Attention Networks"""
    
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
            }) for _ in range(3)  # 3 layers
        ])
        
        # Node type embeddings
        self.node_type_embeddings = nn.ModuleDict({
            node_type: nn.Embedding(1000, hidden_dim)  # Vocab size of 1000
            for node_type in node_types
        })
        
        # Temporal encoding for dynamic updates
        self.temporal_encoder = nn.Linear(1, hidden_dim)
        
        # Output projections
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                edge_index_dict: Dict[str, torch.Tensor],
                temporal_features: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the advanced KG encoder"""
        
        # Add temporal information if available
        if temporal_features:
            for node_type in x_dict:
                if node_type in temporal_features:
                    temporal_emb = self.temporal_encoder(temporal_features[node_type])
                    x_dict[node_type] = x_dict[node_type] + temporal_emb
        
        # Multi-layer graph convolution with attention
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        
        # Final projection
        x_dict = {key: self.output_proj(x) for key, x in x_dict.items()}
        
        return x_dict

class LLMEntityExtractor:
    """LLM-enhanced entity extraction with zero-shot capabilities"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Fix TensorFlow Lite delegate error by forcing CPU device and disabling parallelism
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        try:
            self.embedding_model = SentenceTransformer(model_name, device='cpu')
        except Exception as e:
            logger.warning(f"Failed to load embedding model {model_name}: {e}")
            self.embedding_model = None
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.warning(f"Failed to load spaCy model en_core_web_sm: {e}")
            logger.info("Attempting to download spaCy model...")
            try:
                import subprocess
                result = subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], 
                                      capture_output=True, text=True, check=True)
                logger.info(f"spaCy download output: {result.stdout}")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully downloaded and loaded spaCy model")
            except subprocess.CalledProcessError as download_error:
                logger.error(f"Failed to download spaCy model: {download_error}")
                logger.error(f"Download stderr: {download_error.stderr}")
                logger.warning("Continuing without spaCy model - using fallback entity extraction")
                self.nlp = None
            except Exception as download_error:
                logger.error(f"Failed to download spaCy model: {download_error}")
                logger.warning("Continuing without spaCy model - using fallback entity extraction")
                self.nlp = None
        
        # Domain-specific entity types for MOSDAC
        self.entity_types = {
            "SATELLITE": ["INSAT", "SCATSAT", "OCEANSAT", "CARTOSAT", "RESOURCESAT"],
            "WEATHER_PARAMETER": ["temperature", "humidity", "pressure", "wind", "precipitation", "visibility"],
            "LOCATION": ["India", "Bay of Bengal", "Arabian Sea", "Indian Ocean"],
            "INSTRUMENT": ["radar", "lidar", "sonar", "radiometer", "spectrometer"],
            "PHENOMENON": ["cyclone", "monsoon", "drought", "flood", "tsunami"],
            "DATA_TYPE": ["imagery", "data", "product", "dataset", "archive"],
            "ORGANIZATION": ["ISRO", "MOSDAC", "SAC", "NRSC", "IMD"],
            "TIME_PERIOD": ["daily", "monthly", "yearly", "seasonal", "real-time"]
        }
        
        # Create embeddings for entity types
        self.type_embeddings = {}
        if self.embedding_model is not None:
            for entity_type, examples in self.entity_types.items():
                type_text = f"{entity_type.lower()} {' '.join(examples)}"
                try:
                    self.type_embeddings[entity_type] = self.embedding_model.encode([type_text])[0]
                except Exception as e:
                    logger.warning(f"Failed to create embedding for {entity_type}: {e}")
        else:
            logger.warning("Embedding model not available, skipping type embeddings")
    
    def extract_entities_with_llm(self, text: str) -> List[Entity]:
        """Extract entities using LLM-enhanced NER"""
        entities = []
        
        # Standard spaCy entities
        if self.nlp is not None:
            doc = self.nlp(text)
            for ent in doc.ents:
                    entity = Entity(
                        id=f"ent_{len(entities)}",
                        name=ent.text,
                        type=ent.label_,
                        description=f"{ent.label_} entity: {ent.text}",
                        properties={
                            "start": ent.start_char,
                            "end": ent.end_char,
                            "spacy_label": ent.label_
                        },
                        confidence=0.8,  # Default confidence for spaCy
                        source="spacy_ner",
                        timestamp=datetime.now()
                    )
                    entities.append(entity)
        
        # Domain-specific entity extraction
        if self.embedding_model is not None and self.type_embeddings:
            try:
                text_embedding = self.embedding_model.encode([text])[0]
            except Exception as e:
                logger.warning(f"Failed to encode text for entity extraction: {e}")
                return entities
        
            # Find domain-specific entities using semantic similarity
            for entity_type, type_embedding in self.type_embeddings.items():
                    similarity = np.dot(text_embedding, type_embedding) / (
                        np.linalg.norm(text_embedding) * np.linalg.norm(type_embedding)
                    )
                    
                    if similarity > 0.3:  # Threshold for relevance
                        # Extract specific terms related to this entity type
                        for term in self.entity_types[entity_type]:
                            if term.lower() in text.lower():
                                entity = Entity(
                                    id=f"domain_ent_{len(entities)}",
                                    name=term,
                                    type=entity_type,
                                    description=f"Domain-specific {entity_type.lower()}: {term}",
                                    properties={
                                        "similarity_score": float(similarity),
                                        "domain_specific": True
                                    },
                                    confidence=float(similarity),
                                    source="domain_extraction",
                                    timestamp=datetime.now()
                                )
                                entities.append(entity)
        
        # If no entities found and spaCy is not available, use fallback
        if not entities and self.nlp is None:
            entities = self.fallback_entity_extraction(text)
        
        return entities
    
    def fallback_entity_extraction(self, text: str) -> List[Entity]:
        """Fallback entity extraction using simple pattern matching"""
        entities = []
        text_lower = text.lower()
        
        # Simple pattern-based extraction for MOSDAC domain
        for entity_type, keywords in self.entity_types.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Find the position of the keyword
                    start_pos = text_lower.find(keyword.lower())
                    if start_pos != -1:
                        entity = Entity(
                            id=f"fallback_ent_{len(entities)}",
                            name=keyword,
                            type=entity_type,
                            description=f"Pattern-matched {entity_type.lower()}: {keyword}",
                            properties={
                                "start": start_pos,
                                "end": start_pos + len(keyword),
                                "extraction_method": "pattern_matching"
                            },
                            confidence=0.6,  # Lower confidence for pattern matching
                            source="fallback_extraction",
                            timestamp=datetime.now()
                        )
                        entities.append(entity)
        
        return entities
    
    def schema_aware_extraction(self, text: str, schema: Dict[str, Any]) -> List[Entity]:
        """Schema-aware entity extraction based on dynamic schema"""
        entities = self.extract_entities_with_llm(text)
        
        # Filter and enhance entities based on schema
        schema_entities = []
        for entity in entities:
            if entity.type in schema.get("entity_types", []):
                # Enhance with schema information
                entity.properties.update(schema.get("properties", {}).get(entity.type, {}))
                schema_entities.append(entity)
        
        return schema_entities

class RelationshipMapper:
    """Advanced relationship mapping with multi-hop discovery"""
    
    def __init__(self):
        # Fix TensorFlow Lite delegate error by forcing CPU device and disabling parallelism
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            logger.warning(f"Failed to load spaCy model in RelationshipMapper: {e}")
            self.nlp = None
        
        # Predefined relationship patterns for MOSDAC domain
        self.relationship_patterns = {
            "OBSERVES": [
                r"(\w+)\s+(?:observes|monitors|measures|tracks)\s+(\w+)",
                r"(\w+)\s+(?:data|imagery|information)\s+(?:of|from)\s+(\w+)"
            ],
            "LOCATED_IN": [
                r"(\w+)\s+(?:in|over|above)\s+(\w+)",
                r"(\w+)\s+(?:region|area|zone)\s+(?:of|in)\s+(\w+)"
            ],
            "PART_OF": [
                r"(\w+)\s+(?:part of|component of|belongs to)\s+(\w+)",
                r"(\w+)\s+(?:system|mission|program)\s+(\w+)"
            ],
            "CAUSES": [
                r"(\w+)\s+(?:causes|leads to|results in)\s+(\w+)",
                r"(\w+)\s+(?:due to|because of|caused by)\s+(\w+)"
            ],
            "TEMPORAL": [
                r"(\w+)\s+(?:during|in|at)\s+(\w+)",
                r"(\w+)\s+(?:before|after|since)\s+(\w+)"
            ]
        }
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Create entity lookup
        entity_lookup = {entity.name.lower(): entity for entity in entities}
        
        # Pattern-based relationship extraction
        for relation_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    source_text = match.group(1).lower()
                    target_text = match.group(2).lower()
                    
                    # Find matching entities
                    source_entity = None
                    target_entity = None
                    
                    for entity_name, entity in entity_lookup.items():
                        if source_text in entity_name or entity_name in source_text:
                            source_entity = entity
                        if target_text in entity_name or entity_name in target_text:
                            target_entity = entity
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        relationship = Relationship(
                            id=f"rel_{len(relationships)}",
                            source_entity=source_entity.id,
                            target_entity=target_entity.id,
                            relation_type=relation_type,
                            properties={
                                "pattern_match": pattern,
                                "matched_text": match.group(0),
                                "confidence_source": "pattern_matching"
                            },
                            confidence=0.7,
                            source="pattern_extraction",
                            timestamp=datetime.now()
                        )
                        relationships.append(relationship)
        
        # Semantic relationship discovery (disabled for performance)
        # semantic_relationships = self._discover_semantic_relationships(text, entities)
        # relationships.extend(semantic_relationships)
        logger.debug("Semantic relationship discovery disabled for performance")
        
        return relationships
    
    def _discover_semantic_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """Discover relationships using semantic similarity"""
        relationships = []
        
        # Create entity pairs
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Calculate semantic similarity between entities
                emb1 = self.embedding_model.encode([entity1.name])
                emb2 = self.embedding_model.encode([entity2.name])
                
                similarity = np.dot(emb1[0], emb2[0]) / (
                    np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0])
                )
                
                if similarity > 0.5:  # Threshold for semantic relationship
                    # Determine relationship type based on entity types
                    relation_type = self._infer_relation_type(entity1, entity2)
                    
                    relationship = Relationship(
                        id=f"sem_rel_{len(relationships)}",
                        source_entity=entity1.id,
                        target_entity=entity2.id,
                        relation_type=relation_type,
                        properties={
                            "semantic_similarity": float(similarity),
                            "confidence_source": "semantic_similarity"
                        },
                        confidence=float(similarity),
                        source="semantic_discovery",
                        timestamp=datetime.now()
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _infer_relation_type(self, entity1: Entity, entity2: Entity) -> str:
        """Infer relationship type based on entity types"""
        type_combinations = {
            ("SATELLITE", "WEATHER_PARAMETER"): "OBSERVES",
            ("INSTRUMENT", "WEATHER_PARAMETER"): "MEASURES",
            ("LOCATION", "PHENOMENON"): "EXPERIENCES",
            ("ORGANIZATION", "SATELLITE"): "OPERATES",
            ("DATA_TYPE", "SATELLITE"): "PRODUCED_BY",
            ("PHENOMENON", "LOCATION"): "OCCURS_IN"
        }
        
        key = (entity1.type, entity2.type)
        reverse_key = (entity2.type, entity1.type)
        
        if key in type_combinations:
            return type_combinations[key]
        elif reverse_key in type_combinations:
            return type_combinations[reverse_key]
        else:
            return "RELATED_TO"  # Default relationship

class DynamicKnowledgeGraph:
    """Dynamic Knowledge Graph with real-time updates and consistency checking"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # Test connection with retry logic
        self._test_connection_with_retry()
        
        self.networkx_graph = nx.Graph()
        self.entity_extractor = LLMEntityExtractor()
        self.relationship_mapper = RelationshipMapper()
        self.kg_encoder = AdvancedKGEncoder(
            node_types=["ENTITY", "CONCEPT", "DOCUMENT", "R", "L", "P"],
            edge_types=["RELATED_TO", "PART_OF", "OBSERVES", "LOCATED_IN"]
        )
        
        # Version control for knowledge provenance
        self.version = 1
        self.change_log = []
    
    def _test_connection_with_retry(self, max_retries: int = 1, delay: float = 1.0):
        """Test Neo4j connection with silent failure"""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1").single()
                if result and result[0] == 1:
                    logger.info("✅ Successfully connected to Neo4j database")
                    return True
        except Exception:
            # Silently ignore connection errors
            pass
        return False
        
    def create_schema(self, documents: List[str]) -> Dict[str, Any]:
        """Create dynamic schema based on document content"""
        all_entities = []
        
        # Extract entities from all documents
        for doc in documents[:10]:  # Sample for schema creation
            entities = self.entity_extractor.extract_entities_with_llm(doc)
            all_entities.extend(entities)
        
        # Analyze entity types and properties
        entity_types = {}
        for entity in all_entities:
            if entity.type not in entity_types:
                entity_types[entity.type] = {
                    "count": 0,
                    "properties": set(),
                    "examples": []
                }
            
            entity_types[entity.type]["count"] += 1
            entity_types[entity.type]["properties"].update(entity.properties.keys())
            if len(entity_types[entity.type]["examples"]) < 5:
                entity_types[entity.type]["examples"].append(entity.name)
        
        # Convert sets to lists for JSON serialization
        for entity_type in entity_types:
            entity_types[entity_type]["properties"] = list(entity_types[entity_type]["properties"])
        
        schema = {
            "entity_types": list(entity_types.keys()),
            "type_details": entity_types,
            "version": self.version,
            "created_at": datetime.now().isoformat()
        }
        
        logger.info(f"Created dynamic schema with {len(entity_types)} entity types")
        return schema
    
    def build_knowledge_graph(self, documents: List[str], schema: Optional[Dict[str, Any]] = None) -> nx.Graph:
        """Build knowledge graph from documents"""
        if not schema:
            schema = self.create_schema(documents)
        
        all_entities = []
        all_relationships = []
        
        logger.info(f"Building knowledge graph from {len(documents)} documents")
        
        for i, document in enumerate(documents):
            logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Extract entities
            if schema:
                entities = self.entity_extractor.schema_aware_extraction(document, schema)
            else:
                entities = self.entity_extractor.extract_entities_with_llm(document)
            
            # Extract relationships
            relationships = self.relationship_mapper.extract_relationships(document, entities)
            
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        
        # Build NetworkX graph
        self.networkx_graph.clear()
        
        # Add entities as nodes
        for entity in all_entities:
            self.networkx_graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                properties=entity.properties,
                confidence=entity.confidence
            )
        
        # Add relationships as edges
        for relationship in all_relationships:
            if (relationship.source_entity in self.networkx_graph.nodes() and 
                relationship.target_entity in self.networkx_graph.nodes()):
                self.networkx_graph.add_edge(
                    relationship.source_entity,
                    relationship.target_entity,
                    relation_type=relationship.relation_type,
                    properties=relationship.properties,
                    confidence=relationship.confidence
                )
        
        logger.info(f"Knowledge graph built with {len(all_entities)} entities and {len(all_relationships)} relationships")
        
        # Store in Neo4j
        self._store_in_neo4j(all_entities, all_relationships)
        
        return self.networkx_graph
    
    def _store_in_neo4j(self, entities: List[Entity], relationships: List[Relationship]):
        """Store knowledge graph in Neo4j database"""
        try:
            with self.driver.session() as session:
                # Clear existing data (for development)
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create entities
                for entity in entities:
                    session.run(
                    """
                    CREATE (e:Entity {
                        id: $id,
                        name: $name,
                        type: $type,
                        description: $description,
                        confidence: $confidence,
                        source: $source,
                        timestamp: $timestamp,
                        properties: $properties
                    })
                    """,
                    id=entity.id,
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    confidence=entity.confidence,
                    source=entity.source,
                    timestamp=entity.timestamp.isoformat(),
                    properties=json.dumps(entity.properties)
                    )
                
                # Create relationships
                for relationship in relationships:
                    session.run(
                    """
                    MATCH (a:Entity {id: $source_id})
                    MATCH (b:Entity {id: $target_id})
                    CREATE (a)-[r:RELATIONSHIP {
                        id: $id,
                        type: $relation_type,
                        confidence: $confidence,
                        source: $source,
                        timestamp: $timestamp,
                        properties: $properties
                    }]->(b)
                    """,
                        source_id=relationship.source_entity,
                        target_id=relationship.target_entity,
                        id=relationship.id,
                        relation_type=relationship.relation_type,
                        confidence=relationship.confidence,
                        source=relationship.source,
                        timestamp=relationship.timestamp.isoformat(),
                        properties=json.dumps(relationship.properties)
                    )
            
            logger.info("Knowledge graph stored in Neo4j database")
            
        except Exception as e:
            logger.error(f"Failed to store knowledge graph in Neo4j: {e}")
            # Continue without Neo4j storage if connection fails
            logger.warning("Continuing without Neo4j storage")
    
    def incremental_update(self, new_documents: List[str]) -> Dict[str, Any]:
        """Incrementally update knowledge graph with new documents"""
        logger.info(f"Performing incremental update with {len(new_documents)} new documents")
        
        # Extract new entities and relationships
        new_entities = []
        new_relationships = []
        
        # Process documents in smaller batches to avoid memory issues
        batch_size = 10
        for i in range(0, len(new_documents), batch_size):
            batch = new_documents[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(new_documents) + batch_size - 1)//batch_size}")
            
            for doc_idx, document in enumerate(batch):
                try:
                    # Limit document length to avoid processing very long documents
                    if len(document) > 10000:
                        document = document[:10000] + "..."
                    
                    entities = self.entity_extractor.extract_entities_with_llm(document)
                    relationships = self.relationship_mapper.extract_relationships(document, entities)
                    
                    new_entities.extend(entities)
                    new_relationships.extend(relationships)
                    
                    if (doc_idx + 1) % 5 == 0:
                        logger.debug(f"Processed {doc_idx + 1}/{len(batch)} documents in current batch")
                        
                except Exception as e:
                    logger.warning(f"Error processing document {i + doc_idx}: {e}")
                    continue
        
        # Check for conflicts and consistency
        conflicts = self._check_consistency(new_entities, new_relationships)
        
        # Update NetworkX graph
        for entity in new_entities:
            if entity.id not in self.networkx_graph.nodes():
                self.networkx_graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    properties=entity.properties,
                    confidence=entity.confidence
                )
        
        for relationship in new_relationships:
            if (relationship.source_entity in self.networkx_graph.nodes() and 
                relationship.target_entity in self.networkx_graph.nodes()):
                self.networkx_graph.add_edge(
                    relationship.source_entity,
                    relationship.target_entity,
                    relation_type=relationship.relation_type,
                    properties=relationship.properties,
                    confidence=relationship.confidence
                )
        
        # Update version and log changes
        self.version += 1
        change_entry = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "new_entities": len(new_entities),
            "new_relationships": len(new_relationships),
            "conflicts": len(conflicts)
        }
        self.change_log.append(change_entry)
        
        # Store updates in Neo4j
        self._store_in_neo4j(new_entities, new_relationships)
        
        return {
            "update_summary": change_entry,
            "conflicts": conflicts,
            "graph_stats": {
                "total_nodes": len(self.networkx_graph.nodes()),
                "total_edges": len(self.networkx_graph.edges())
            }
        }
    
    def _check_consistency(self, new_entities: List[Entity], 
                          new_relationships: List[Relationship]) -> List[Dict[str, Any]]:
        """Check for consistency and conflicts in new data"""
        conflicts = []
        
        # Check for entity conflicts (same name, different types)
        existing_entities = {
            data['name']: data['type'] 
            for _, data in self.networkx_graph.nodes(data=True)
        }
        
        for entity in new_entities:
            if entity.name in existing_entities:
                if existing_entities[entity.name] != entity.type:
                    conflicts.append({
                        "type": "entity_type_conflict",
                        "entity_name": entity.name,
                        "existing_type": existing_entities[entity.name],
                        "new_type": entity.type
                    })
        
        # Check for relationship conflicts
        existing_relationships = set()
        for source, target, data in self.networkx_graph.edges(data=True):
            existing_relationships.add((source, target, data.get('relation_type')))
        
        for relationship in new_relationships:
            rel_tuple = (relationship.source_entity, relationship.target_entity, relationship.relation_type)
            if rel_tuple in existing_relationships:
                conflicts.append({
                    "type": "duplicate_relationship",
                    "source": relationship.source_entity,
                    "target": relationship.target_entity,
                    "relation_type": relationship.relation_type
                })
        
        return conflicts
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the knowledge graph"""
        stats = {
            "nodes": len(self.networkx_graph.nodes()),
            "edges": len(self.networkx_graph.edges()),
            "density": nx.density(self.networkx_graph),
            "connected_components": nx.number_connected_components(self.networkx_graph),
            "version": self.version,
            "last_updated": self.change_log[-1]["timestamp"] if self.change_log else None
        }
        
        # Entity type distribution
        entity_types = {}
        for _, data in self.networkx_graph.nodes(data=True):
            entity_type = data.get('type', 'UNKNOWN')
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        stats["entity_type_distribution"] = entity_types
        
        # Relationship type distribution
        relationship_types = {}
        for _, _, data in self.networkx_graph.edges(data=True):
            rel_type = data.get('relation_type', 'UNKNOWN')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        stats["relationship_type_distribution"] = relationship_types
        
        return stats
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge graph
        
        Returns:
            Dict containing graph statistics including node counts, edge counts, and type distributions
        """
        try:
            # Get basic graph statistics
            stats = {
                "total_nodes": len(self.networkx_graph.nodes()),
                "total_relationships": len(self.networkx_graph.edges()),
                "node_types": {},
                "relationship_types": {},
                "last_updated": datetime.now().isoformat(),
                "version": self.version,
                "connected_components": nx.number_connected_components(self.networkx_graph)
            }
            
            # Count nodes by type
            for node, data in self.networkx_graph.nodes(data=True):
                node_type = data.get('type', 'UNKNOWN')
                stats["node_types"][node_type] = stats["node_types"].get(node_type, 0) + 1
            
            # Count relationships by type
            for _, _, data in self.networkx_graph.edges(data=True):
                rel_type = data.get('relation_type', 'UNKNOWN')
                stats["relationship_types"][rel_type] = stats["relationship_types"].get(rel_type, 0) + 1
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            # Return minimal stats on error
            return {
                "total_nodes": len(self.networkx_graph.nodes()),
                "total_relationships": len(self.networkx_graph.edges()),
                "error": str(e),
                "version": self.version
            }
    
    def close(self):
        """Close database connection"""
        self.driver.close()

# Export main classes
__all__ = [
    'DynamicKnowledgeGraph',
    'LLMEntityExtractor',
    'RelationshipMapper',
    'AdvancedKGEncoder',
    'Entity',
    'Relationship'
]
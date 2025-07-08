"""
Advanced LLM Handler with Context-Aware Generation
Supports multiple LLM backends and advanced prompting techniques
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
import httpx
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from loguru import logger
from utils.config import get_settings

@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = None
    stream: bool = False

@dataclass
class ConversationContext:
    """Context for multi-turn conversations"""
    history: List[Dict[str, str]]
    user_profile: Dict[str, Any]
    session_id: str
    domain_context: str
    timestamp: datetime

class ContextAwareGenerator:
    """Context-aware response generator with advanced prompting"""
    
    def __init__(self, 
                 model_name: str = "llama2:latest",
                 ollama_base_url: str = "http://localhost:11434"):
        
        # Store Ollama-specific model name
        self.ollama_model_name = model_name
        self.ollama_base_url = ollama_base_url
        # Fix TensorFlow Lite delegate error by forcing CPU device and disabling parallelism
        import os
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        
        # Domain-specific prompts for MOSDAC
        self.domain_prompts = {
            "weather_data": """You are a meteorological data expert specializing in MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) systems. 
            Provide accurate, technical information about weather data, satellite observations, and atmospheric measurements. 
            Use scientific terminology appropriately and cite data sources when possible.""",
            
            "satellite_imagery": """You are a remote sensing specialist with expertise in satellite imagery analysis. 
            Help users understand satellite data products, image interpretation, and earth observation techniques. 
            Focus on ISRO satellites like INSAT, SCATSAT, OCEANSAT, and CARTOSAT series.""",
            
            "climate_analysis": """You are a climate scientist specializing in Indian monsoon patterns and climate change analysis. 
            Provide insights on climate trends, seasonal variations, and long-term climate patterns using MOSDAC data.""",
            
            "oceanographic_data": """You are an oceanographer expert in marine data analysis. 
            Help with sea surface temperature, ocean currents, salinity measurements, and marine weather patterns 
            around the Indian Ocean region.""",
            
            "technical_documentation": """You are a technical documentation specialist for MOSDAC systems. 
            Provide clear, step-by-step guidance on data access, processing workflows, and system usage.""",
            
            "general_inquiry": """You are a helpful MOSDAC AI assistant. Provide accurate information about 
            meteorological and oceanographic data, satellite systems, and earth observation capabilities."""
        }
        
        # Initialize conversation memory
        self.conversation_memory = {}
        
        logger.info(f"Context-aware generator initialized with Ollama model: {model_name}")
    
    async def generate_response(self, 
                              query: str,
                              retrieval_results: List[Any],
                              context: ConversationContext,
                              config: GenerationConfig = None) -> Dict[str, Any]:
        """Generate context-aware response"""
        
        if config is None:
            config = GenerationConfig()
        
        # Build context-aware prompt
        prompt = self._build_contextual_prompt(query, retrieval_results, context)
        
        # Generate response
        if config.stream:
            response_generator = self._generate_streaming_response(prompt, config)
            return {
                "response_stream": response_generator,
                "context": context,
                "prompt_tokens": len(prompt.split()),
                "generation_config": config
            }
        else:
            response_text = await self._generate_response(prompt, config)
            
            # Post-process response
            processed_response = self._post_process_response(response_text, query, context)
            
            return {
                "response": processed_response,
                "context": context,
                "prompt_tokens": len(prompt.split()),
                "response_tokens": len(processed_response.split()),
                "generation_config": config,
                "confidence_score": self._calculate_confidence(processed_response, retrieval_results)
            }
    
    def _build_contextual_prompt(self, 
                                query: str, 
                                retrieval_results: List[Any], 
                                context: ConversationContext) -> str:
        """Build context-aware prompt with retrieved information"""
        
        # Get domain-specific system prompt
        domain = context.domain_context
        system_prompt = self.domain_prompts.get(domain, self.domain_prompts["general_inquiry"])
        
        # Build conversation history
        history_text = ""
        if context.history:
            history_text = "\n\nConversation History:\n"
            for turn in context.history[-3:]:  # Last 3 turns
                history_text += f"Human: {turn.get('human', '')}\n"
                history_text += f"Assistant: {turn.get('assistant', '')}\n"
        
        # Build retrieved context
        context_text = ""
        if retrieval_results:
            context_text = "\n\nRelevant Information:\n"
            for i, result in enumerate(retrieval_results[:5]):  # Top 5 results
                if hasattr(result, 'content'):
                    content = result.content
                elif isinstance(result, tuple):
                    content = result[0].content if hasattr(result[0], 'content') else str(result[0])
                else:
                    content = str(result)
                
                context_text += f"[Source {i+1}]: {content[:500]}...\n"
        
        # Build user profile context
        profile_text = ""
        if context.user_profile:
            expertise_level = context.user_profile.get('expertise_level', 'general')
            interests = context.user_profile.get('interests', [])
            if interests:
                profile_text = f"\n\nUser Context: Expertise level: {expertise_level}, Interests: {', '.join(interests)}"
        
        # Combine all parts
        full_prompt = f"""{system_prompt}

{history_text}

{context_text}

{profile_text}

Current Question: {query}

Please provide a comprehensive, accurate response based on the available information. If you're uncertain about any details, please indicate this clearly. Focus on being helpful while maintaining scientific accuracy.

Response:"""
        
        return full_prompt
    
    async def _generate_response(self, prompt: str, config: GenerationConfig) -> str:
        """Generate response using configured LLM provider"""
        settings = get_settings()
        
        if settings.LLM_PROVIDER.lower() == "nebius":
            return await self._generate_nebius_response(prompt, config)
        else:
            return await self._generate_ollama_response(prompt, config)
    
    async def _generate_nebius_response(self, prompt: str, config: GenerationConfig) -> str:
        """Generate response using Nebius AI API"""
        settings = get_settings()
        
        if not settings.NEBUIS_API_KEY:
            logger.error("Nebius AI API key not configured")
            return "Error: Nebius AI API key not configured. Please set the NEBUIS_API_KEY in your environment variables."
        
        try:
            headers = {
                "Authorization": f"Bearer {settings.NEBUIS_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": settings.DEFAULT_LLM_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "stop": config.stop_sequences if config.stop_sequences else None
            }
            
            # Use longer timeout and better error handling for network issues
            async with httpx.AsyncClient(timeout=120.0) as client:
                try:
                    response = await client.post(
                        settings.NEBUIS_API_URL,
                        headers=headers,
                        json=payload
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if content:
                            logger.info("Nebius AI response generated successfully")
                            return content
                        else:
                            logger.error("Nebius AI returned empty content")
                            logger.info("Falling back to Ollama due to empty response")
                            return await self._generate_ollama_response(prompt, config)
                    else:
                        error_text = response.text
                        logger.error(f"Nebius AI API error: {response.status_code} - {error_text}")
                        
                        # Parse error for more specific information
                        try:
                            error_data = response.json()
                            error_message = error_data.get("error", {}).get("message", "Unknown error")
                            logger.error(f"Nebius AI error details: {error_message}")
                        except:
                            pass
                        
                        # Try fallback to Ollama if Nebius fails
                        logger.info("Falling back to Ollama due to Nebius API error")
                        return await self._generate_ollama_response(prompt, config)
                        
                except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
                    logger.error(f"Network error connecting to Nebius AI: {str(e)}")
                    logger.info("Falling back to Ollama due to network error")
                    return await self._generate_ollama_response(prompt, config)
                    
        except Exception as e:
            logger.error(f"Unexpected error with Nebius AI: {str(e)}")
            logger.info("Falling back to Ollama due to unexpected error")
            return await self._generate_ollama_response(prompt, config)
    
    async def _generate_ollama_response(self, prompt: str, config: GenerationConfig) -> str:
        """Generate response using Ollama API"""
        settings = get_settings()
        
        try:
            # Use the specific Ollama model name from settings, fallback to instance variable
            ollama_model = getattr(settings, 'OLLAMA_MODEL', self.ollama_model_name)
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": ollama_model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": config.max_tokens,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "repeat_penalty": config.repetition_penalty,
                    },
                    "stream": False
                }
                
                if config.stop_sequences:
                    payload["options"]["stop"] = config.stop_sequences
                
                logger.info(f"Attempting Ollama request with model: {ollama_model}")
                
                response = await client.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    if response_text:
                        logger.info("Ollama response generated successfully")
                        return response_text
                    else:
                        logger.error("Ollama returned empty response")
                        return "I apologize, but I received an empty response. This might be due to the model not being available locally."
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return f"I apologize, but the local AI service is not available (Status: {response.status_code}). Please ensure Ollama is running with the '{ollama_model}' model installed."
                    
        except httpx.ConnectError as e:
            logger.error(f"Cannot connect to Ollama: {str(e)}")
            return "I apologize, but I cannot connect to the local AI service. Please ensure Ollama is running on http://localhost:11434"
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request: {str(e)}"
    
    async def _generate_streaming_response(self, 
                                         prompt: str, 
                                         config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "options": {
                        "num_predict": config.max_tokens,
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "repeat_penalty": config.repetition_penalty,
                    },
                    "stream": True
                }
                
                async with client.stream(
                    "POST",
                    f"{self.ollama_base_url}/api/generate",
                    json=payload
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "response" in data:
                                    yield data["response"]
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield "Error occurred during response generation."
    
    def _post_process_response(self, 
                              response: str, 
                              query: str, 
                              context: ConversationContext) -> str:
        """Post-process the generated response"""
        
        # Remove any incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'
        
        # Add domain-specific formatting
        if context.domain_context == "technical_documentation":
            response = self._format_technical_response(response)
        elif context.domain_context in ["weather_data", "oceanographic_data"]:
            response = self._format_scientific_response(response)
        
        # Add helpful suggestions if appropriate
        if "?" in query and len(response) < 100:
            response += "\n\nWould you like me to provide more specific information about any particular aspect?"
        
        return response.strip()
    
    def _format_technical_response(self, response: str) -> str:
        """Format technical documentation responses"""
        
        # Add step numbering if it looks like a procedure
        if "first" in response.lower() or "then" in response.lower():
            lines = response.split('\n')
            formatted_lines = []
            step_count = 1
            
            for line in lines:
                if any(word in line.lower() for word in ['first', 'then', 'next', 'finally']):
                    formatted_lines.append(f"{step_count}. {line}")
                    step_count += 1
                else:
                    formatted_lines.append(line)
            
            response = '\n'.join(formatted_lines)
        
        return response
    
    def _format_scientific_response(self, response: str) -> str:
        """Format scientific data responses"""
        
        # Highlight important measurements and units
        response = re.sub(r'(\d+\.?\d*)\s*(°C|°F|K|hPa|mb|m/s|km/h|mm|cm|m)', 
                         r'**\1 \2**', response)
        
        # Highlight satellite names
        satellites = ['INSAT', 'SCATSAT', 'OCEANSAT', 'CARTOSAT', 'RESOURCESAT']
        for satellite in satellites:
            response = re.sub(f'\\b{satellite}\\b', f'**{satellite}**', response, flags=re.IGNORECASE)
        
        return response
    
    def _calculate_confidence(self, response: str, retrieval_results: List[Any]) -> float:
        """Calculate confidence score for the response"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence if response is well-structured
        if len(response.split('.')) > 2:
            confidence += 0.1
        
        # Increase confidence if response contains specific data
        if re.search(r'\d+\.?\d*\s*(°C|°F|K|hPa|mb|m/s|km/h|mm|cm|m)', response):
            confidence += 0.1
        
        # Increase confidence based on retrieval results quality
        if retrieval_results:
            avg_retrieval_score = sum(
                result[1] if isinstance(result, tuple) else 0.5 
                for result in retrieval_results
            ) / len(retrieval_results)
            confidence += avg_retrieval_score * 0.3
        
        # Decrease confidence for uncertain language
        uncertain_phrases = ['might be', 'could be', 'possibly', 'perhaps', 'not sure']
        for phrase in uncertain_phrases:
            if phrase in response.lower():
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def update_conversation_memory(self, 
                                  session_id: str, 
                                  human_message: str, 
                                  assistant_response: str):
        """Update conversation memory"""
        
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        
        self.conversation_memory[session_id].append({
            "human": human_message,
            "assistant": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 10 turns
        if len(self.conversation_memory[session_id]) > 10:
            self.conversation_memory[session_id] = self.conversation_memory[session_id][-10:]
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.conversation_memory.get(session_id, [])
    
    def clear_conversation_memory(self, session_id: str):
        """Clear conversation memory for a session"""
        if session_id in self.conversation_memory:
            del self.conversation_memory[session_id]

class MultiAgentLLMOrchestrator:
    """Multi-agent LLM system for specialized tasks"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.agents = {
            "retrieval_agent": ContextAwareGenerator("llama2:latest", base_url),
            "reasoning_agent": ContextAwareGenerator("llama2:latest", base_url),
            "synthesis_agent": ContextAwareGenerator("llama2:latest", base_url),
            "verification_agent": ContextAwareGenerator("llama2:latest", base_url)
        }
        
        # Specialized prompts for each agent
        self.agent_prompts = {
            "retrieval_agent": "You are a specialized retrieval agent. Your task is to identify the most relevant information for answering the user's query.",
            "reasoning_agent": "You are a reasoning agent. Your task is to perform logical reasoning and multi-step analysis based on the provided information.",
            "synthesis_agent": "You are a synthesis agent. Your task is to combine information from multiple sources into a coherent, comprehensive response.",
            "verification_agent": "You are a verification agent. Your task is to fact-check information and ensure accuracy and consistency."
        }
    
    async def process_with_agents(self, 
                                query: str, 
                                retrieval_results: List[Any],
                                context: ConversationContext) -> Dict[str, Any]:
        """Process query using multiple specialized agents"""
        
        # Step 1: Retrieval agent analyzes relevance
        retrieval_analysis = await self._run_agent(
            "retrieval_agent", 
            f"Analyze the relevance of the following information for the query: '{query}'\n\nInformation: {str(retrieval_results[:3])}",
            context
        )
        
        # Step 2: Reasoning agent performs analysis
        reasoning_result = await self._run_agent(
            "reasoning_agent",
            f"Based on the query '{query}' and the relevant information, perform detailed reasoning and analysis.\n\nRelevant info: {retrieval_analysis}",
            context
        )
        
        # Step 3: Synthesis agent creates final response
        synthesis_result = await self._run_agent(
            "synthesis_agent",
            f"Create a comprehensive response to: '{query}'\n\nBased on reasoning: {reasoning_result}",
            context
        )
        
        # Step 4: Verification agent checks accuracy
        verification_result = await self._run_agent(
            "verification_agent",
            f"Verify the accuracy and consistency of this response: {synthesis_result}\n\nOriginal query: {query}",
            context
        )
        
        return {
            "final_response": synthesis_result,
            "verification": verification_result,
            "reasoning_trace": {
                "retrieval_analysis": retrieval_analysis,
                "reasoning_result": reasoning_result,
                "synthesis_result": synthesis_result,
                "verification_result": verification_result
            },
            "confidence_score": self._calculate_multi_agent_confidence(verification_result)
        }
    
    async def _run_agent(self, 
                        agent_name: str, 
                        prompt: str, 
                        context: ConversationContext) -> str:
        """Run a specific agent"""
        
        agent = self.agents[agent_name]
        agent_prompt = self.agent_prompts[agent_name]
        
        full_prompt = f"{agent_prompt}\n\n{prompt}"
        
        # Create a simple context for the agent
        agent_context = ConversationContext(
            history=[],
            user_profile=context.user_profile,
            session_id=f"{context.session_id}_{agent_name}",
            domain_context=context.domain_context,
            timestamp=datetime.now()
        )
        
        config = GenerationConfig(max_tokens=512, temperature=0.3)
        result = await agent.generate_response("", [], agent_context, config)
        
        return result.get("response", "")
    
    def _calculate_multi_agent_confidence(self, verification_result: str) -> float:
        """Calculate confidence based on verification agent output"""
        
        confidence = 0.7  # Base confidence for multi-agent system
        
        # Increase confidence if verification is positive
        positive_indicators = ["accurate", "correct", "consistent", "verified", "reliable"]
        negative_indicators = ["inaccurate", "incorrect", "inconsistent", "uncertain", "questionable"]
        
        for indicator in positive_indicators:
            if indicator in verification_result.lower():
                confidence += 0.05
        
        for indicator in negative_indicators:
            if indicator in verification_result.lower():
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))

# Export main classes
__all__ = [
    'ContextAwareGenerator',
    'MultiAgentLLMOrchestrator',
    'GenerationConfig',
    'ConversationContext'
]
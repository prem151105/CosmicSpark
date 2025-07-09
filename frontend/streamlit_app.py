"""
Advanced MOSDAC AI Help Bot - Streamlit Frontend
Interactive interface with knowledge graph visualization and multimodal capabilities
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from streamlit_chat import message
from streamlit_option_menu import option_menu
import base64
from io import BytesIO
import uuid

# Page configuration
st.set_page_config(
    page_title="MOSDAC AI Help Bot",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .knowledge-graph {
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
SESSION_ID = str(uuid.uuid4())

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = SESSION_ID
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {
        'expertise_level': 'general',
        'interests': []
    }

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        # Debug: Show the URL being called
        if st.session_state.get('debug_mode', False):
            st.info(f"Making {method} request to: {url}")
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code == 200:
            result = response.json()
            if st.session_state.get('debug_mode', False):
                st.success(f"API request successful: {response.status_code}")
            return result
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return {}
    except requests.exceptions.ConnectionError as e:
        st.error(f"Cannot connect to API at {API_BASE_URL}. Backend may be down. Error: {str(e)}")
        return {}
    except requests.exceptions.Timeout:
        st.error("Request timed out. Backend may be overloaded.")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {}

def get_system_status() -> Dict:
    """Get system health status"""
    return make_api_request("/health")

def send_query(query: str, stream: bool = False) -> Dict:
    """Send query to the API"""
    data = {
        "query": query,
        "session_id": st.session_state.session_id,
        "user_profile": st.session_state.user_profile,
        "stream": stream,
        "max_tokens": 1024,
        "temperature": 0.7
    }
    return make_api_request("/query", "POST", data)

def upload_document(content: str, title: str, metadata: Dict = None) -> Dict:
    """Upload document to the system"""
    data = {
        "content": content,
        "title": title,
        "metadata": metadata or {},
        "modality": "text"
    }
    return make_api_request("/documents/upload", "POST", data)

def create_knowledge_graph_viz(kg_stats: Dict) -> go.Figure:
    """Create knowledge graph visualization"""
    if not kg_stats:
        return go.Figure()
    
    # Create sample network for visualization
    G = nx.Graph()
    
    # Add sample nodes and edges based on stats
    entities = ["MOSDAC", "Satellite Data", "Weather", "Ocean", "Climate", "INSAT", "SCATSAT"]
    for entity in entities:
        G.add_node(entity)
    
    # Add sample edges
    edges = [
        ("MOSDAC", "Satellite Data"),
        ("MOSDAC", "Weather"),
        ("MOSDAC", "Ocean"),
        ("Satellite Data", "INSAT"),
        ("Satellite Data", "SCATSAT"),
        ("Weather", "Climate"),
        ("Ocean", "Climate")
    ]
    G.add_edges_from(edges)
    
    # Get positions
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node traces
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_info.append(f"Entity: {node}<br>Connections: {len(list(G.neighbors(node)))}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            size=30,
            color='lightblue',
            line=dict(width=2, color='darkblue')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=dict(
                           text='MOSDAC Knowledge Graph',
                           font=dict(size=16)
                       ),
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[dict(
                           text="Interactive Knowledge Graph Visualization",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="#888", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">🛰️ MOSDAC AI Help Bot</h1>', unsafe_allow_html=True)
    st.markdown("*Advanced AI Assistant for Meteorological and Oceanographic Data*")
    
    # Removed backend status indicator as requested
    
    # Sidebar navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/white?text=MOSDAC", width=200)
        
        selected = option_menu(
            "Navigation",
            ["Chat Interface", "Knowledge Graph", "Document Upload", "System Status", "Settings"],
            icons=['chat-dots', 'diagram-3', 'cloud-upload', 'gear', 'sliders'],
            menu_icon="cast",
            default_index=0,
        )
        
        # User Profile Settings
        st.subheader("User Profile")
        expertise_level = st.selectbox(
            "Expertise Level",
            ["general", "technical", "expert"],
            index=0
        )
        
        interests = st.multiselect(
            "Areas of Interest",
            ["Weather Data", "Satellite Imagery", "Ocean Data", "Climate Analysis", "Technical Documentation"],
            default=[]
        )
        
        st.session_state.user_profile = {
            'expertise_level': expertise_level,
            'interests': interests
        }
        
        # Debug mode toggle
        st.session_state.debug_mode = st.checkbox("Debug Mode", value=False)
        
        # Connection test
        if st.button("Test Backend Connection"):
            with st.spinner("Testing connection..."):
                status = get_system_status()
                if status:
                    st.success(f"✅ Backend connected! Status: {status.get('status', 'unknown')}")
                else:
                    st.error("❌ Cannot connect to backend")
        
        # Show current API URL
        if st.session_state.get('debug_mode', False):
            st.info(f"API URL: {API_BASE_URL}")
    
    # Main content based on selection
    if selected == "Chat Interface":
        chat_interface()
    elif selected == "Knowledge Graph":
        knowledge_graph_page()
    elif selected == "Document Upload":
        document_upload_page()
    elif selected == "System Status":
        system_status_page()
    elif selected == "Settings":
        settings_page()

def chat_interface():
    """Chat interface page"""
    st.header("💬 Chat with MOSDAC AI")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                message(msg["content"], is_user=True, key=f"user_{i}")
            else:
                message(msg["content"], key=f"bot_{i}")
                
                # Show additional info if available
                if "metadata" in msg:
                    with st.expander("Response Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Confidence", f"{msg['metadata'].get('confidence_score', 0):.2f}")
                        with col2:
                            st.metric("Intent", msg['metadata'].get('intent', 'Unknown'))
                        with col3:
                            st.metric("Sources", len(msg['metadata'].get('retrieval_results', [])))
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Ask me anything about MOSDAC data, satellites, weather, or oceanography:",
                placeholder="e.g., What is the latest INSAT-3D data available?",
                label_visibility="collapsed"
            )
        
        with col2:
            submit_button = st.form_submit_button("Send", use_container_width=True)
    
    if submit_button and user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            # Send query to API
            response = send_query(user_input)
            
            if response:
                # Add bot response
                bot_message = {
                    "role": "assistant",
                    "content": response.get("response", "I apologize, but I couldn't generate a response."),
                    "metadata": {
                        "confidence_score": response.get("confidence_score", 0),
                        "intent": response.get("intent", "unknown"),
                        "entities": response.get("entities", []),
                        "retrieval_results": response.get("retrieval_results", []),
                        "reasoning_path": response.get("reasoning_path", [])
                    }
                }
                st.session_state.messages.append(bot_message)
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I'm sorry, I'm having trouble connecting to the backend. Please try again later."
                })
        
        st.rerun()
    
    # Quick action buttons
    st.subheader("Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🛰️ Satellite Data", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Tell me about available satellite data"})
            st.rerun()
    
    with col2:
        if st.button("🌊 Ocean Data", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "What ocean data is available?"})
            st.rerun()
    
    with col3:
        if st.button("🌤️ Weather Info", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "Show me weather information"})
            st.rerun()
    
    with col4:
        if st.button("📚 Documentation", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": "How do I access MOSDAC data?"})
            st.rerun()

def knowledge_graph_page():
    """Knowledge graph visualization page"""
    st.header("🕸️ Knowledge Graph Visualization")
    
    # Get KG statistics
    kg_stats = make_api_request("/knowledge-graph/stats")
    
    # Debug: Print the raw response
    if st.session_state.get('debug_mode', False):
        st.json(kg_stats)
    
    if kg_stats:
        # Extract statistics with better error handling
        total_nodes = kg_stats.get("total_nodes") or kg_stats.get("nodes", 0)
        total_relationships = kg_stats.get("total_relationships") or kg_stats.get("relationships") or kg_stats.get("edges", 0)
        node_types = kg_stats.get("node_types") or len(kg_stats.get("node_types_dict", {})) or 0
        relationship_types = kg_stats.get("relationship_types") or len(kg_stats.get("relationship_types_dict", {})) or 0
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f'<div class="metric-card"><h3>{total_nodes}</h3><p>Total Entities</p></div>',
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f'<div class="metric-card"><h3>{total_relationships}</h3><p>Relationships</p></div>',
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f'<div class="metric-card"><h3>{node_types}</h3><p>Entity Types</p></div>',
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f'<div class="metric-card"><h3>{relationship_types}</h3><p>Relation Types</p></div>',
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Knowledge graph visualization
        st.subheader("Interactive Knowledge Graph")
        
        with st.container():
            fig = create_knowledge_graph_viz(kg_stats)
            st.plotly_chart(fig, use_container_width=True)
        
        # Entity search
        st.subheader("Entity Explorer")
        search_term = st.text_input("Search for entities:", placeholder="e.g., INSAT, weather, ocean")
        
        if search_term:
            st.info(f"Searching for entities related to: {search_term}")
            # In a real implementation, this would query the knowledge graph
            st.write("Search functionality would be implemented here with actual KG queries.")
    
    else:
        st.warning("Knowledge graph data not available. Please check if the backend is running.")

def document_upload_page():
    """Document upload page"""
    st.header("📄 Document Upload & Processing")
    
    tab1, tab2 = st.tabs(["Text Upload", "File Upload"])
    
    with tab1:
        st.subheader("Upload Text Content")
        
        with st.form("text_upload_form"):
            title = st.text_input("Document Title", placeholder="Enter document title")
            content = st.text_area("Document Content", height=300, placeholder="Paste your document content here...")
            
            # Metadata
            st.subheader("Metadata (Optional)")
            col1, col2 = st.columns(2)
            
            with col1:
                author = st.text_input("Author")
                category = st.selectbox("Category", ["Weather", "Ocean", "Satellite", "Climate", "Technical", "Other"])
            
            with col2:
                source = st.text_input("Source")
                date = st.date_input("Date")
            
            submit_text = st.form_submit_button("Upload Document")
            
            if submit_text and title and content:
                metadata = {
                    "author": author,
                    "category": category,
                    "source": source,
                    "date": str(date) if date else None
                }
                
                with st.spinner("Uploading document..."):
                    result = upload_document(content, title, metadata)
                    
                    if result:
                        st.success(f"Document uploaded successfully! ID: {result.get('document_id', 'Unknown')}")
                    else:
                        st.error("Failed to upload document.")
    
    with tab2:
        st.subheader("Upload File")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['txt', 'pdf', 'docx', 'md'],
            help="Supported formats: TXT, PDF, DOCX, MD"
        )
        
        if uploaded_file is not None:
            # Display file details
            st.write("**File Details:**")
            st.write(f"- Name: {uploaded_file.name}")
            st.write(f"- Size: {uploaded_file.size} bytes")
            st.write(f"- Type: {uploaded_file.type}")
            
            if st.button("Process File"):
                with st.spinner("Processing file..."):
                    # In a real implementation, this would send the file to the API
                    st.success("File processing would be implemented here with actual API call.")

def system_status_page():
    """System status and monitoring page"""
    st.header("⚙️ System Status & Monitoring")
    
    # Get system status
    status = get_system_status()
    
    if status:
        # Overall status
        overall_status = status.get("status", "unknown")
        if overall_status == "healthy":
            st.success(f"🟢 System Status: {overall_status.upper()}")
        else:
            st.warning(f"🟡 System Status: {overall_status.upper()}")
        
        # Component status
        st.subheader("Component Status")
        components = status.get("components", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            for component, is_healthy in list(components.items())[:3]:
                status_icon = "🟢" if is_healthy else "🔴"
                st.write(f"{status_icon} {component.replace('_', ' ').title()}: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        with col2:
            for component, is_healthy in list(components.items())[3:]:
                status_icon = "🟢" if is_healthy else "🔴"
                st.write(f"{status_icon} {component.replace('_', ' ').title()}: {'Healthy' if is_healthy else 'Unhealthy'}")
        
        # Statistics
        st.subheader("System Statistics")
        statistics = status.get("statistics", {})
        
        if statistics:
            # Vector store stats
            if "vector_store" in statistics:
                vs_stats = statistics["vector_store"]
                st.write("**Vector Store:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", vs_stats.get("total_documents", 0))
                with col2:
                    st.metric("Collections", vs_stats.get("collections", 0))
                with col3:
                    st.metric("Size (MB)", vs_stats.get("size_mb", 0))
            
            # Knowledge graph stats
            if "knowledge_graph" in statistics:
                kg_stats = statistics["knowledge_graph"]
                st.write("**Knowledge Graph:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Nodes", kg_stats.get("total_nodes", 0))
                with col2:
                    st.metric("Relationships", kg_stats.get("total_relationships", 0))
                with col3:
                    # Handle case where node_types might be a dictionary
                    node_types = kg_stats.get("node_types", {})
                    node_types_count = len(node_types) if isinstance(node_types, dict) else node_types
                    st.metric("Node Types", node_types_count)
        
        # Refresh button
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    else:
        st.error("Unable to retrieve system status. Please check if the backend is running.")

def settings_page():
    """Settings and configuration page"""
    st.header("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("API Configuration")
    
    with st.form("api_config"):
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        timeout = st.number_input("Request Timeout (seconds)", min_value=1, max_value=300, value=30)
        
        st.form_submit_button("Update API Settings")
    
    # Chat Settings
    st.subheader("Chat Settings")
    
    with st.form("chat_config"):
        max_tokens = st.slider("Max Response Tokens", 100, 2000, 1024)
        temperature = st.slider("Response Creativity", 0.0, 1.0, 0.7, 0.1)
        
        st.form_submit_button("Update Chat Settings")
    
    # Session Management
    st.subheader("Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.success("Chat history cleared!")
    
    with col2:
        if st.button("Reset Session", type="secondary"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("Session reset!")
    
    # Export/Import
    st.subheader("Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export Chat History"):
            if st.session_state.messages:
                chat_data = json.dumps(st.session_state.messages, indent=2)
                st.download_button(
                    label="Download Chat History",
                    data=chat_data,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No chat history to export.")
    
    with col2:
        uploaded_history = st.file_uploader("Import Chat History", type=['json'])
        if uploaded_history:
            try:
                imported_data = json.load(uploaded_history)
                st.session_state.messages = imported_data
                st.success("Chat history imported successfully!")
            except Exception as e:
                st.error(f"Error importing chat history: {e}")

if __name__ == "__main__":
    main()
# CosmicSpark Visual Flow

## User Interaction Flow

```mermaid
flowchart TD
    A[User] -->|Input Query| B[Streamlit Frontend]
    B -->|API Request| C[FastAPI Backend]
    C --> D[Query Processing]
    D --> E[Vector Store Search]
    D --> F[Knowledge Graph Query]
    E --> G[Retrieve Relevant Documents]
    F --> H[Retrieve Related Entities]
    G --> I[Combine Results]
    H --> I
    I --> J[Generate Response]
    J --> K[Return Response]
    K --> L[Display Results]
    L --> M[Visualize Knowledge Graph]
```

## Data Processing Pipeline

```mermaid
flowchart LR
    A[Input Documents] --> B[Document Preprocessing]
    B --> C[Text Extraction]
    C --> D[Entity Recognition]
    D --> E[Relationship Extraction]
    E --> F[Knowledge Graph Update]
    C --> G[Text Chunking]
    G --> H[Embedding Generation]
    H --> I[Vector Store Update]
```

## System Architecture

```mermaid
graph TD
    subgraph Frontend
        A[Streamlit UI] --> B[API Client]
    end
    
    subgraph Backend
        C[FastAPI Server]
        D[Vector Store]
        E[Knowledge Graph]
        F[LLM Integration]
    end
    
    B --> C
    C --> D
    C --> E
    C --> F
    
    style A fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style D fill:#bfb,stroke:#333
    style E fill:#fbb,stroke:#333
    style F fill:#ffb,stroke:#333
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Process Request] --> B{Valid Request?}
    B -->|Yes| C[Process]
    B -->|No| D[Return 400 Error]
    C --> E{Process Success?}
    E -->|Yes| F[Return 200 Success]
    E -->|No| G[Handle Error]
    G --> H{Recoverable?}
    H -->|Yes| C
    H -->|No| I[Return 500 Error]
```

## Knowledge Graph Visualization

```mermaid
graph LR
    A[Document] -->|Contains| B[Entities]
    B -->|Related to| C[Other Entities]
    B -->|Has Type| D[Entity Types]
    C -->|Connected via| E[Relationships]
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#fbb,stroke:#333
    style E fill:#ffb,stroke:#333
```

## Search Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    participant V as Vector Store
    participant K as Knowledge Graph
    
    U->>F: Enter Search Query
    F->>B: POST /api/search
    B->>V: Query Embeddings
    B->>K: Query Graph
    V-->>B: Relevant Documents
    K-->>B: Related Entities
    B->>B: Combine Results
    B-->>F: Return Response
    F-->>U: Display Results
```

## Document Ingestion Flow

```mermaid
flowchart TD
    A[Upload Document] --> B[Extract Text]
    B --> C[Process Text]
    C --> D[Extract Entities]
    D --> E[Update Knowledge Graph]
    C --> F[Generate Embeddings]
    F --> G[Update Vector Store]
    E --> H[Success]
    G --> H
```

## Authentication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend
    
    U->>F: Enter Credentials
    F->>B: POST /auth/login
    B-->>F: JWT Token
    F->>F: Store Token
    F->>B: API Requests with Token
    B->>B: Verify Token
    B-->>F: Response Data
```

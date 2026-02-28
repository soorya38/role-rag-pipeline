# Role RAG Service — Design Document

## Overview

The Role RAG Service is a **multi-user Retrieval-Augmented Generation** pipeline.
Each user has an **isolated FAISS vector index**. Ingestion and retrieval are always
scoped by `user_id` — no cross-user data leakage is possible.

---

## System Architecture

```mermaid
graph TB
    subgraph "Backend — FastAPI Service"
        API["API Layer<br/>api/routes.py"]

        subgraph "Retrieval Pipeline"
            RP["retrieval/pipeline.py"]
            LLM["Groq API<br/>llama-3.3-70b-versatile"]
        end

        subgraph "Ingestion Pipeline"
            IP["ingestion/pipeline.py"]
        end

        VS["vectorstore/faiss_store.py"]

        subgraph "Core"
            CFG["core/config.py"]
            LOG["core/logging.py"]
        end

        SCH["schemas/models.py"]
    end

    subgraph "Per-User FAISS Indexes"
        U1["vector_store/user1/"]
        U2["vector_store/user2/"]
        UN["vector_store/.../"]
    end

    API -->|"POST /api/ingest (user_id, file)"| IP
    API -->|"POST /api/match (user_id, query)"| RP

    IP --> VS
    RP --> VS

    VS -->|"save to user dir"| U1
    VS -->|"save to user dir"| U2
    VS -->|"save to user dir"| UN
    VS -->|"load user dir"| U1
    VS -->|"load user dir"| U2

    RP --> LLM
    LLM -->|LLMAnalysis| RP

    CFG -.->|settings| RP
    CFG -.->|settings| IP
    CFG -.->|settings| VS
    LOG -.->|logger| API
    LOG -.->|logger| RP
    LOG -.->|logger| IP
    SCH -.->|models| RP
    SCH -.->|models| API
```

---

## Multi-User Isolation

```mermaid
graph LR
    subgraph "Ingestion"
        A1["POST /api/ingest<br/>user_id=user1, file=roles1.txt"]
        A2["POST /api/ingest<br/>user_id=user2, file=roles2.txt"]
    end

    subgraph "Per-User FAISS"
        F1["vector_store/user1/"]
        F2["vector_store/user2/"]
    end

    subgraph "Matching"
        B1["POST /api/match<br/>user_id=user1, query=..."]
        B2["POST /api/match<br/>user_id=user2, query=..."]
    end

    A1 -->|"indexes roles"| F1
    A2 -->|"indexes roles"| F2
    B1 -->|"searches ONLY"| F1
    B2 -->|"searches ONLY"| F2
```

---

## Ingestion Pipeline

```mermaid
flowchart LR
    A["📤 Uploaded roles file<br/>(multipart/form-data)"] --> B["read bytes<br/>→ UTF-8 string"]
    B --> C["parse_roles()\nRegex split on 'role:'"]
    C --> D["build_documents()\nOne Document per section"]
    D --> E["chunk_documents()\nchunk=450 overlap=90"]
    E --> F["generate_embeddings()\nHuggingFace MiniLM-L6-v2"]
    F --> G["FAISS.from_documents()"]
    G --> H["💾 vector_store/<user_id>/"]
```

| Step | Function | Output |
|------|----------|--------|
| 1 | API reads `UploadFile` | Raw UTF-8 content |
| 2 | `parse_roles()` | `List[Dict]` — one dict per role |
| 3 | `build_documents()` | `List[Document]` — one per section |
| 4 | `chunk_documents()` | Smaller `List[Document]` chunks |
| 5 | `save_vector_store(user_id)` | FAISS index at `vector_store/<user_id>/` |

---

## Retrieval Pipeline

```mermaid
flowchart TD
    Q["📝 Job Description Query<br/>+ user_id"] --> LOAD["load_vector_store(user_id)<br/>reads vector_store/user_id/"]
    LOAD --> SIM["similarity_search_with_score()<br/>top_k results"]
    SIM --> AGG["aggregate_results()<br/>group by role, best L2 score"]
    AGG --> R["List[RetrievalResult]"]

    R --> LLM1["llm_summarise_and_rank()"]
    R --> LLM2["llm_fit_analysis()"]
    R --> LLM3["llm_cover_letter()"]

    LLM1 --> AN["LLMAnalysis"]
    LLM2 --> AN
    LLM3 --> AN

    R --> RESP["MatchResponse (user_id + results + analysis)"]
    AN --> RESP
```

---

## API Request-Response Flow

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI as FastAPI (routes.py)
    participant Ingestion as ingestion/pipeline.py
    participant Retrieval as retrieval/pipeline.py
    participant FAISS as FAISS (vector_store/<user_id>/)
    participant Groq as Groq API

    Note over Client,Groq: Step 1 — Ingest roles for a user
    Client->>FastAPI: POST /api/ingest { user_id=user1, file=roles.txt }
    FastAPI->>Ingestion: run_ingestion_pipeline(content, user_id)
    Ingestion->>FAISS: save_local(vector_store/user1/)
    FAISS-->>FastAPI: IngestResponse { chunks_indexed }
    FastAPI-->>Client: 200 OK

    Note over Client,Groq: Step 2 — Match a job description (same user)
    Client->>FastAPI: POST /api/match { user_id=user1, query, top_k }
    FastAPI->>Retrieval: retrieve(query, user_id=user1, top_k)
    Retrieval->>FAISS: load_local(vector_store/user1/)
    FAISS-->>Retrieval: vector store
    Retrieval->>FAISS: similarity_search_with_score(query)
    FAISS-->>Retrieval: List[(Document, score)]
    Retrieval->>Retrieval: aggregate_results()
    Retrieval-->>FastAPI: List[RetrievalResult]

    FastAPI->>Retrieval: run_llm_analysis(query, results)
    Retrieval->>Groq: summarise_and_rank / fit_analysis / cover_letter
    Groq-->>Retrieval: LLMAnalysis
    Retrieval-->>FastAPI: LLMAnalysis
    FastAPI-->>Client: MatchResponse { user_id, results, analysis }
```

---

## Module Dependency Graph

```mermaid
graph LR
    main["app/main.py"] --> routes["api/routes.py"]
    routes --> retrieval["retrieval/pipeline.py"]
    routes --> ingestion["ingestion/pipeline.py"]
    routes --> schemas["schemas/models.py"]
    retrieval --> vectorstore["vectorstore/faiss_store.py"]
    retrieval --> schemas
    retrieval --> cfg["core/config.py"]
    retrieval --> log["core/logging.py"]
    ingestion --> vectorstore
    ingestion --> cfg
    ingestion --> log
    vectorstore --> cfg
    vectorstore --> log
    routes --> log
    main --> cfg
```

---

## Data Models

```mermaid
classDiagram
    class MatchRequest {
        +str user_id
        +str query
        +int top_k = 10
    }

    class IngestResponse {
        +str user_id
        +int chunks_indexed
        +str message
    }

    class RetrievalResult {
        +str role
        +str tools
        +str projects
        +float score
        +as_dict() dict
        +as_text_block() str
    }

    class LLMAnalysis {
        +str summary_and_ranking
        +str fit_analysis
        +str cover_letter
    }

    class MatchResponse {
        +str user_id
        +List~RetrievalResult~ results
        +LLMAnalysis analysis
    }

    MatchRequest --> MatchResponse : triggers
    MatchResponse o-- RetrievalResult
    MatchResponse o-- LLMAnalysis
```

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | _(required)_ | Groq API authentication key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model identifier |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `VECTOR_STORE_PATH` | `vector_store` | Root directory for all per-user FAISS indexes |
| `CHUNK_SIZE` | `450` | Token chunk size for the text splitter |
| `CHUNK_OVERLAP` | `90` | Overlap between adjacent chunks |
| `HOST` | `0.0.0.0` | FastAPI server bind host |
| `PORT` | `8000` | FastAPI server bind port |
| `RELOAD` | `true` | Enable uvicorn hot-reload |

---

## Technology Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Vector Store | FAISS (CPU), one index per user |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |
| Containerisation | Docker |

# Role RAG Service

A FastAPI-based **Retrieval-Augmented Generation (RAG)** service that matches candidate roles to job descriptions using:
- **FAISS** — local vector store for fast similarity search
- **HuggingFace Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Groq LLM** — `llama-3.3-70b-versatile` for ranking, fit analysis, and cover letter generation

---

## Documentation

| Document | Description |
|---|---|
| [`openapi.yaml`](./openapi.yaml) | Full OpenAPI 3.1 specification — schemas, examples, and error responses |
| [`docs/design.md`](./docs/design.md) | Architecture & design doc with Mermaid diagrams |

---

## Project Structure

```
backend/services/rag/
├── app/
│   ├── main.py              # FastAPI application entrypoint
│   ├── api/
│   │   └── routes.py        # /health and /api/match endpoints
│   ├── core/
│   │   ├── config.py        # Central config (env vars + defaults)
│   │   └── logging.py       # Structured key=value logger
│   ├── ingestion/
│   │   └── pipeline.py      # File → parse → chunk → FAISS ingestion
│   ├── retrieval/
│   │   └── pipeline.py      # Vector search + Groq LLM analysis
│   ├── vectorstore/
│   │   └── faiss_store.py   # FAISS load/save helpers
│   └── schemas/
│       └── models.py        # Pydantic API models + internal dataclasses
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
cd backend/services/rag
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export GROQ_API_KEY=your_groq_api_key
export VECTOR_STORE_PATH=vector_store   # optional, default: vector_store
```

### 3. Ingest roles data

```bash
cd backend/services/rag
python -m app.ingestion.pipeline
```

This reads `demofile.txt` (or your roles file), parses roles, creates embeddings, and saves the FAISS index to `vector_store/`.

### 4. Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/match` | Match a job description to candidate roles |

### `POST /api/match` — Request body

```json
{
  "query": "Senior Python developer with FastAPI and Docker experience...",
  "top_k": 5
}
```

### `POST /api/match` — Response

```json
{
  "results": [
    { "role": "Backend Developer", "tools": "Python, Docker", "projects": "LLM Pipeline", "score": 0.1823 }
  ],
  "analysis": {
    "summary_and_ranking": "...",
    "fit_analysis": "...",
    "cover_letter": "..."
  }
}
```

---

## Docker

```bash
cd backend/services/rag
docker build -t role-rag .
docker run -e GROQ_API_KEY=your_key -p 8000:8000 role-rag
```

---

## Configuration

All settings are read from environment variables (see `app/core/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `VECTOR_STORE_PATH` | `vector_store` | Path to FAISS index directory |
| `CHUNK_SIZE` | `450` | Text chunk size for ingestion |
| `CHUNK_OVERLAP` | `90` | Chunk overlap for ingestion |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

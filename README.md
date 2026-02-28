# Role RAG Multi-User Pipeline

A production-grade Retrieval-Augmented Generation (RAG) service built with FastAPI, FAISS, and Groq. This service supports multiple isolated users, allowing each user to maintain their own private vector store of roles and experience.

## Features

- **Multi-User Isolation**: Each user has a dedicated FAISS index. Data never leaks between users.
- **Automated Ingestion**: Upload plain-text roles files via API to build/update your personal vector store.
- **Intelligent Matching**: Semantic search finds the most relevant roles for a given job description.
- **LLM Analysis**: Optional integration with Groq (Llama 3.3 70B) for fit analysis, ranking, and cover letter generation.
- **Graceful Degradation**: Returns vector search results even if the LLM API key is missing or invalid.
- **Robust Testing**: Comprehensive 43-case test suite covering ingestion, accuracy, isolation, and edge cases.

## Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Vector Store**: FAISS (CPU)
- **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM**: Groq SDK (Llama 3.3)
- **Logging**: Structured JSON logging

---

## Quickstart

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/services/rag/requirements.txt
```

### 2. Configure API Key
Export your Groq API key:
```bash
export GROQ_API_KEY=gsk_your_key_here
```

### 3. Run the Service
```bash
# From the root directory
source venv/bin/activate
cd backend/services/rag
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

### `GET /health`
Liveness check. Returns `{"status": "ok"}`.

### `POST /api/ingest`
Ingest a roles file for a specific user.
- **Form Data**:
  - `user_id`: Unique string identifying the user.
  - `file`: Plain text file containing roles/experiences.

### `POST /api/match`
Search for matching roles and generate analysis.
- **JSON Body**:
  ```json
  {
    "user_id": "user1",
    "query": "Senior Python Developer with Docker",
    "top_k": 3
  }
  ```
- **Response**: Includes `results` (vector matches) and `analysis` (LLM output). If LLM fails, `analysis` is `null` and `analysis_error` contains the reason.

---

## Automated Testing

The project includes a robust test script that handles server lifecycle, dummy data ingestion, and isolation checks.

```bash
chmod +x test_service.sh
./test_service.sh
```

**What the tests verify:**
- Ingestion of multiple users (`user1`, `user2`, `user3`).
- **Isolation**: Verifying `user1` cannot see `user2`'s data, etc.
- **Accuracy**: Checking if the most relevant role is actually retrieved.
- **Edge Cases**: Handling empty queries, non-existent users, and malformed JSON.

---

## Project Structure

```text
.
├── backend/
│   └── services/
│       └── rag/
│           ├── app/              # Core application logic
│           │   ├── api/          # FastAPI routes
│           │   ├── core/         # Config and logging
│           │   ├── ingestion/    # Text splitting and indexing
│           │   ├── retrieval/    # Vector search and LLM calls
│           │   ├── schemas/      # Pydantic models
│           │   └── vectorstore/  # FAISS wrappers
│           ├── Dockerfile
│           └── requirements.txt
├── test_service.sh               # 43-case automated test suite
├── user1_roles.txt               # Sample test data
├── user2_roles.txt
└── user3_roles.txt
```

## License
MIT

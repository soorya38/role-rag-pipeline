from __future__ import annotations

import os


# -----------------------------------------------------------------------------
# Configuration — central place for all tunable settings.
# Values are read from environment variables with sensible defaults.
# -----------------------------------------------------------------------------

# Embedding model
EMBEDDING_MODEL_NAME: str = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)

# FAISS vector store path
VECTOR_STORE_PATH: str = os.environ.get("VECTOR_STORE_PATH", "vector_store")

# Groq LLM settings
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY: str = os.environ.get(
    "GROQ_API_KEY",
    "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",
)

# Ingestion chunking settings
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "450"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "90"))

# FastAPI server settings
HOST: str = os.environ.get("HOST", "0.0.0.0")
PORT: int = int(os.environ.get("PORT", "8000"))
RELOAD: bool = os.environ.get("RELOAD", "true").lower() == "true"

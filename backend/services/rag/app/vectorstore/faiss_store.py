from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.core.config import EMBEDDING_MODEL_NAME, VECTOR_STORE_PATH
from app.core.logging import get_logger

log = get_logger("vectorstore")


def _user_store_path(user_id: str, base_path: str = VECTOR_STORE_PATH) -> str:
    """Resolve the per-user FAISS index directory: <base>/<user_id>/"""
    return str(Path(base_path) / user_id)


def generate_embeddings() -> HuggingFaceEmbeddings:
    log.info("Loading local embedding model", extra={"model_name": EMBEDDING_MODEL_NAME})
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    log.info("Embedding model loaded", extra={"model_name": EMBEDDING_MODEL_NAME})
    return embeddings


# -----------------------------------------------------------------------------
# save_vector_store persists documents + embeddings to a per-user FAISS index.
#
# Stored at: {VECTOR_STORE_PATH}/{user_id}/
# -----------------------------------------------------------------------------
def save_vector_store(
    docs: List[Document],
    user_id: str,
    base_path: str = VECTOR_STORE_PATH,
) -> FAISS:
    store_path = _user_store_path(user_id, base_path)

    log.info(
        "Storing documents in FAISS vector DB",
        extra={"user_id": user_id, "doc_count": len(docs), "store_path": store_path},
    )

    embeddings = generate_embeddings()
    vector_store = FAISS.from_documents(docs, embedding=embeddings)

    Path(store_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(store_path)

    log.info(
        "FAISS vector store saved",
        extra={"user_id": user_id, "store_path": store_path, "doc_count": len(docs)},
    )
    return vector_store


# -----------------------------------------------------------------------------
# load_vector_store loads a user's persisted FAISS index from disk.
#
# Raises FileNotFoundError if no index has been ingested for this user yet.
# -----------------------------------------------------------------------------
def load_vector_store(
    user_id: str,
    base_path: str = VECTOR_STORE_PATH,
) -> FAISS:
    store_path = _user_store_path(user_id, base_path)

    if not Path(store_path).exists():
        raise FileNotFoundError(
            f"No vector store found for user '{user_id}'. "
            f"Please ingest a roles file for this user first via POST /api/ingest."
        )

    log.info(
        "Loading FAISS vector store",
        extra={"user_id": user_id, "store_path": store_path, "model_name": EMBEDDING_MODEL_NAME},
    )

    embeddings = generate_embeddings()
    vector_store = FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    log.info("FAISS vector store loaded", extra={"user_id": user_id, "store_path": store_path})
    return vector_store

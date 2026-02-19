from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Dict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# CHUNK_SIZE defines how large each text chunk should be when preparing
# data for RAG/embedding pipelines.
CHUNK_SIZE = 450

# CHUNK_OVERLAP ensures adjacent chunks share context.
CHUNK_OVERLAP = 90


# -----------------------------------------------------------------------------
# Structured logging setup.
#
# Outputs JSON-style log records to stdout with consistent fields:
#   timestamp, level, logger, message, and any extra context kwargs.
# -----------------------------------------------------------------------------
class StructuredFormatter(logging.Formatter):
    """Formats log records as structured JSON-style key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        # Base fields always present in every log line.
        fields = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra context passed via the `extra` kwarg.
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "id", "levelname", "levelno",
                "lineno", "message", "module", "msecs", "msg", "name",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "thread", "threadName", "taskName",
            }:
                fields[key] = value

        return " | ".join(f"{k}={v}" for k, v in fields.items())


def _get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

    return logger


log = _get_logger("ingestion")


# -----------------------------------------------------------------------------
# read_file reads a text file from the provided path and returns its content.
#
# Example:
# INPUT:
#   path = "roles.txt"
#
# roles.txt content:
#   role: Developer
#   role's tools: Python, Docker
#
# OUTPUT:
#   "role: Developer\nrole's tools: Python, Docker\n"
# -----------------------------------------------------------------------------
def read_file(path: str) -> str:
    log.info("Reading file", extra={"file_path": path})

    p = Path(path)

    if not p.exists():
        log.error("File not found", extra={"file_path": path})
        raise FileNotFoundError(f"File not found: {path}")

    content = p.read_text(encoding="utf-8")

    log.info(
        "File read successfully",
        extra={"file_path": path, "size_bytes": len(content.encode("utf-8"))},
    )

    return content


# -----------------------------------------------------------------------------
# parse_roles extracts structured role data from raw text.
#
# Example:
# INPUT:
# content = '''
# role: Backend Developer
# role's tools: Go, PostgreSQL
# role's projects: Auth API
#
# role: Frontend Developer
# role's tools: React, Tailwind
# role's projects: Dashboard UI
# '''
#
# OUTPUT:
# [
#   {
#     "role": "Backend Developer",
#     "tools": "Go, PostgreSQL",
#     "projects": "Auth API"
#   },
#   {
#     "role": "Frontend Developer",
#     "tools": "React, Tailwind",
#     "projects": "Dashboard UI"
#   }
# ]
# -----------------------------------------------------------------------------
def parse_roles(content: str) -> List[Dict]:
    log.info("Parsing roles from content", extra={"content_length": len(content)})

    role_blocks = re.split(r"\n\s*role\s*:", content, flags=re.IGNORECASE)

    parsed_roles = []

    for block in role_blocks:
        block = block.strip()

        if not block:
            continue

        lines = block.splitlines()
        role_name = lines[0].strip()

        tools = ""
        projects = ""

        for line in lines[1:]:

            if re.match(r"role's tools\s*:", line, re.IGNORECASE):
                tools = line.split(":", 1)[1].strip()

            elif re.match(r"role's projects\s*:", line, re.IGNORECASE):
                projects = line.split(":", 1)[1].strip()

        log.debug(
            "Parsed role block",
            extra={"role": role_name, "tools": tools, "projects": projects},
        )

        parsed_roles.append(
            {
                "role": role_name,
                "tools": tools,
                "projects": projects,
            }
        )

    log.info("Role parsing complete", extra={"roles_found": len(parsed_roles)})

    return parsed_roles


# -----------------------------------------------------------------------------
# build_documents converts parsed roles into LangChain Document objects.
#
# Example:
# INPUT:
# parsed_roles = [
#   {
#     "role": "Backend Developer",
#     "tools": "Go, PostgreSQL",
#     "projects": "Auth API"
#   }
# ]
# source = "roles.txt"
#
# OUTPUT (conceptual):
# [
#   Document(
#     page_content="Backend Developer",
#     metadata={role="Backend Developer", section="role", source="roles.txt"}
#   ),
#   Document(
#     page_content="Go, PostgreSQL",
#     metadata={role="Backend Developer", section="tools", source="roles.txt"}
#   ),
#   Document(
#     page_content="Auth API",
#     metadata={role="Backend Developer", section="projects", source="roles.txt"}
#   )
# ]
# -----------------------------------------------------------------------------
def build_documents(parsed_roles: List[Dict], source: str) -> List[Document]:
    log.info(
        "Building documents from parsed roles",
        extra={"source": source, "role_count": len(parsed_roles)},
    )

    docs: List[Document] = []

    for role in parsed_roles:
        role_name = role["role"]

        sections = {
            "role": role_name,
            "tools": role["tools"],
            "projects": role["projects"],
        }

        for section, text in sections.items():
            if not text:
                log.debug(
                    "Skipping empty section",
                    extra={"role": role_name, "section": section},
                )
                continue

            doc = Document(
                page_content=text,
                metadata={
                    "role": role_name,
                    "section": section,
                    "source": source,
                    "chunk_id": uuid4().hex,
                },
            )
            docs.append(doc)

            log.debug(
                "Document created",
                extra={
                    "role": role_name,
                    "section": section,
                    "content_length": len(text),
                },
            )

    log.info("Document build complete", extra={"document_count": len(docs)})

    return docs


# -----------------------------------------------------------------------------
# chunk_documents splits documents into smaller chunks for embedding.
#
# Example:
# INPUT:
# docs = [
#   Document(page_content="Very long text about backend development...")
# ]
#
# OUTPUT (conceptual):
# [
#   Document(page_content="Very long text about backend ..."),
#   Document(page_content="...continued chunk with overlap...")
# ]
#
# Exact chunk size depends on CHUNK_SIZE and CHUNK_OVERLAP.
# -----------------------------------------------------------------------------
def chunk_documents(docs: List[Document]) -> List[Document]:
    log.info(
        "Chunking documents",
        extra={
            "input_doc_count": len(docs),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", ",", " "],
    )

    chunked = splitter.split_documents(docs)

    log.info(
        "Chunking complete",
        extra={"input_doc_count": len(docs), "output_chunk_count": len(chunked)},
    )

    return chunked


# -----------------------------------------------------------------------------
# ingest_roles runs the full ingestion pipeline:
# read file → parse roles → build documents → chunk them.
#
# Example:
# INPUT:
#   file_path = "roles.txt"
#
# roles.txt:
#   role: Backend Developer
#   role's tools: Go, PostgreSQL
#   role's projects: Auth API
#
# OUTPUT:
#   List[Document] ready for embedding/vector DB ingestion.
#
# Example conceptual output:
# [
#   Document(page_content="Backend Developer", metadata=...),
#   Document(page_content="Go, PostgreSQL", metadata=...),
#   Document(page_content="Auth API", metadata=...)
# ]
# -----------------------------------------------------------------------------
def ingest_roles(file_path: str) -> List[Document]:
    log.info("Starting ingestion pipeline", extra={"file_path": file_path})

    content = read_file(file_path)
    parsed_roles = parse_roles(content)
    base_docs = build_documents(parsed_roles, file_path)
    chunked_docs = chunk_documents(base_docs)

    log.info(
        "Ingestion pipeline complete",
        extra={"file_path": file_path, "total_chunks": len(chunked_docs)},
    )

    return chunked_docs


# -----------------------------------------------------------------------------
# generate_local_embeddings creates embeddings using a local HuggingFace model.
#
# Uses sentence-transformers model locally (no external API required).
#
# Example:
# INPUT:
#   docs = [Document("Backend Developer"), ...]
#
# OUTPUT:
#   embeddings object ready for vector DB ingestion.
# -----------------------------------------------------------------------------
def generate_local_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    log.info("Loading local embedding model", extra={"model_name": model_name})

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    log.info("Embedding model loaded", extra={"model_name": model_name})

    return embeddings


# -----------------------------------------------------------------------------
# store_in_local_vector_db stores documents + embeddings in FAISS vector DB.
#
# FAISS runs fully locally and supports similarity search efficiently.
#
# Example:
# INPUT:
#   docs = chunked documents
#
# OUTPUT:
#   persisted FAISS index in ./vector_store directory.
# -----------------------------------------------------------------------------
def store_in_local_vector_db(docs: List[Document]):
    log.info(
        "Storing documents in local FAISS vector DB",
        extra={"doc_count": len(docs)},
    )

    embeddings = generate_local_embeddings()

    vector_store = FAISS.from_documents(
        docs,
        embedding=embeddings,
    )

    save_path = "vector_store"
    vector_store.save_local(save_path)

    log.info(
        "FAISS vector store saved",
        extra={"save_path": save_path, "doc_count": len(docs)},
    )

    return vector_store


if __name__ == "__main__":
    FILE_PATH = "demofile.txt"

    log.info("Script started", extra={"file_path": FILE_PATH})

    docs = ingest_roles(FILE_PATH)

    print(f"Generated {len(docs)} chunks")

    for d in docs[:3]:
        print("\n---")
        print(d.page_content)
        print(d.metadata)

    # Generate embeddings + store in local FAISS vector DB
    store_in_local_vector_db(docs)

    print("\nEmbeddings stored in local FAISS vector DB.")

    log.info("Script finished successfully", extra={"total_chunks": len(docs)})
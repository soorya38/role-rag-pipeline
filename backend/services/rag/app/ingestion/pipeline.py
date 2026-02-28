from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.core.logging import get_logger
from app.vectorstore.faiss_store import save_vector_store

log = get_logger("ingestion")


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
        parsed_roles.append({"role": role_name, "tools": tools, "projects": projects})

    log.info("Role parsing complete", extra={"roles_found": len(parsed_roles)})
    return parsed_roles


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

    log.info("Document build complete", extra={"document_count": len(docs)})
    return docs


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
# run_ingestion_pipeline — main entry point.
#
# Accepts raw text content + a user_id so the API layer can pass uploaded
# file bytes directly without touching the filesystem.
#
# Stores the resulting FAISS index at: vector_store/<user_id>/
# -----------------------------------------------------------------------------
def run_ingestion_pipeline(content: str, user_id: str) -> List[Document]:
    log.info("Starting ingestion pipeline", extra={"user_id": user_id})

    parsed_roles = parse_roles(content)
    base_docs = build_documents(parsed_roles, source=f"upload:{user_id}")
    chunked_docs = chunk_documents(base_docs)

    save_vector_store(chunked_docs, user_id=user_id)

    log.info(
        "Ingestion pipeline complete",
        extra={"user_id": user_id, "total_chunks": len(chunked_docs)},
    )
    return chunked_docs

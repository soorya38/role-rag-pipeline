from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------------------------------------------------
# Structured logging — same setup as ingestion pipeline for consistency.
# -----------------------------------------------------------------------------
class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = {
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
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


log = _get_logger("retrieval")


# -----------------------------------------------------------------------------
# RetrievalResult holds the aggregated role/tools/projects for a matched role.
# -----------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    role: str
    tools: str = ""
    projects: str = ""
    score: float = 0.0

    def as_dict(self) -> Dict:
        return {
            "role": self.role,
            "tools": self.tools,
            "projects": self.projects,
            "score": round(self.score, 4),
        }


# -----------------------------------------------------------------------------
# load_vector_store loads a persisted FAISS index from disk.
#
# Example:
# INPUT:
#   store_path = "vector_store"
#
# OUTPUT:
#   FAISS vector store object ready for similarity search.
# -----------------------------------------------------------------------------
def load_vector_store(store_path: str = "vector_store") -> FAISS:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    log.info(
        "Loading FAISS vector store",
        extra={"store_path": store_path, "model_name": model_name},
    )

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    vector_store = FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    log.info("FAISS vector store loaded", extra={"store_path": store_path})

    return vector_store


# -----------------------------------------------------------------------------
# retrieve_raw_docs performs similarity search against the vector store.
#
# Example:
# INPUT:
#   query = "software engineer with Docker and Python"
#   vector_store = <FAISS instance>
#   top_k = 10
#
# OUTPUT:
#   List of (Document, score) tuples sorted by relevance.
# -----------------------------------------------------------------------------
def retrieve_raw_docs(
    query: str,
    vector_store: FAISS,
    top_k: int = 10,
) -> List[tuple[Document, float]]:
    log.info(
        "Running similarity search",
        extra={"query_length": len(query), "top_k": top_k},
    )

    results: List[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, k=top_k
    )

    log.info(
        "Similarity search complete",
        extra={"results_returned": len(results)},
    )

    for doc, score in results:
        log.debug(
            "Raw result",
            extra={
                "role": doc.metadata.get("role"),
                "section": doc.metadata.get("section"),
                "score": round(float(score), 4),
                "content_preview": doc.page_content[:60],
            },
        )

    return results


# -----------------------------------------------------------------------------
# aggregate_results groups raw search hits by role and merges sections
# (role name, tools, projects) into unified RetrievalResult objects.
#
# The best (lowest) score per role is kept as the representative score,
# since FAISS returns L2 distances (lower = more similar).
#
# Example:
# INPUT:
# raw_docs = [
#   (Document(page_content="Go, PostgreSQL", metadata={role: "Backend Dev", section: "tools"}), 0.23),
#   (Document(page_content="Auth API", metadata={role: "Backend Dev", section: "projects"}), 0.41),
# ]
#
# OUTPUT:
# [
#   RetrievalResult(role="Backend Dev", tools="Go, PostgreSQL", projects="Auth API", score=0.23)
# ]
# -----------------------------------------------------------------------------
def aggregate_results(
    raw_docs: List[tuple[Document, float]],
) -> List[RetrievalResult]:
    log.info("Aggregating results by role", extra={"raw_doc_count": len(raw_docs)})

    role_map: Dict[str, RetrievalResult] = {}

    for doc, score in raw_docs:
        meta = doc.metadata
        role_name: str = meta.get("role", "Unknown")
        section: str = meta.get("section", "")
        dist = float(score)

        if role_name not in role_map:
            role_map[role_name] = RetrievalResult(role=role_name, score=dist)
            log.debug("New role entry created", extra={"role": role_name})
        else:
            # Keep the best (lowest L2 distance) score as the representative.
            if dist < role_map[role_name].score:
                role_map[role_name].score = dist

        result = role_map[role_name]

        if section == "tools" and not result.tools:
            result.tools = doc.page_content
            log.debug("Tools set", extra={"role": role_name, "tools": result.tools})

        elif section == "projects" and not result.projects:
            result.projects = doc.page_content
            log.debug(
                "Projects set", extra={"role": role_name, "projects": result.projects}
            )

    aggregated = sorted(role_map.values(), key=lambda r: r.score)

    log.info(
        "Aggregation complete",
        extra={"unique_roles_found": len(aggregated)},
    )

    return aggregated


# -----------------------------------------------------------------------------
# retrieve fetches roles, tools, and projects matching the provided query.
#
# This is the main entry point for the retrieval pipeline.
#
# Example:
# INPUT:
#   query = "software engineer with Docker and Python working on LLM projects"
#   store_path = "vector_store"
#   top_k = 10
#
# OUTPUT:
# [
#   RetrievalResult(
#     role="Backend Developer",
#     tools="Python, Docker",
#     projects="LLM Evaluation Pipeline",
#     score=0.1823
#   ),
#   ...
# ]
# -----------------------------------------------------------------------------
def retrieve(
    query: str,
    store_path: str = "vector_store",
    top_k: int = 10,
) -> List[RetrievalResult]:
    log.info(
        "Retrieval pipeline started",
        extra={"store_path": store_path, "top_k": top_k},
    )

    vector_store = load_vector_store(store_path)
    raw_docs = retrieve_raw_docs(query, vector_store, top_k=top_k)
    results = aggregate_results(raw_docs)

    log.info(
        "Retrieval pipeline complete",
        extra={"roles_retrieved": len(results)},
    )

    return results


# -----------------------------------------------------------------------------
# print_results pretty-prints retrieval results to stdout.
# -----------------------------------------------------------------------------
def print_results(results: List[RetrievalResult]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  RETRIEVAL RESULTS  ({len(results)} role(s) matched)")
    print(f"{'=' * 60}")

    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] Role     : {r.role}")
        print(f"    Tools    : {r.tools or '—'}")
        print(f"    Projects : {r.projects or '—'}")
        print(f"    Score    : {round(r.score, 4)}  (L2 distance, lower = better)")

    print(f"\n{'=' * 60}\n")


# -----------------------------------------------------------------------------
# Main — run with the provided job description query.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    query = """About the job

About the projects: we are building LLM evaluation and training datasets to train LLM to work on realistic software engineering problems. One of our approaches, in this project, is to build verifiable SWE tasks based on public repository histories in a synthetic approach with human-in-the-loop; while expanding the dataset coverage to different types of tasks in terms of programming language, difficulty level, and etc.

About the Role:
We are looking for experienced software engineers (tech lead level) who are familiar with high-quality public GitHub repositories and can contribute to this project. This role involves hands-on software engineering work, including development environment automation, issue triaging, and evaluating test coverage and quality

Why Join Us?
Turing is one of the world's fastest-growing AI companies accelerating the advancement and deployment of powerful AI systems. You'll be at the forefront of evaluating how LLMs interact with real code, influencing the future of AI-assisted software development. This is a unique opportunity to blend practical software engineering with AI research.

What does day-to-day look like:
* Analyze and triage GitHub issues across trending open-source libraries.
* Set up and configure code repositories, including Dockerization and environment setup.
* Evaluating unit test coverage and quality.
* Modify and run codebases locally to assess LLM performance in bug-fixing scenarios.
* Collaborate with researchers to design and identify repositories and issues that are challenging for LLMs.
* Opportunities to lead a team of junior engineers to collaborate on projects.

Required Skills:
* Minimum 3+ years of overall experience
* Strong experience with at least one of the following languages: C++
* Proficiency with Git, Docker, and basic software pipeline setup.
* Ability to understand and navigate complex codebases.
* Comfortable running, modifying, and testing real-world projects locally.
* Experience contributing to or evaluating open-source projects is a plus.

Nice to Have:
* Previous participation in LLM research or evaluation projects.
* Experience building or testing developer tools or automation agents.
* Perks of Freelancing With Turing:
* Work in a fully remote environment.
* Opportunity to work on cutting-edge AI projects with leading LLM companies.

Offer Details:
* Commitments Required: At least 4 hours per day and minimum 20 hours per week with overlap of 4 hours with PST. (We have 3 options of time commitment: 20 hrs/week, 30 hrs/week or 40 hrs/week)
* Employment type: Contractor assignment (no medical/paid leave)
"""

    log.info("Script started")

    results = retrieve(query, store_path="vector_store", top_k=10)
    print_results(results)

    log.info("Script finished", extra={"roles_matched": len(results)})

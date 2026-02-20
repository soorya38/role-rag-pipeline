from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import List, Dict

from groq import Groq

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
# Groq client setup.
#
# Reads GROQ_API_KEY from environment. Raises clearly if the key is missing.
# Model is pinned to llama-3.3-70b-versatile.
# -----------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = "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" #os.environ.get("GROQ_API_KEY")


def _get_groq_client() -> Groq:
    api_key = GROQ_API_KEY

    if not api_key:
        log.error("GROQ_API_KEY environment variable not set")
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it before running:\n"
            "  export GROQ_API_KEY=your_key_here"
        )

    log.info("Groq client initialised", extra={"model": GROQ_MODEL})
    return Groq(api_key=api_key)


def _call_groq(client: Groq, system_prompt: str, user_prompt: str) -> str:
    """Low-level wrapper around the Groq chat completions endpoint."""
    log.info(
        "Calling Groq LLM",
        extra={
            "model": GROQ_MODEL,
            "system_prompt_length": len(system_prompt),
            "user_prompt_length": len(user_prompt),
        },
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    result = response.choices[0].message.content.strip()

    log.info(
        "Groq LLM response received",
        extra={"response_length": len(result)},
    )

    return result


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

    def as_text_block(self) -> str:
        """Renders the result as a readable text block for LLM prompts."""
        return (
            f"Role: {self.role}\n"
            f"Tools: {self.tools or '—'}\n"
            f"Projects: {self.projects or '—'}\n"
            f"Similarity Score (L2, lower=better): {round(self.score, 4)}"
        )


# -----------------------------------------------------------------------------
# LLMAnalysis holds the three outputs produced by the Groq LLM calls.
# -----------------------------------------------------------------------------
@dataclass
class LLMAnalysis:
    summary_and_ranking: str
    fit_analysis: str
    cover_letter: str


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
# _build_roles_context serialises all RetrievalResult objects into a single
# text block suitable for injection into LLM prompts.
# -----------------------------------------------------------------------------
def _build_roles_context(results: List[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{i}] {r.as_text_block()}" for i, r in enumerate(results, start=1)
    )


# -----------------------------------------------------------------------------
# llm_summarise_and_rank asks the LLM to summarise and rank the retrieved
# roles against the job query.
#
# Example:
# INPUT:
#   query        = "LLM evaluation engineer with C++ and Docker..."
#   results      = [RetrievalResult(...), ...]
#   groq_client  = <Groq instance>
#
# OUTPUT:
#   Plain-text summary with ranked roles and brief reasoning per role.
# -----------------------------------------------------------------------------
def llm_summarise_and_rank(
    query: str,
    results: List[RetrievalResult],
    groq_client: Groq,
) -> str:
    log.info("LLM task: summarise and rank", extra={"role_count": len(results)})

    system_prompt = (
        "You are a technical recruiter assistant. "
        "Given a job description and a list of candidate roles retrieved from a vector database, "
        "summarise each role and rank them from most to least relevant to the job. "
        "For each role provide a one-sentence justification. "
        "Be concise and structured."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"RETRIEVED ROLES:\n{_build_roles_context(results)}\n\n"
        "Task: Summarise each role in one sentence, then provide a final ranked list "
        "(1 = best match) with a brief reason for each ranking."
    )

    result = _call_groq(groq_client, system_prompt, user_prompt)

    log.info("LLM summarise and rank complete")

    return result


# -----------------------------------------------------------------------------
# llm_fit_analysis asks the LLM to produce a fit analysis including an
# estimated match percentage and detailed reasoning per role.
#
# Example:
# INPUT:
#   query        = "LLM evaluation engineer with C++ and Docker..."
#   results      = [RetrievalResult(...), ...]
#   groq_client  = <Groq instance>
#
# OUTPUT:
#   Structured fit analysis with match % and reasoning for each role.
# -----------------------------------------------------------------------------
def llm_fit_analysis(
    query: str,
    results: List[RetrievalResult],
    groq_client: Groq,
) -> str:
    log.info("LLM task: fit analysis", extra={"role_count": len(results)})

    system_prompt = (
        "You are a technical recruiter performing a detailed fit analysis. "
        "For each candidate role, compare it against the job description and produce: "
        "1) A match percentage (0-100%) based on skills, tools, and project relevance. "
        "2) Key strengths — what aligns well. "
        "3) Key gaps — what is missing or misaligned. "
        "Be analytical and specific."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"RETRIEVED ROLES:\n{_build_roles_context(results)}\n\n"
        "Task: For EACH role produce a fit analysis with match %, strengths, and gaps."
    )

    result = _call_groq(groq_client, system_prompt, user_prompt)

    log.info("LLM fit analysis complete")

    return result


# -----------------------------------------------------------------------------
# llm_cover_letter asks the LLM to draft a tailored cover letter using the
# best-matching role's tools and projects aligned to the job description.
#
# Example:
# INPUT:
#   query        = "LLM evaluation engineer with C++ and Docker..."
#   results      = [RetrievalResult(...), ...]  # first item is best match
#   groq_client  = <Groq instance>
#
# OUTPUT:
#   A professional cover letter tailored to the job and best-matched role.
# -----------------------------------------------------------------------------
def llm_cover_letter(
    query: str,
    results: List[RetrievalResult],
    groq_client: Groq,
) -> str:
    log.info(
        "LLM task: cover letter",
        extra={"best_match_role": results[0].role if results else "N/A"},
    )

    best_match = results[0] if results else None

    system_prompt = (
        "You are a professional career coach. "
        "Draft a concise, compelling cover letter (3-4 paragraphs) for a candidate "
        "applying to the provided job. Use the candidate's role, tools, and project "
        "experience to tailor the letter. Keep the tone professional and enthusiastic."
    )

    candidate_context = (
        best_match.as_text_block()
        if best_match
        else "No candidate role information available."
    )

    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"CANDIDATE PROFILE (best matched role):\n{candidate_context}\n\n"
        "Task: Write a tailored cover letter for this candidate applying to the above job."
    )

    result = _call_groq(groq_client, system_prompt, user_prompt)

    log.info("LLM cover letter complete")

    return result


# -----------------------------------------------------------------------------
# run_llm_analysis orchestrates all three LLM tasks and returns an LLMAnalysis.
#
# Example:
# INPUT:
#   query   = "LLM evaluation engineer..."
#   results = [RetrievalResult(...), ...]
#
# OUTPUT:
#   LLMAnalysis(
#     summary_and_ranking = "...",
#     fit_analysis        = "...",
#     cover_letter        = "..."
#   )
# -----------------------------------------------------------------------------
def run_llm_analysis(
    query: str,
    results: List[RetrievalResult],
) -> LLMAnalysis:
    log.info("Starting LLM analysis pipeline", extra={"role_count": len(results)})

    groq_client = _get_groq_client()

    summary_and_ranking = llm_summarise_and_rank(query, results, groq_client)
    fit_analysis = llm_fit_analysis(query, results, groq_client)
    cover_letter = llm_cover_letter(query, results, groq_client)

    log.info("LLM analysis pipeline complete")

    return LLMAnalysis(
        summary_and_ranking=summary_and_ranking,
        fit_analysis=fit_analysis,
        cover_letter=cover_letter,
    )


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
# print_llm_analysis pretty-prints all three LLM outputs to stdout.
# -----------------------------------------------------------------------------
def print_llm_analysis(analysis: LLMAnalysis) -> None:
    sections = [
        ("SUMMARY & RANKING", analysis.summary_and_ranking),
        ("FIT ANALYSIS", analysis.fit_analysis),
        ("COVER LETTER", analysis.cover_letter),
    ]

    for title, content in sections:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(content)

    print(f"\n{'=' * 60}\n")


# -----------------------------------------------------------------------------
# Main — run with the provided job description query.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    query = """About The Role

We are seeking a skilled Backend Developer to join our engineering team. Unlike traditional siloed development roles, this position is designed for an engineer who values end-to-end ownership. You will not only write code but also ensure its performance, reliability, and stability in a live production environment.

You will drive performance optimization across our AWS-hosted Node.js, GraphQL, and MongoDB stack. This is a hands-on development role where "support" means engineering solutions to complex production challenges and preventing them from recurring. You will act as a critical bridge between product and engineering, gaining deep system mastery through direct involvement in quality assurance and release cycles.

Work Schedule

Mandatory Overlap: You must be available to collaborate with the US team between 11:00 PM IST and 3:00 AM IST.

Purpose: This overlap ensures seamless handoffs, collaborative debugging, and direct interaction with US-based product and engineering teams.

Key Responsibilities

Core Development & Performance Engineering

Design & Optimize: Write and fine-tune complex MongoDB aggregation pipelines for high-volume analytics, dashboards, and large dataset operations.

Backend Architecture: Maintain and enhance Node.js and GraphQL services, ensuring scalable schemas and efficient resolvers.

System Reliability: Take ownership of production uptime by diagnosing and resolving backend issues within the AWS ecosystem (Elastic Beanstalk, EC2, CloudFront).

Production Ownership & Quality Assurance

Release Integrity: Collaborate with QA teams during pre-release testing to gain a holistic understanding of new features, ensuring you know the code inside and out before it hits production.

Root Cause Analysis: Go beyond "patching" bugs. Analyze logs and metrics (CloudWatch, ELB) to identify architectural bottlenecks and implement long-term reliability improvements.

Incident Response: Participate in on-call rotations to maintain service availability, acting as the engineering authority during incident response.

Technical Requirements

Core Stack:

MongoDB: Deep expertise in aggregation frameworks and query optimization is essential.

Node.js: Proficiency with Express or Apollo Server.

GraphQL: Strong grasp of schema design and resolver implementation.

Infrastructure & Tooling:

AWS: Hands-on experience with Beanstalk, EC2, S3, and ELB.

Observability: Familiarity with logging tools (CloudWatch, New Relic, or Datadog) to trace data flow and performance issues.

DevOps: Experience with CI/CD pipelines (GitHub Actions) and deployment troubleshooting is highly preferred.

Soft Skills & Mindset

Ownership: You view code in production as your responsibility. You are focused on long-term stability rather than quick fixes.

Communication: You can clearly articulate technical complexities to distributed teams across different time zones.

Adaptability: You thrive in a startup environment and are eager to learn the product inside-out through direct exposure to QA and triage processes.

Engagement Type: Fulltime

Direct-hire on the Delightree Payroll

Job Type: Permanent

Location: Remote

Working time: 11 PM to 3 AM IST ( Rest hours anytime during day time)

Interview Process - 2 Rounds
"""

    log.info("Script started")

    # ── Stage 1: Vector retrieval ─────────────────────────────────────────────
    results = retrieve(query, store_path="vector_store", top_k=10)
    print_results(results)

    # ── Stage 2: LLM analysis via Groq ───────────────────────────────────────
    analysis = run_llm_analysis(query, results)
    print_llm_analysis(analysis)

    log.info("Script finished", extra={"roles_matched": len(results)})
from __future__ import annotations

from typing import List, Dict

from groq import Groq
from langchain_core.documents import Document

from app.core.config import GROQ_API_KEY, GROQ_MODEL
from app.core.logging import get_logger
from app.schemas.models import RetrievalResult, LLMAnalysis
from app.vectorstore.faiss_store import load_vector_store

log = get_logger("retrieval")


# -----------------------------------------------------------------------------
# Groq client setup.
# -----------------------------------------------------------------------------
def _get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        log.error("GROQ_API_KEY environment variable not set")
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it before running:\n"
            "  export GROQ_API_KEY=your_key_here"
        )
    log.info("Groq client initialised", extra={"model": GROQ_MODEL})
    return Groq(api_key=GROQ_API_KEY)


def _call_groq(client: Groq, system_prompt: str, user_prompt: str) -> str:
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
    log.info("Groq LLM response received", extra={"response_length": len(result)})
    return result


# -----------------------------------------------------------------------------
# retrieve_raw_docs — similarity search scoped to the user's FAISS index.
# -----------------------------------------------------------------------------
def retrieve_raw_docs(
    query: str,
    vector_store,
    top_k: int = 10,
) -> List[tuple[Document, float]]:
    log.info(
        "Running similarity search",
        extra={"query_length": len(query), "top_k": top_k},
    )

    results: List[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, k=top_k
    )

    log.info("Similarity search complete", extra={"results_returned": len(results)})

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
            if dist < role_map[role_name].score:
                role_map[role_name].score = dist

        result = role_map[role_name]

        if section == "tools" and not result.tools:
            result.tools = doc.page_content
        elif section == "projects" and not result.projects:
            result.projects = doc.page_content

    aggregated = sorted(role_map.values(), key=lambda r: r.score)
    log.info("Aggregation complete", extra={"unique_roles_found": len(aggregated)})
    return aggregated


# -----------------------------------------------------------------------------
# LLM analysis helpers.
# -----------------------------------------------------------------------------
def _build_roles_context(results: List[RetrievalResult]) -> str:
    return "\n\n".join(
        f"[{i}] {r.as_text_block()}" for i, r in enumerate(results, start=1)
    )


def llm_summarise_and_rank(query: str, results: List[RetrievalResult], client: Groq) -> str:
    log.info("LLM task: summarise and rank", extra={"role_count": len(results)})
    system_prompt = (
        "You are a technical recruiter assistant. "
        "Given a job description and a list of candidate roles retrieved from a vector database, "
        "summarise each role and rank them from most to least relevant to the job. "
        "For each role provide a one-sentence justification. Be concise and structured."
    )
    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"RETRIEVED ROLES:\n{_build_roles_context(results)}\n\n"
        "Task: Summarise each role in one sentence, then provide a final ranked list "
        "(1 = best match) with a brief reason for each ranking."
    )
    result = _call_groq(client, system_prompt, user_prompt)
    log.info("LLM summarise and rank complete")
    return result


def llm_fit_analysis(query: str, results: List[RetrievalResult], client: Groq) -> str:
    log.info("LLM task: fit analysis", extra={"role_count": len(results)})
    system_prompt = (
        "You are a technical recruiter performing a detailed fit analysis. "
        "For each candidate role, compare it against the job description and produce: "
        "1) A match percentage (0-100%). 2) Key strengths. 3) Key gaps. "
        "Be analytical and specific."
    )
    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"RETRIEVED ROLES:\n{_build_roles_context(results)}\n\n"
        "Task: For EACH role produce a fit analysis with match %, strengths, and gaps."
    )
    result = _call_groq(client, system_prompt, user_prompt)
    log.info("LLM fit analysis complete")
    return result


def llm_cover_letter(query: str, results: List[RetrievalResult], client: Groq) -> str:
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
        best_match.as_text_block() if best_match else "No candidate role information available."
    )
    user_prompt = (
        f"JOB DESCRIPTION:\n{query}\n\n"
        f"CANDIDATE PROFILE (best matched role):\n{candidate_context}\n\n"
        "Task: Write a tailored cover letter for this candidate applying to the above job."
    )
    result = _call_groq(client, system_prompt, user_prompt)
    log.info("LLM cover letter complete")
    return result


def run_llm_analysis(query: str, results: List[RetrievalResult]) -> LLMAnalysis:
    log.info("Starting LLM analysis pipeline", extra={"role_count": len(results)})
    client = _get_groq_client()
    summary_and_ranking = llm_summarise_and_rank(query, results, client)
    fit_analysis = llm_fit_analysis(query, results, client)
    cover_letter = llm_cover_letter(query, results, client)
    log.info("LLM analysis pipeline complete")
    return LLMAnalysis(
        summary_and_ranking=summary_and_ranking,
        fit_analysis=fit_analysis,
        cover_letter=cover_letter,
    )


# -----------------------------------------------------------------------------
# retrieve — main entry point, scoped to a specific user's FAISS index.
# -----------------------------------------------------------------------------
def retrieve(
    query: str,
    user_id: str,
    top_k: int = 10,
) -> List[RetrievalResult]:
    log.info(
        "Retrieval pipeline started",
        extra={"user_id": user_id, "top_k": top_k},
    )
    vector_store = load_vector_store(user_id=user_id)
    raw_docs = retrieve_raw_docs(query, vector_store, top_k=top_k)
    results = aggregate_results(raw_docs)
    log.info(
        "Retrieval pipeline complete",
        extra={"user_id": user_id, "roles_retrieved": len(results)},
    )
    return results

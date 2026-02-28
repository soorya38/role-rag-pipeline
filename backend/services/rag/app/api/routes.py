from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.core.logging import get_logger
from app.schemas.models import (
    IngestResponse,
    MatchRequest,
    MatchResponse,
    RetrievalResult,
    RetrievalResultSchema,
    LLMAnalysisSchema,
)
from app.ingestion.pipeline import run_ingestion_pipeline
from app.retrieval.pipeline import retrieve, run_llm_analysis

log = get_logger("api.routes")

router = APIRouter()


# -----------------------------------------------------------------------------
# GET /health — liveness probe
# -----------------------------------------------------------------------------
@router.get("/health")
def health_check():
    return {"status": "ok"}


# -----------------------------------------------------------------------------
# POST /api/ingest
#
# Accepts a plain-text roles file upload and a user_id form field.
# Builds and saves a per-user FAISS index at vector_store/<user_id>/.
# Re-ingesting for the same user_id overwrites their existing index.
# -----------------------------------------------------------------------------
@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_roles(
    user_id: str = Form(..., description="Unique identifier for the user."),
    file: UploadFile = File(..., description="Plain-text roles file (UTF-8)."),
):
    try:
        log.info("Received ingest request", extra={"user_id": user_id, "upload_file": file.filename})

        raw_bytes = await file.read()
        content = raw_bytes.decode("utf-8")

        chunked_docs = run_ingestion_pipeline(content=content, user_id=user_id)

        return IngestResponse(
            user_id=user_id,
            chunks_indexed=len(chunked_docs),
            message=f"Successfully indexed {len(chunked_docs)} chunks for user '{user_id}'.",
        )
    except Exception as e:
        log.error(f"Ingestion failed for user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------------------------------------------------------
# POST /api/match
#
# Retrieves candidate roles from the user's own FAISS index and runs
# LLM analysis. Returns a 404 if no index exists for the given user_id.
# -----------------------------------------------------------------------------
@router.post("/api/match", response_model=MatchResponse)
def match_role(request: MatchRequest):
    try:
        log.info(
            "Received match request",
            extra={
                "user_id": request.user_id,
                "query_length": len(request.query),
                "top_k": request.top_k,
            },
        )

        # 1. Vector retrieval scoped to this user
        results: List[RetrievalResult] = retrieve(
            query=request.query,
            user_id=request.user_id,
            top_k=request.top_k,
        )

        # 2. LLM analysis (degrades gracefully — returns None on error)
        analysis, analysis_error = run_llm_analysis(request.query, results)

        # 3. Format response
        results_formatted = [
            RetrievalResultSchema(
                role=r.role,
                tools=r.tools,
                projects=r.projects,
                score=round(r.score, 4),
            )
            for r in results
        ]

        analysis_formatted = (
            LLMAnalysisSchema(
                summary_and_ranking=analysis.summary_and_ranking,
                fit_analysis=analysis.fit_analysis,
                cover_letter=analysis.cover_letter,
            )
            if analysis
            else None
        )

        return MatchResponse(
            user_id=request.user_id,
            results=results_formatted,
            analysis=analysis_formatted,
            analysis_error=analysis_error,
        )

    except FileNotFoundError as e:
        log.warning(f"No index found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error(f"Error processing match request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

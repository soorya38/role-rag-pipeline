from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal dataclasses — used by the retrieval & ingestion pipelines.
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """Aggregated role information returned by the retrieval pipeline."""

    role: str
    tools: str = ""
    projects: str = ""
    score: float = 0.0

    def as_dict(self) -> dict:
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


@dataclass
class LLMAnalysis:
    """Three outputs produced by the Groq LLM analysis calls."""

    summary_and_ranking: str
    fit_analysis: str
    cover_letter: str


# ---------------------------------------------------------------------------
# Pydantic models — used by the FastAPI routes for request/response schemas.
# ---------------------------------------------------------------------------

class MatchRequest(BaseModel):
    user_id: str
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1)


class IngestResponse(BaseModel):
    user_id: str
    chunks_indexed: int
    message: str


class RetrievalResultSchema(BaseModel):
    role: str
    tools: str
    projects: str
    score: float


class LLMAnalysisSchema(BaseModel):
    summary_and_ranking: str
    fit_analysis: str
    cover_letter: str


class MatchResponse(BaseModel):
    user_id: str
    results: List[RetrievalResultSchema]
    # None when LLM is unavailable (e.g. missing/invalid GROQ_API_KEY)
    analysis: Optional[LLMAnalysisSchema] = None
    analysis_error: Optional[str] = None

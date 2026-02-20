from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import uvicorn
import json

# Import retrieval pipeline functions and classes
from retrival_pipeline import (
    retrieve,
    run_llm_analysis,
    RetrievalResult,
    LLMAnalysis,
    _get_logger
)

log = _get_logger("fastapi")

app = FastAPI(
    title="Role RAG API",
    description="API for fetching candidate roles using a RAG pipeline based on job descriptions.",
    version="1.0.0"
)

# Allow CORS so frontends can easily connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models for Request/Response Schemas (API Spec)
# -----------------------------------------------------------------------------
class MatchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10

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
    results: List[RetrievalResultSchema]
    analysis: LLMAnalysisSchema

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/match", response_model=MatchResponse)
def match_role(request: MatchRequest):
    try:
        log.info(f"Received match request. Query length: {len(request.query)}, top_k: {request.top_k}")
        
        # 1. Retrieve candidates
        results_objs: List[RetrievalResult] = retrieve(request.query, store_path="vector_store", top_k=request.top_k)
        
        # 2. Run LLM on candidates
        analysis_obj: LLMAnalysis = run_llm_analysis(request.query, results_objs)
        
        # Format results for response
        results_formatted = [
            RetrievalResultSchema(
                role=r.role,
                tools=r.tools,
                projects=r.projects,
                score=round(r.score, 4)
            )
            for r in results_objs
        ]
        
        analysis_formatted = LLMAnalysisSchema(
            summary_and_ranking=analysis_obj.summary_and_ranking,
            fit_analysis=analysis_obj.fit_analysis,
            cover_letter=analysis_obj.cover_letter
        )
        
        return MatchResponse(
            results=results_formatted,
            analysis=analysis_formatted
        )
        
    except Exception as e:
        log.error(f"Error processing match request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import HOST, PORT, RELOAD

app = FastAPI(
    title="Role RAG API",
    description="API for fetching candidate roles using a RAG pipeline based on job descriptions.",
    version="1.0.0",
)

# Allow CORS so frontends (e.g. Streamlit) can connect.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=RELOAD)

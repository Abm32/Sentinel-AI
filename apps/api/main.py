"""
Sentinel Clinical API — FastAPI app entrypoint.

Run with:

    uvicorn apps.api.main:app --reload

from the project root (so the `packages` import root resolves correctly).

Router registration grows incrementally as the API layer is built out —
see apps/api/routers/. Each router owns one resource area:
  - health.py       -> GET /api/health
  - investigations.py -> POST/GET /api/investigations, WebSocket stream
  - upload.py        -> POST /api/investigations/{case_id}/upload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.api.routers import health, investigations, upload

app = FastAPI(
    title="Sentinel Clinical API",
    description=(
        "Autonomous adverse drug event investigation engine. "
        "Orchestrates the LangGraph investigation pipeline "
        "(planner -> retrieval -> tool_agent -> hypothesis -> reporter -> reviewer) "
        "and exposes it over HTTP + WebSocket for the dashboard frontend."
    ),
    version="0.1.0",
)

# Permissive CORS for local development / hackathon demo. Tighten
# allow_origins to the actual frontend origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(investigations.router, prefix="/api")
app.include_router(upload.router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"service": "sentinel-clinical-api", "docs": "/docs"}

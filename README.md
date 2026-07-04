# Sentinel Clinical

Sentinel Clinical is an autonomous adverse drug event investigation
engine. Given a clinical incident (e.g. "patient admitted after
fluorouracil therapy, presenting with neutropenia"), it plans an
investigation, gathers evidence, generates competing hypotheses with
honest confidence scores, drafts a report, and puts that report through
a second, skeptical AI reviewer before it can be finalized.

The design commitment that runs through the whole system: **Sentinel
refuses to guess.** If the evidence needed to confirm a root cause
(e.g. a pharmacogenomic phenotype) was never retrieved, the investigation
says so explicitly instead of fabricating a confident answer — and the
Reviewer approves that refusal, rather than treating "I don't know" as a
failure. See `docs/ARCHITECTURE.md` for how each agent enforces this.

## Architecture at a glance

Two Vultr Serverless Inference models, each doing the job it's actually
built for — plus Microsoft Azure as the secondary/optional data layer:

- **VultronRetriever** (Flash/Core/Prime, Qwen3.5-based, #1 on ViDoRe V3)
  — Sentinel's core evidence retrieval engine, via `/v1/rerank`. It
  reads clinical evidence the way a clinician does, with full layout
  awareness (tables, charts, scans), and reranks candidate evidence by
  relevance to the investigation.
- **A Vultr-hosted chat-completion model** — the reasoning engine behind
  planning, hypothesis generation, report synthesis, and review.
  VultronRetriever cannot do this job (it has no chat-completion
  capability, confirmed against Vultr's own docs); this is a standard
  `/v1/chat/completions` call on the same endpoint and API key.

```
Document Upload → Azure AI Document Intelligence (OCR + extraction, optional)
                        ↓
                Candidate evidence chunks (Azure AI Search or local fallback)
                        ↓
                VultronRetriever rerank (/v1/rerank) ← layout-aware retrieval
                        ↓
                Azure Cosmos DB (investigation state + evidence storage, optional)
                        ↓
          LangGraph Investigation Engine (orchestration)
       planner → retrieval → tool_agent → hypothesis → reporter → reviewer
                        ↓
      Vultr-hosted chat model (/v1/chat/completions) — reasoning
                        ↓
              anukriti-pgx-core (deterministic PGx tool)
```

Every cloud dependency — both Vultr models and all three Azure services
— is **optional at runtime**. Each one has a deterministic or local
fallback, so the full investigation pipeline runs end-to-end with zero
cloud credentials configured. See `docs/ARCHITECTURE.md` for the full
breakdown of what each service does and what its fallback is.

## Project layout

```
packages/
  agents/          Planner, Retrieval, Tool Agent, Hypothesis, Reporter, Reviewer
  tools/           pgx-core adapter, Document Intelligence adapter,
                   AI Search / evidence-candidate adapter,
                   VultronRetriever rerank adapter, tool registry
  database/        Cosmos DB client (+ local JSON fallback)
  schemas/         InvestigationState (LangGraph state channel)
  graph.py         Builds and runs the LangGraph investigation graph
  llm.py           Vultr Serverless Inference chat model (reasoning)
  config.py        Vultr rerank + Azure availability checks
apps/
  api/             FastAPI app: REST + WebSocket streaming
docs/
  ARCHITECTURE.md  Full multi-cloud, two-model architecture writeup
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in real values, or leave blank to run on fallbacks
```

## Running

Smoke-test the investigation graph directly (no API, no server):

```bash
python -m packages.graph
```

Runs both demo scenarios end-to-end: a confirmed root cause that gets
rejected for thin evidence and re-investigated to approval, and a
refusal (no genotype evidence) that gets approved on the first pass.

Run the API:

```bash
uvicorn apps.api.main:app --reload
```

Then:

```bash
# Health check — reports which backend (cloud vs. fallback) each subsystem is using
curl http://127.0.0.1:8000/api/health

# Start an investigation
curl -X POST http://127.0.0.1:8000/api/investigations \
  -H "Content-Type: application/json" \
  -d '{"case_id": "case-1", "incident": "Patient admitted after fluorouracil therapy. Symptoms: neutropenia, mucositis, diarrhea, fever."}'

# Poll for progress / the final report
curl http://127.0.0.1:8000/api/investigations/case-1

# Upload a clinical document (lab report, EHR note, FDA label PDF)
curl -X POST http://127.0.0.1:8000/api/investigations/case-1/upload -F "file=@lab_report.pdf"
```

Or connect to `ws://127.0.0.1:8000/api/investigations/case-1/stream` for
live, node-by-node progress as the graph runs instead of polling.

## Environment variables

See `.env.example` for the full list. Every credential is optional —
copy the file to `.env` and fill in only the services you want to run
against real cloud backends; leave the rest blank to use the
deterministic/local fallback for that subsystem.

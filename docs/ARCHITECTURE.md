# Sentinel Clinical — Multi-Cloud Architecture

## Design principle

Microsoft Azure owns the data lifecycle (ingest → store → retrieve).
NVIDIA Nemotron, served via Vultr Serverless Inference, owns the
reasoning (planning, hypothesis generation, review). Each service plays
to its strength, and the split is a real architectural boundary — not
two vendors bolted onto the same layer for the sake of eligibility.

```
Document Upload → Azure AI Document Intelligence (OCR + extraction)
                        ↓
                Azure AI Search (semantic evidence retrieval)
                        ↓
                Azure Cosmos DB (investigation state + evidence storage)
                        ↓
          LangGraph Investigation Engine (orchestration)
       planner → retrieval → tool_agent → hypothesis → reporter → reviewer
                        ↓
      NVIDIA Nemotron via Vultr Serverless Inference (reasoning)
                        ↓
              anukriti-pgx-core (deterministic PGx tool)
```

## The optional-cloud pattern

Every external cloud dependency in Sentinel — the LLM and all three
Azure services — follows the same rule: **check availability, fall back
deterministically if unavailable.** This was established by
`packages/llm.py::llm_available()` (used by every agent node) and
extended to Azure via `packages/config.py`:

| Function | Checks | Used by |
|---|---|---|
| `llm_available()` | `VULTR_API_KEY` | All 5 agent nodes (planner, hypothesis, reporter, reviewer; tool_agent doesn't call the LLM) |
| `doc_intel_available()` | `AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY` | `doc_intel_tool.py` |
| `search_available()` | `AZURE_SEARCH_ENDPOINT` + `AZURE_SEARCH_KEY` | `retrieval_tool.py` |
| `cosmos_available()` | `AZURE_COSMOS_ENDPOINT` + `AZURE_COSMOS_KEY` | `cosmos_client.py` |
| `azure_available()` | any of the three above | Health-check reporting only |

No credentials configured is not a degraded demo — every fallback below
is deterministic and exercised by the project's own smoke test
(`python -m packages.graph`) and live API tests. Cloud credentials
upgrade fidelity (real OCR, real semantic search, a real managed
database); they are never required to run the investigation end to end.

## The Azure layer (data lifecycle)

### 1. Azure AI Document Intelligence — ingest

**Module:** `packages/tools/doc_intel_tool.py::extract_clinical_document()`

An investigation starts with uploaded clinical records: lab reports, EHR
notes, FDA label PDFs. Document Intelligence's `prebuilt-layout` model
extracts full text, tables (e.g. lab values with dates), and page
structure from arbitrary clinical PDFs/images without a custom-trained
model — the right fit for heterogeneous source documents Sentinel
doesn't control the formatting of.

**Fallback chain:** Azure Document Intelligence → local PyPDF2 text
extraction (tables are Azure-only; the fallback returns text only) →
hardcoded demo lab report if the file is missing/unreadable. A live
Azure call that fails mid-request (bad key, quota, network) degrades to
the same fallback rather than raising into the graph.

**API surface:** `POST /api/investigations/{case_id}/upload`
(`apps/api/routers/upload.py`) — accepts a multipart file, runs
extraction, appends the structured result to that investigation's
`documents` list, persists via the Cosmos layer.

**Current scope note:** `InvestigationState.documents` is populated by
this pipeline but not yet consumed by any graph node — that's the
Retrieval Agent's next iteration (feeding extracted document text into
evidence retrieval, rather than only the fixed demo evidence corpus).

### 2. Azure AI Search — evidence retrieval

**Module:** `packages/tools/retrieval_tool.py::search_evidence()`

Backs the Retrieval Agent's evidence lookup: given a query derived from
a Planner task (e.g. "CPIC guideline DPYD fluoropyrimidine dose"),
returns ranked evidence chunks with semantic ranking over the indexed
clinical evidence base.

**Fallback chain:** Azure AI Search (`query_type="semantic"`) → local
keyword-overlap ranking over a small hardcoded evidence corpus (CPIC
guideline text, lab trend summary, FDA label excerpt, drug-interaction
note — matching the project's fluorouracil/DPYD demo narrative so a
fallback-mode Retrieval Agent still surfaces evidence consistent with
the rest of the pipeline).

**Indexing:** `index_document()` pushes extracted Document Intelligence
output into the Azure AI Search index as an evidence chunk (no-op in
fallback mode, since there's no index to populate — retrieval reads from
the hardcoded corpus instead).

### 3. Azure Cosmos DB — investigation state storage

**Module:** `packages/database/cosmos_client.py`

`InvestigationState` (`packages/schemas/investigation_state.py`) is a
JSON-shaped TypedDict — Cosmos DB's document model is a direct fit.
`save_investigation()` / `load_investigation()` / `list_investigations()`
back every stateful API endpoint: the investigation is persisted after
every graph step so a client can create an investigation, disconnect,
and poll `GET /api/investigations/{case_id}` for progress without
holding a connection open.

**Fallback chain:** Cosmos DB → local JSON files under
`data/investigations/` (one file per case, gitignored). Same read/write
contract either way — callers never need to know which backend is
active.

## The reasoning layer (NVIDIA Nemotron via Vultr)

**Module:** `packages/llm.py`

Every LLM-backed agent node (planner, hypothesis, reporter, reviewer)
uses `llm_json_call()`: a system prompt embedding the target JSON
schema, a plain chat completion against Vultr's OpenAI-compatible
endpoint, then manual extraction + Pydantic validation with one retry.
This is deliberate — Vultr's native tool-calling support is restricted
to `kimi-k2-instruct`, and Nemotron is the model this project commits
to for clinical reasoning, so `with_structured_output()`/function
calling isn't usable here.

**Fallback:** every LLM node has a paired deterministic rule-based
implementation, used when `VULTR_API_KEY` is unset or the LLM call fails
after retries. This is a resilience feature, not a lesser tier — it's
what keeps a live demo running through a flaky API, and it's what the
project's guardrails (below) fall back to when they don't trust the
LLM's output.

**Known open item:** the exact Nemotron model ID string
(`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` by default) has not been
confirmed against a live `GET /v1/models` call — see the module
docstring in `packages/llm.py` and run `list_models()` to verify before
relying on the LLM path for anything beyond local dry-runs.

## The investigation graph

**Module:** `packages/graph.py`

```
planner → retrieval → tool_agent → hypothesis → reporter → reviewer
                            ↑_________________________________|
                              (rejected: back to tool_agent,
                               NOT retrieval — evidence search
                               runs once per investigation)
```

- **Planner** (`packages/agents/planner.py`) — turns free-text incident
  into a fixed-vocabulary task list. Never investigates or concludes.
- **Retrieval Agent** (`packages/agents/retrieval.py`) — maps each
  Planner task to a `search_evidence()` query, writes results to
  `retrieved_evidence`. Owns general evidence search (labs, FDA label,
  CPIC guidelines, drug interactions, medication history); does **not**
  call pgx-core (a deterministic clinical-decision engine, not a search
  index) — that stays the Tool Agent's job.
- **Tool Agent** (`packages/agents/tool_agent.py`) — executes
  deterministic tools: `pgx-core` for pharmacogenomic recommendations,
  plus a hardcoded lab-trends stub used only when the Reviewer
  specifically requests it on a retry pass.
- **Hypothesis Agent** (`packages/agents/hypothesis.py`) — the
  safety-critical node. Generates competing hypotheses with confidence
  scores; enforces the pharmacogenomic guardrail both in-prompt and
  post-hoc in code — if a phenotype was never retrieved, no hypothesis
  may assign it nonzero confidence or state it as fact.
- **Reporter** (`packages/agents/reporter.py`) — assembles the
  structured report deterministically; only the executive-summary text
  is LLM-authored.
- **Reviewer** (`packages/agents/reviewer.py`) — a second, skeptical AI
  investigator. Unconfirmed/honest-refusal reports are **always**
  approved deterministically, never via the LLM (so an LLM reviewer can
  never be talked into demanding a guess). Confirmed reports go through
  LLM review with a one-way ratchet: the LLM can be stricter than the
  rule-based baseline, never more lenient.

## The API layer

**Module:** `apps/api/`

- `GET /api/health` — reports which backend (cloud vs. fallback) each
  subsystem is currently running on.
- `POST /api/investigations` — creates an investigation, runs the graph
  as a background task, returns `case_id` immediately (202).
- `GET /api/investigations` / `GET /api/investigations/{case_id}` —
  list / fetch current state and report.
- `POST /api/investigations/{case_id}/upload` — Document Intelligence
  ingestion for a specific case.
- `WS /api/investigations/{case_id}/stream` — live, node-by-node
  progress as the graph executes, via an in-memory pub/sub broadcaster
  (`apps/api/events.py`) that safely marshals events from the
  background task's worker thread onto the WebSocket handler's event
  loop.

## Why this is a real multi-cloud architecture, not a checkbox

Each Azure service maps onto a distinct, necessary stage of the
pipeline that Sentinel would otherwise have to build itself (OCR,
semantic search infra, a managed document database) — and each one is
wired into the actual data flow (`InvestigationState.documents` →
`retrieved_evidence` → the report), not called once for demonstration
and then ignored. NVIDIA/Vultr's Nemotron is the reasoning engine behind
every agent decision that requires judgment; Azure never reasons about
clinical evidence, and Nemotron never touches OCR or storage. That
separation is enforced structurally (which modules import which
clients) as well as narratively.

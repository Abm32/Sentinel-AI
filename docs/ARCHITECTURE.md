# Sentinel Clinical — Multi-Cloud Architecture

## Design principle

Two Vultr Serverless Inference models, each doing the job it's built
for, plus Microsoft Azure as an optional secondary data layer:

- **VultronRetriever** — the core evidence retrieval engine. A visual
  document retrieval model family (#1 on the ViDoRe V3 benchmark) that
  reads clinical evidence with layout awareness — tables, charts, scans
  — the way a clinician does, and reranks candidate evidence by
  relevance to the investigation. Exposed via Vultr Serverless
  Inference's `/v1/rerank` endpoint.
- **A Vultr-hosted chat-completion model** — the reasoning engine.
  Handles planning, hypothesis generation, report synthesis, and
  review: everything that requires generating text or judgment, which
  VultronRetriever cannot do (see "Why not VultronRetriever for
  reasoning" below).

Both models run on the same infrastructure — Vultr Serverless
Inference, same endpoint, same API key — which is what satisfies the
"VultronRetriever + Vultr Serverless Inference" requirement without
forcing a reranking model to pretend it's a reasoning model.

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

![Sentinel Clinical architecture diagram](./architecture-diagram.png)

## Why not VultronRetriever for reasoning

This is worth stating plainly, because an earlier draft of this
architecture pointed a chat client directly at a VultronRetriever model
ID, which does not work. Confirmed against Vultr's own documentation
(`docs.vultr.com/how-to-rank-documents-with-vultronretriever-on-vultr-
serverless-inference`, fetched 2026-07-04):

- VultronRetriever is exposed **only** via `POST /v1/rerank`. It takes a
  query and a list of documents (plain text, or page images as OpenAI-
  style `image_url` content parts) and returns a relevance score per
  document. That is its entire interface.
- It has **no chat-completion capability** and **no embeddings
  endpoint** (`/v1/embeddings` returns `404` for these models, per
  Vultr's own "Handle API Limits and Errors" section). It cannot
  generate text, cannot follow instructions, cannot produce JSON, and
  cannot plan or reason about anything. It can only score relevance.
- Vultr's own multimodal RAG pipeline example (in that same guide) pairs
  VultronRetriever with a *separate* chat model (`Qwen/Qwen3.6-27B` in
  their example) to generate the final answer — confirming that even
  Vultr's own reference implementation treats retrieval and reasoning
  as two distinct models, not one.

Building Sentinel's Planner/Hypothesis/Reporter/Reviewer nodes on
VultronRetriever would ship code that fails on every call. The
two-model split below is the correct reading of "use VultronRetriever
as the core [document-understanding] reasoning engine" — VultronRetriever
reasons over *documents* (which pages/passages matter), a separate chat
model reasons over the *investigation* (what they mean, what's missing,
whether the conclusion holds).

## The optional-cloud pattern

Every external dependency in Sentinel — both Vultr models and all three
Azure services — follows the same rule: **check availability, fall back
deterministically if unavailable.** Centralized in `packages/config.py`
and `packages/llm.py`:

| Function | Checks | Used by |
|---|---|---|
| `rerank_available()` | `VULTR_API_KEY` | `vultron_rerank_tool.py` |
| `llm_available()` | `VULTR_API_KEY` | All 4 chat-model agent nodes (planner, hypothesis, reporter, reviewer) |
| `doc_intel_available()` | `AZURE_DOC_INTEL_ENDPOINT` + `AZURE_DOC_INTEL_KEY` | `doc_intel_tool.py` |
| `search_available()` | `AZURE_SEARCH_ENDPOINT` + `AZURE_SEARCH_KEY` | `retrieval_tool.py` |
| `cosmos_available()` | `AZURE_COSMOS_ENDPOINT` + `AZURE_COSMOS_KEY` | `cosmos_client.py` |
| `azure_available()` | any of the three Azure checks above | Health-check reporting only |

`rerank_available()` and `llm_available()` both check the same
`VULTR_API_KEY` — deliberately: they're the same account, same
subscription, same endpoint, just different model IDs (`VULTR_RERANK_MODEL`
vs. `VULTR_CHAT_MODEL`). No credentials configured is not a degraded
demo — every fallback below is deterministic and exercised by the
project's own smoke test (`python -m packages.graph`) and live API
tests. Cloud credentials upgrade fidelity; they are never required to
run the investigation end to end.

## The retrieval layer (VultronRetriever + Azure, secondary)

### VultronRetriever — evidence reranking

**Module:** `packages/tools/vultron_rerank_tool.py::rerank_evidence()`

Given a query (derived from a Planner task, e.g. "CPIC guideline DPYD
fluoropyrimidine dose") and a set of candidate evidence chunks,
VultronRetriever scores and reorders them by relevance. This is the
precision pass in Sentinel's two-stage retrieval (see below) — it
catches evidence a keyword match would miss or misrank, because it
reads with layout/context awareness rather than pure token overlap.

**Model tiers** (confirmed real model ID strings, Vultr docs fetched
2026-07-04):

| Model ID | Size | Use when |
|---|---|---|
| `vultr/VultronRetrieverFlash-Qwen3.5-0.8B` | 0.8B | Default — fast, cost-efficient, strong quality |
| `vultr/VultronRetrieverCore-Qwen3.5-4.5B` | 4.5B | Denser documents |
| `vultr/VultronRetrieverPrime-Qwen3.5-8B` | 8B | Maximum quality on hard/cluttered documents |

Configurable via `VULTR_RERANK_MODEL`; defaults to Flash.

**Fallback:** no `VULTR_API_KEY` configured, or the live call fails ⇒
pass-through — candidates are returned in their original order, tagged
`reranked_by: "none (pass-through fallback)"` so it's visible in the
data itself that reranking was skipped, not silently assumed.

**Current scope note:** this module reranks *extracted text* chunks.
VultronRetriever's primary designed use case — scoring relevance
directly from rendered *page images* (catching a lab values table or a
chart that text extraction misses entirely) — is a natural next
iteration once the Retrieval Agent has access to rendered page images
rather than only Document-Intelligence-extracted text; the request
shape in `vultron_rerank_tool.py` already supports both.

### Two-stage retrieval

**Module:** `packages/agents/retrieval.py::retrieval_node()`

1. **Candidate generation** (`retrieval_tool.py::search_evidence()`) —
   a cheap, broad lookup over the evidence base: Azure AI Search if
   configured, otherwise a local keyword-overlap match over a small
   hardcoded evidence corpus. Casts a wide net (5 candidates per task).
2. **Reranking** (`vultron_rerank_tool.py::rerank_evidence()`) —
   VultronRetriever re-scores those candidates against the
   investigation-specific query and reorders them. Top 3 are kept per
   task.

Azure AI Search's role in this pipeline is explicitly secondary: it
supplies *candidates* for VultronRetriever to rerank. VultronRetriever
is what Sentinel calls its core retrieval engine, per the hackathon's
requirement — Azure AI Search is a fallback-friendly source of
candidates upstream of it, not a competing retrieval path.

### Azure AI Document Intelligence — ingest (still valid, secondary)

**Module:** `packages/tools/doc_intel_tool.py::extract_clinical_document()`

Unchanged from before: OCR + structured extraction (text, tables) from
uploaded clinical PDFs/images via Azure's `prebuilt-layout` model,
falling back to local PyPDF2 text extraction or a hardcoded demo
document. Feeds `InvestigationState.documents`, and is the upstream
source of the text chunks that `search_evidence()` and, eventually,
VultronRetriever's page-image path would consume.

### Azure Cosmos DB — investigation state storage (still valid, secondary)

**Module:** `packages/database/cosmos_client.py`

Unchanged: `InvestigationState`'s JSON shape maps directly onto Cosmos
DB's document model. `save_investigation()` / `load_investigation()` /
`list_investigations()` persist state after every graph step, falling
back to local JSON files under `data/investigations/` (gitignored).

## The reasoning layer (Vultr-hosted chat model)

**Module:** `packages/llm.py`

Every reasoning-requiring agent node (planner, hypothesis, reporter,
reviewer) uses `llm_json_call()`: a system prompt embedding the target
JSON schema, a plain chat completion against Vultr's OpenAI-compatible
`/v1/chat/completions` endpoint, then manual extraction + Pydantic
validation with one retry. This is deliberate — tool-calling support on
Vultr Serverless Inference is restricted to specific models (confirmed:
`kimi-k2-instruct`), so `with_structured_output()`/function calling
can't be assumed to work regardless of which chat model is configured;
prompt-based JSON works against any OpenAI-compatible endpoint.

**Fallback:** every LLM node has a paired deterministic rule-based
implementation, used when `VULTR_API_KEY` is unset or the LLM call fails
after retries. This is a resilience feature, not a lesser tier — it's
what keeps a live demo running through a flaky API, and it's what the
project's guardrails fall back to when they don't trust the LLM's
output (see `hypothesis.py`'s pharmacogenomic guardrail).

**Known open item:** the exact chat-model ID string
(`moonshotai/kimi-k2-instruct` by default) has not been confirmed
against a live `GET /v1/models` call under your specific subscription
— see `packages/llm.py`'s module docstring and run `list_models()` /
`print_model_catalog()` to verify before relying on the LLM path for
anything beyond local dry-runs.

## The investigation graph

**Module:** `packages/graph.py`

```
planner → retrieval → tool_agent → hypothesis → reporter → reviewer
                            ↑_________________________________|
                              (rejected: back to tool_agent,
                               NOT retrieval — evidence search
                               runs once per investigation)
```

- **Planner** (chat model) — turns free-text incident into a
  fixed-vocabulary task list. Never investigates or concludes.
- **Retrieval Agent** (VultronRetriever) — maps each Planner task to a
  candidate-search + rerank pass, writes results to `retrieved_evidence`.
  Owns general evidence search; does **not** call pgx-core (a
  deterministic clinical-decision engine, not a document to rerank).
- **Tool Agent** (deterministic) — executes `pgx-core` for
  pharmacogenomic recommendations, plus a hardcoded lab-trends stub
  used only when the Reviewer specifically requests it on a retry pass.
- **Hypothesis Agent** (chat model) — the safety-critical node.
  Generates competing hypotheses with confidence scores; enforces the
  pharmacogenomic guardrail both in-prompt and post-hoc in code.
- **Reporter** (chat model for the executive summary only; everything
  else deterministic extraction) — assembles the structured report.
- **Reviewer** (chat model for confirmed reports; unconfirmed/honest-
  refusal reports are **always** approved deterministically, never via
  the LLM) — a second, skeptical AI investigator, with a one-way
  ratchet: the LLM can be stricter than the rule-based baseline, never
  more lenient.

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
  progress as the graph executes.

## Why this satisfies the hackathon requirement

VultronRetriever is Sentinel's core retrieval engine — every general
evidence-search task in the investigation is reranked through it, and
that reranking demonstrably changes which evidence reaches the
Hypothesis Agent. The reasoning layer (planning, hypothesis generation,
report synthesis, review) runs on a separate, chat-capable model — this
is explicitly allowed by the hackathon's own rules ("optionally use
other agents or models to facilitate chat, UI interactions, or
secondary tasks"), and is the only architecture that actually works,
since VultronRetriever has no chat-completion capability to build
reasoning on top of. Both models run on Vultr Serverless Inference:
same endpoint, same API key, same infrastructure. Azure's three
services remain genuinely useful — OCR/extraction, candidate generation
upstream of reranking, and state storage — but are explicitly secondary
to the Vultr-hosted retrieval-plus-reasoning core, and every one of them
is optional at runtime.

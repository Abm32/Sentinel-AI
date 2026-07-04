"""
Investigation state storage: Azure Cosmos DB, with a local-JSON fallback.

`InvestigationState` (packages/schemas/investigation_state.py) is a
JSON-shaped TypedDict — a natural fit for Cosmos DB's document model.
This module persists it so the API layer (apps/api) can create an
investigation, kick off the graph asynchronously, and let a client poll
or reconnect to fetch progress/results without holding the graph run in
memory.

Same fallback pattern as every other cloud dependency in this project
(see `packages/config.py::cosmos_available`): if Cosmos DB credentials
are not configured, or a live call fails, this falls back to local JSON
files under `data/investigations/` — one file per case, keyed by
case_id. The demo must work with zero cloud credentials.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from packages.config import cosmos_available

_DATABASE_NAME = "sentinel-clinical"
_CONTAINER_NAME = "investigations"

# Local fallback storage root. Relative to the project root so it works
# regardless of cwd when the API/graph is invoked from different entry
# points.
_LOCAL_STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "investigations"


def _local_path(case_id: str) -> Path:
    # case_id is expected to be a simple slug (e.g. "demo-case-A"); guard
    # against path traversal regardless.
    safe_id = "".join(c for c in case_id if c.isalnum() or c in ("-", "_"))
    return _LOCAL_STORAGE_DIR / f"{safe_id}.json"


def _local_save(state: dict[str, Any]) -> None:
    _LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    path = _local_path(state["case_id"])
    path.write_text(json.dumps(state, indent=2, default=str))


def _local_load(case_id: str) -> dict[str, Any] | None:
    path = _local_path(case_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _local_list() -> list[dict[str, Any]]:
    if not _LOCAL_STORAGE_DIR.exists():
        return []
    results = []
    for path in sorted(_LOCAL_STORAGE_DIR.glob("*.json")):
        try:
            results.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return results


def _get_cosmos_container():
    from azure.cosmos import CosmosClient

    client = CosmosClient(
        os.getenv("AZURE_COSMOS_ENDPOINT"),
        credential=os.getenv("AZURE_COSMOS_KEY"),
    )
    database = client.get_database_client(_DATABASE_NAME)
    return database.get_container_client(_CONTAINER_NAME)


def save_investigation(state: dict[str, Any]) -> None:
    """
    Persist an InvestigationState (or any JSON-serializable partial/full
    state dict containing `case_id`) as a document keyed by `case_id`.

    Tries Cosmos DB first if credentials are configured; falls back to a
    local JSON file otherwise, or if the live call fails.
    """
    if cosmos_available():
        try:
            container = _get_cosmos_container()
            container.upsert_item({"id": state["case_id"], **state})
            return
        except Exception:
            pass
    _local_save(state)


def load_investigation(case_id: str) -> dict[str, Any] | None:
    """
    Load a previously saved investigation by case_id. Returns None if not
    found (in either backend).
    """
    if cosmos_available():
        try:
            container = _get_cosmos_container()
            return container.read_item(item=case_id, partition_key=case_id)
        except Exception:
            pass
    return _local_load(case_id)


def list_investigations() -> list[dict[str, Any]]:
    """
    List all saved investigations. Used by the API's `GET
    /api/investigations` endpoint.
    """
    if cosmos_available():
        try:
            container = _get_cosmos_container()
            return list(container.read_all_items())
        except Exception:
            pass
    return _local_list()

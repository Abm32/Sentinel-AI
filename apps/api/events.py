"""
In-memory event broadcaster for live investigation progress.

The investigation graph runs inside a FastAPI BackgroundTask (see
investigations.py::_execute_investigation) — a plain `def` (sync)
function. FastAPI/Starlette runs sync background tasks in a threadpool
thread, NOT on the main asyncio event loop thread. The WebSocket
streaming endpoint (routers/investigations.py::stream_investigation),
by contrast, runs on the event loop.

That split matters: `asyncio.Queue` is only safe to use from within a
single event loop. Calling `queue.put_nowait()` directly from the
background task's threadpool thread is a race — it must be scheduled
onto the loop via `loop.call_soon_threadsafe()` instead. This module
captures the running loop at subscribe time and uses it to marshal
`publish`/`finish` calls back onto the loop thread safely.

Deliberately in-process, in-memory, no external broker — this is a
single-process FastAPI app (uvicorn, no multi-worker deployment target
for the hackathon demo). If Sentinel ever runs behind multiple worker
processes, this would need to move to a shared broker (Redis pub/sub,
etc.) since queues and the captured loop reference are per-process.
Documented explicitly rather than silently breaking in that scenario.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

# One (loop, queue) pair per subscriber. Keyed by case_id; multiple
# subscribers (e.g. two browser tabs) can watch the same case_id.
_subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = defaultdict(list)

# Sentinel value pushed to a queue to signal "no more events" — lets a
# subscriber's consumption loop terminate instead of blocking forever.
# Public (not underscore-prefixed): callers outside this module (the
# WebSocket handler) need to compare against it directly.
DONE = object()


def subscribe(case_id: str) -> asyncio.Queue:
    """Register a new subscriber queue for `case_id`. MUST be called
    from within the event loop that will later consume the queue (i.e.
    from the WebSocket handler's async context) — the loop is captured
    here so `publish`/`finish` can marshal events back onto it safely
    from the background task's separate thread. Call `unsubscribe` when
    the WebSocket disconnects to avoid leaking entries for clients that
    dropped mid-stream."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _subscribers[case_id].append((loop, queue))
    return queue


def unsubscribe(case_id: str, queue: asyncio.Queue) -> None:
    subscribers = _subscribers.get(case_id)
    if not subscribers:
        return
    _subscribers[case_id] = [(loop, q) for loop, q in subscribers if q is not queue]
    if not _subscribers[case_id]:
        _subscribers.pop(case_id, None)


def publish(case_id: str, node_name: str, state_update: dict) -> None:
    """Called from the background task (a threadpool thread, NOT the
    event loop — see module docstring) after each graph step. Fans the
    event out to every current subscriber for this case_id via
    `call_soon_threadsafe`, so it's safe to call from any thread. Safe
    to call with zero subscribers (e.g. no one connected to the
    WebSocket yet) — the event is simply dropped, since GET polling
    remains the source of truth for state, not this channel."""
    for loop, queue in list(_subscribers.get(case_id, [])):
        loop.call_soon_threadsafe(queue.put_nowait, (node_name, state_update))


def finish(case_id: str) -> None:
    """Called from the background task once the graph run has ended
    (successfully or not). Signals every current subscriber to stop
    waiting for more events."""
    for loop, queue in list(_subscribers.get(case_id, [])):
        loop.call_soon_threadsafe(queue.put_nowait, DONE)

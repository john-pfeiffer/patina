"""Tiny pub/sub for Server-Sent Events. Workers publish from threads; each SSE
client drains its own queue from the async side."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Event:
    name: str
    data: dict[str, Any]

    def encode(self) -> str:
        return f"event: {self.name}\ndata: {json.dumps(self.data, separators=(',', ':'))}\n\n"


class EventBus:
    def __init__(self, maxsize: int = 1000):
        self._subs: set[queue.Queue[Event]] = set()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def subscribe(self) -> queue.Queue[Event]:
        q: queue.Queue[Event] = queue.Queue(maxsize=self._maxsize)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[Event]) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, name: str, data: dict[str, Any]) -> None:
        ev = Event(name, data)
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                # A client that stopped reading loses events rather than
                # stalling the worker; it will re-sync by refetching.
                pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subs)

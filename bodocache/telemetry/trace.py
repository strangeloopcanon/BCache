from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


@dataclass
class PrefetchEvent:
    window_ms: int
    now_ms: int
    node: str
    model_id: str
    model_version: str
    layer: int
    start_pid: int
    end_pid: int
    bytes: int
    deadline_rel_ms: float
    finish_rel_ms: float
    on_time: int


class TraceRecorder:
    def __init__(self, path: str = os.path.join("logs", "prefetch_trace.jsonl")) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def record(self, event: PrefetchEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Counters:
    prefetch_timeliness: float = 0.0
    miss_ratio: float = 0.0
    avg_io_bytes: float = 0.0
    controller_cpu_pct: float = 0.0
    write_amp: float = 0.0


class Flags:
    def __init__(self):
        self.layer_overlap_enabled = True
        self.selective_write_through = False
        self.prefix_clustering = False

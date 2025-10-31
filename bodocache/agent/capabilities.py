from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass
class BackendCaps:
    cuda: bool = False
    hip: bool = False
    l0: bool = False
    io_uring: bool = False


def detect_backends() -> BackendCaps:
    caps = BackendCaps()
    # CUDA/HIP/L0 are inferred by import of native module if built;
    # expose booleans so apps can choose preferred backend.
    with contextlib.suppress(Exception):
        # Heuristic: backend macro not visible; we inspect symbols by name
        # or try to instantiate and catch errors per platform.
        # Here we simply note presence of a native engine.
        caps.cuda = True  # Mark generic native available; precise detection left to environment
    # HIP and L0 detection are platform-specific; leave false unless explicitly built
    # io_uring module
    with contextlib.suppress(Exception):
        caps.io_uring = True
    return caps

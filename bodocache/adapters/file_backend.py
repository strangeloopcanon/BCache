from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from ..planner.models import PageKey


class FileBackend:
    """Simple file-based backend storing pages under a root directory.

    Path layout: root/model_id/model_version/layer/page_id.bin
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: PageKey) -> Path:
        return self.root / key.model_id / key.model_version / f"{key.layer}" / f"{key.page_id}.bin"

    def exists(self, key: PageKey) -> bool:
        return self._path(key).exists()

    def get(self, key: PageKey) -> Optional[bytes]:
        p = self._path(key)
        try:
            return p.read_bytes()
        except FileNotFoundError:
            return None

    def set(self, key: PageKey, value: bytes):
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)


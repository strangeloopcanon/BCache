from __future__ import annotations

from typing import Any, Dict, List
import time

from bodocache.integrations.base import KVRequest
from bodocache.integrations.sglang_adapter import SGLangBCacheAdapter
from bodocache.integrations.sglang_glue import SGLangHook
from bodocache.integrations.ptr import from_torch_tensor


class SGLangRequestBuilder:
    def __init__(self, page_bytes: int = 256 * 1024, tenant: str = "default") -> None:
        self.page_bytes = page_bytes
        self.tenant = tenant

    def build_requests(self, sglang_state: Any) -> List[KVRequest]:
        requests: List[KVRequest] = []
        now_ms = int(time.time() * 1000)
        for seq in getattr(sglang_state, "active_sequences", []):
            prefix_id = getattr(seq, "prefix_id", str(seq))
            for layer in range(getattr(sglang_state, "num_layers", 0)):
                page_start, page_end = self._lookup_pages(seq, layer)
                if page_start <= page_end:
                    requests.append(
                        KVRequest(
                            req_id=f"{seq}:{layer}",
                            node=getattr(sglang_state, "node", "n0"),
                            model_id=getattr(sglang_state, "model_id", "m"),
                            model_version=getattr(sglang_state, "model_version", "v"),
                            prefix_id=prefix_id,
                            layer=int(layer),
                            page_start=int(page_start),
                            page_end=int(page_end),
                            page_bytes=self.page_bytes,
                            tenant=self.tenant,
                            est_fill_ms=1.0,
                            tier_src=0,
                            tier_dst=2,
                            deadline_ms=now_ms + 20,
                        )
                    )
        return requests

    def _lookup_pages(self, seq: Any, layer: int) -> tuple[int, int]:
        # TODO: implement using your KV page table
        return (0, -1)


def make_dest_resolver(kv_manager: Any):
    def dest_resolver(info: Dict[str, Any]):
        layer = info["layer"]
        start_pid = info["start_pid"]
        end_pid = info["end_pid"]
        # tensor = kv_manager.get_tensor_slice(layer, start_pid, end_pid)
        tensor = None
        if tensor is None:
            return None
        return from_torch_tensor(tensor)

    return dest_resolver


def make_hook(adapter: SGLangBCacheAdapter, builder: SGLangRequestBuilder, kv_manager: Any) -> SGLangHook:
    return SGLangHook(
        adapter,
        build_requests=builder.build_requests,
        dest_resolver=make_dest_resolver(kv_manager),
    )


from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from bodocache.planner.api import (
    LayerLatency,
    PlannerRequest,
    PlannerWindow,
    TenantCapacity,
    TierCapacity,
)


class KVRequest(BaseModel):
    """A single KV page interval request originating from an engine.

    All fields map directly to columns expected by the planner scheduler.
    """

    req_id: str
    node: str
    model_id: str
    model_version: str
    prefix_id: str
    layer: int
    page_start: int
    page_end: int
    page_bytes: int = Field(default=256 * 1024, ge=0)
    tenant: str = "default"
    est_fill_ms: float = Field(default=1.0, ge=0.0)
    tier_src: int = 0  # storage
    tier_dst: int = 2  # gpu
    deadline_ms: int = 0
    prefix_tokens: list[int] | None = None
    route_hint: str | None = None
    pcluster: int | None = None

    @field_validator("layer", "page_start", "page_end", "tier_src", "tier_dst")
    @classmethod
    def _non_negative(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("numeric fields must be non-negative")
        return int(value)

    def as_planner_request(self) -> PlannerRequest:
        return PlannerRequest(
            req_id=self.req_id,
            node=self.node,
            model_id=self.model_id,
            model_version=self.model_version,
            prefix_id=self.prefix_id,
            layer=self.layer,
            page_start=self.page_start,
            page_end=self.page_end,
            page_bytes=self.page_bytes,
            tenant=self.tenant,
            est_fill_ms=self.est_fill_ms,
            tier_src=self.tier_src,
            tier_dst=self.tier_dst,
            deadline_ms=self.deadline_ms,
            prefix_tokens=self.prefix_tokens,
            route_hint=self.route_hint,
            pcluster=self.pcluster,
        )


class PlannerInputs(BaseModel):
    """Planner side inputs for a single planning window."""

    requests: list[KVRequest]
    window_ms: int = 20
    now_ms: int = 0
    # Per-tier capacities: bytes per window and free bytes; indices by tier id
    bandwidth_caps: dict[int, int] | None = None
    free_bytes: dict[int, int] | None = None
    # Per-tenant bandwidth caps (bytes per window) per tier
    tenant_caps: list[tuple[str, int, int]] | None = None  # (tenant, tier, cap)
    # Per-layer latencies (ms)
    layer_lat_ms: dict[int, float] | None = None

    def _default_bandwidth_caps(self) -> dict[int, int]:
        return {0: 60 * 1024 * 1024, 1: 200 * 1024 * 1024, 2: 500 * 1024 * 1024}

    def _tier_caps(self) -> list[TierCapacity]:
        bw_caps = self.bandwidth_caps or self._default_bandwidth_caps()
        free = self.free_bytes or {0: 1 << 60, 1: 1 << 60, 2: 1 << 60}
        tiers = sorted(set((0, 1, 2)) | set(bw_caps.keys()) | set(free.keys()))
        return [
            TierCapacity(
                tier=int(tier),
                bandwidth_caps=int(bw_caps.get(tier, 0)),
                free_bytes=int(free.get(tier, 1 << 60)),
            )
            for tier in tiers
        ]

    def _tenant_caps(self) -> list[TenantCapacity]:
        if self.tenant_caps:
            return [
                TenantCapacity(tenant=str(t), tier=int(tier), bandwidth_caps=int(cap))
                for (t, tier, cap) in self.tenant_caps
            ]
        tenants = {req.tenant for req in self.requests}
        return [
            TenantCapacity(tenant=tenant, tier=tier, bandwidth_caps=1 << 60)
            for tenant in tenants
            for tier in (0, 1, 2)
        ]

    def _layer_latencies(self) -> list[LayerLatency]:
        if self.layer_lat_ms:
            return [
                LayerLatency(layer=int(layer), lat_ms=float(value))
                for layer, value in sorted(self.layer_lat_ms.items())
            ]
        layers = sorted({req.layer for req in self.requests})
        return [LayerLatency(layer=layer, lat_ms=1.0) for layer in layers]

    def to_planner_window(self) -> PlannerWindow:
        planner_requests = [req.as_planner_request() for req in self.requests]
        tier_caps = self._tier_caps()
        tenant_caps = self._tenant_caps()
        layer_lat = self._layer_latencies()
        return PlannerWindow(
            requests=planner_requests,
            now_ms=int(self.now_ms),
            tier_caps=tier_caps,
            tenant_caps=tenant_caps,
            layer_latencies=layer_lat,
        )


def build_dataframes(pi: PlannerInputs):
    """Construct the DataFrames required by scheduler.run_window from inputs."""
    window = pi.to_planner_window()
    return window.to_dataframes()

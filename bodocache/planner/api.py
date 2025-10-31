from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from bodocache.planner import scheduler
from bodocache.planner.models import DEFAULT_PAGE_BYTES


class PlannerRequest(BaseModel):
    req_id: str
    node: str
    model_id: str
    model_version: str
    prefix_id: str
    layer: int
    page_start: int
    page_end: int
    tier_src: int
    tier_dst: int
    deadline_ms: int
    page_bytes: int = Field(default=DEFAULT_PAGE_BYTES, ge=0)
    tenant: str = "default"
    est_fill_ms: float = Field(default=1.0, ge=0.0)
    pcluster: int | None = None
    prefix_tokens: list[int] | None = None
    route_hint: str | None = None

    @field_validator("layer", "page_start", "page_end", "tier_src", "tier_dst")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("integer fields must be non-negative")
        return int(value)

    @field_validator("prefix_tokens")
    @classmethod
    def _normalize_tokens(cls, value: Sequence[int] | None) -> list[int] | None:
        if value is None:
            return None
        return [int(v) for v in value]

    @model_validator(mode="after")
    def _validate_interval(self) -> PlannerRequest:
        if int(self.page_end) < int(self.page_start):
            raise ValueError("page_end must be >= page_start")
        return self


class HeatEntry(BaseModel):
    layer: int
    page_id: int
    decay_hits: int = 1
    tenant_weight: float = 1.0
    size_bytes: int = DEFAULT_PAGE_BYTES

    @field_validator("layer", "page_id")
    @classmethod
    def _positive(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("layer and page_id must be non-negative")
        return int(value)


class TierCapacity(BaseModel):
    tier: int
    bandwidth_caps: int
    free_bytes: int


class TenantCapacity(BaseModel):
    tenant: str
    tier: int
    bandwidth_caps: int


class LayerLatency(BaseModel):
    layer: int
    lat_ms: float = 1.0


class PlannerConfig(BaseModel):
    pmin: float = 1.0
    umin: float = 0.0
    min_io_bytes: int = 512 * 1024
    alpha: float = 1.0
    beta: float = 0.0
    window_ms: int = 20
    max_ops_per_tier: int = 64
    enable_admission: bool = True
    enable_eviction: bool = True
    enforce_tier_caps: bool = True


class PlannerWindow(BaseModel):
    requests: list[PlannerRequest]
    now_ms: int
    heat: list[HeatEntry] = Field(default_factory=list)
    tier_caps: list[TierCapacity] = Field(default_factory=list)
    tenant_caps: list[TenantCapacity] = Field(default_factory=list)
    layer_latencies: list[LayerLatency] = Field(default_factory=list)

    @field_validator("now_ms")
    @classmethod
    def _now_non_negative(cls, value: int) -> int:
        if int(value) < 0:
            raise ValueError("now_ms must be non-negative")
        return int(value)

    def _requests_dataframe(self) -> pd.DataFrame:
        if not self.requests:
            return pd.DataFrame(
                columns=[
                    "req_id",
                    "node",
                    "model_id",
                    "model_version",
                    "prefix_id",
                    "layer",
                    "page_start",
                    "page_end",
                    "tier_src",
                    "tier_dst",
                    "deadline_ms",
                    "page_bytes",
                    "tenant",
                    "est_fill_ms",
                    "pcluster",
                    "prefix_tokens",
                    "route_hint",
                ]
            )
        frame = pd.DataFrame([req.model_dump() for req in self.requests])
        frame["layer"] = frame["layer"].astype(np.int64)
        frame["page_start"] = frame["page_start"].astype(np.int64)
        frame["page_end"] = frame["page_end"].astype(np.int64)
        frame["page_bytes"] = frame["page_bytes"].astype(np.int64)
        frame["tier_src"] = frame["tier_src"].astype(np.int64)
        frame["tier_dst"] = frame["tier_dst"].astype(np.int64)
        frame["deadline_ms"] = frame["deadline_ms"].astype(np.int64)
        frame["est_fill_ms"] = frame["est_fill_ms"].astype(float)
        if "pcluster" in frame.columns:
            if frame["pcluster"].isna().all():
                frame = frame.drop(columns=["pcluster"])
            else:
                frame["pcluster"] = frame["pcluster"].astype("Int64").fillna(-1).astype(np.int64)
        return frame

    def _heat_dataframe(self, page_bytes_default: int = DEFAULT_PAGE_BYTES) -> pd.DataFrame:
        if self.heat:
            data = [entry.model_dump() for entry in self.heat]
            frame = pd.DataFrame(data)
        else:
            req_df = self._requests_dataframe()
            frame = req_df[["layer", "page_start", "page_bytes"]].rename(
                columns={"page_start": "page_id", "page_bytes": "size_bytes"}
            )
            frame["decay_hits"] = np.int64(1)
            frame["tenant_weight"] = np.float64(1.0)
        frame["size_bytes"] = frame.get("size_bytes", page_bytes_default).astype(np.int64)
        return frame.groupby(["layer", "page_id"], as_index=False).agg(
            decay_hits=("decay_hits", "sum"),
            tenant_weight=("tenant_weight", "first"),
            size_bytes=("size_bytes", "max"),
        )

    def _tier_caps_dataframe(self) -> pd.DataFrame:
        if self.tier_caps:
            entries = [cap.model_dump() for cap in self.tier_caps]
        else:
            entries = [
                {"tier": 0, "bandwidth_caps": 60 * 1024 * 1024, "free_bytes": 1 << 60},
                {"tier": 1, "bandwidth_caps": 200 * 1024 * 1024, "free_bytes": 1 << 60},
                {"tier": 2, "bandwidth_caps": 500 * 1024 * 1024, "free_bytes": 1 << 60},
            ]
        frame = pd.DataFrame(entries)
        frame["tier"] = frame["tier"].astype(np.int64)
        frame["bandwidth_caps"] = frame["bandwidth_caps"].astype(np.int64)
        frame["free_bytes"] = frame["free_bytes"].astype(np.int64)
        return frame

    def _tenant_caps_dataframe(self) -> pd.DataFrame:
        if self.tenant_caps:
            entries = [cap.model_dump() for cap in self.tenant_caps]
        else:
            tenants = {req.tenant for req in self.requests}
            entries = [
                {"tenant": tenant, "tier": tier, "bandwidth_caps": 1 << 60}
                for tenant in tenants
                for tier in (0, 1, 2)
            ]
        frame = pd.DataFrame(entries)
        if frame.empty:
            return pd.DataFrame(columns=["tenant", "tier", "bandwidth_caps"])
        frame["tier"] = frame["tier"].astype(np.int64)
        frame["bandwidth_caps"] = frame["bandwidth_caps"].astype(np.int64)
        return frame

    def _layer_latency_dataframe(self) -> pd.DataFrame:
        if self.layer_latencies:
            entries = [lat.model_dump() for lat in self.layer_latencies]
        else:
            layers = sorted({req.layer for req in self.requests})
            entries = [{"layer": layer, "lat_ms": 1.0} for layer in layers]
        frame = pd.DataFrame(entries)
        if frame.empty:
            return pd.DataFrame(columns=["layer", "lat_ms"])
        frame["layer"] = frame["layer"].astype(np.int64)
        frame["lat_ms"] = frame["lat_ms"].astype(float)
        return frame

    def to_dataframes(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        req_df = self._requests_dataframe()
        heat_df = self._heat_dataframe()
        tier_caps_df = self._tier_caps_dataframe()
        tenant_caps_df = self._tenant_caps_dataframe()
        layer_lat_df = self._layer_latency_dataframe()
        return req_df, heat_df, tier_caps_df, tenant_caps_df, layer_lat_df


class PlanOp(BaseModel):
    node: str
    tier_src: int
    tier_dst: int
    pcluster: int
    layer: int
    run_id: int
    bytes: int
    deadline_ms: int
    fanout: int | None = None
    overlap: int | None = None
    priority: float | None = None
    start_pid: int | None = None
    end_pid: int | None = None
    page_bytes: int | None = None


class EvictionEntry(BaseModel):
    layer: int
    page_id: int


class AdmissionEntry(BaseModel):
    layer: int
    page_id: int
    tier_dst: int


class PlannerResult(BaseModel):
    plan: list[PlanOp] = Field(default_factory=list)
    evictions: list[EvictionEntry] = Field(default_factory=list)
    admissions: list[AdmissionEntry] = Field(default_factory=list)

    @classmethod
    def from_dataframes(
        cls,
        plan_df: pd.DataFrame,
        evict_df: pd.DataFrame,
        admission_df: pd.DataFrame,
    ) -> PlannerResult:
        plan = []
        if not plan_df.empty:
            plan = [PlanOp(**row) for row in plan_df.to_dict(orient="records")]
        evictions = []
        if not evict_df.empty:
            evictions = [EvictionEntry(**row) for row in evict_df.to_dict(orient="records")]
        admissions = []
        if not admission_df.empty:
            admissions = [AdmissionEntry(**row) for row in admission_df.to_dict(orient="records")]
        return cls(plan=plan, evictions=evictions, admissions=admissions)

    def as_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        plan_df = pd.DataFrame([op.model_dump() for op in self.plan])
        evict_df = pd.DataFrame([row.model_dump() for row in self.evictions])
        admission_df = pd.DataFrame([row.model_dump() for row in self.admissions])
        for frame in (plan_df, evict_df, admission_df):
            if not frame.empty:
                frame.reset_index(drop=True, inplace=True)
        return plan_df, evict_df, admission_df


def plan_window(window: PlannerWindow, config: PlannerConfig | None = None) -> PlannerResult:
    cfg = config or PlannerConfig()
    req_df, heat_df, tier_caps_df, tenant_caps_df, layer_lat_df = window.to_dataframes()
    if req_df.empty:
        return PlannerResult()
    plan_df, evict_df, admission_df = scheduler.run_window(
        req_df,
        heat_df,
        tier_caps_df,
        tenant_caps_df,
        layer_lat_df,
        now_ms=window.now_ms,
        pmin=cfg.pmin,
        umin=cfg.umin,
        min_io_bytes=cfg.min_io_bytes,
        alpha=cfg.alpha,
        beta=cfg.beta,
        window_ms=cfg.window_ms,
        max_ops_per_tier=cfg.max_ops_per_tier,
        enable_admission=cfg.enable_admission,
        enable_eviction=cfg.enable_eviction,
        enforce_tier_caps=cfg.enforce_tier_caps,
    )
    return PlannerResult.from_dataframes(plan_df, evict_df, admission_df)

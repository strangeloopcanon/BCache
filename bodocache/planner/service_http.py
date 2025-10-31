from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pandas as pd

from bodocache.planner.api import (
    HeatEntry,
    LayerLatency,
    PlannerConfig,
    PlannerRequest,
    PlannerWindow,
    TenantCapacity,
    TierCapacity,
    plan_window,
)
from bodocache.planner.models import DEFAULT_PAGE_BYTES


def df_from_json(obj: Any, columns: list[str]) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(obj, columns=columns)


def plan_from_payload(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req = df_from_json(
        payload.get("requests"),
        [
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
            "prefix_tokens",
            "pcluster",
        ],
    )
    heat = df_from_json(
        payload.get("heat"), ["layer", "page_id", "decay_hits", "tenant_weight", "size_bytes"]
    )
    tiers = df_from_json(payload.get("tier_caps"), ["tier", "bandwidth_caps", "free_bytes"])
    tenant_caps = df_from_json(payload.get("tenant_caps"), ["tenant", "tier", "bandwidth_caps"])
    lats = df_from_json(payload.get("layer_lat"), ["layer", "lat_ms"])
    now_ms = int(payload.get("now_ms", 0))
    knobs = payload.get("knobs", {})

    requests_payload = [
        PlannerRequest(
            req_id=str(row["req_id"]),
            node=str(row["node"]),
            model_id=str(row["model_id"]),
            model_version=str(row["model_version"]),
            prefix_id=str(row["prefix_id"]),
            layer=int(row["layer"]),
            page_start=int(row["page_start"]),
            page_end=int(row["page_end"]),
            tier_src=int(row["tier_src"]),
            tier_dst=int(row["tier_dst"]),
            deadline_ms=int(row["deadline_ms"]),
            page_bytes=int(row["page_bytes"]),
            tenant=str(row["tenant"]),
            est_fill_ms=float(row["est_fill_ms"]),
            prefix_tokens=list(row.get("prefix_tokens", []) or []),
            pcluster=int(row.get("pcluster", -1)),
        )
        for row in req.to_dict(orient="records")
    ]
    heat_entries = [
        HeatEntry(
            layer=int(row["layer"]),
            page_id=int(row["page_id"]),
            decay_hits=int(row["decay_hits"]),
            tenant_weight=float(row["tenant_weight"]),
            size_bytes=int(row.get("size_bytes", DEFAULT_PAGE_BYTES)),
        )
        for row in heat.to_dict(orient="records")
    ]
    tier_cap_entries = [
        TierCapacity(
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
            free_bytes=int(row["free_bytes"]),
        )
        for row in tiers.to_dict(orient="records")
    ]
    tenant_cap_entries = [
        TenantCapacity(
            tenant=str(row["tenant"]),
            tier=int(row["tier"]),
            bandwidth_caps=int(row["bandwidth_caps"]),
        )
        for row in tenant_caps.to_dict(orient="records")
    ]
    layer_lat_entries = [
        LayerLatency(layer=int(row["layer"]), lat_ms=float(row["lat_ms"]))
        for row in lats.to_dict(orient="records")
    ]
    window = PlannerWindow(
        requests=requests_payload,
        now_ms=now_ms,
        heat=heat_entries,
        tier_caps=tier_cap_entries,
        tenant_caps=tenant_cap_entries,
        layer_latencies=layer_lat_entries,
    )
    config = PlannerConfig(
        pmin=float(knobs.get("pmin", 1.0)),
        umin=float(knobs.get("umin", 0.0)),
        min_io_bytes=int(knobs.get("min_io_bytes", 512 * 1024)),
        alpha=float(knobs.get("alpha", 1.0)),
        beta=float(knobs.get("beta", 0.0)),
        window_ms=int(knobs.get("window_ms", 20)),
        max_ops_per_tier=int(knobs.get("max_ops_per_tier", 64)),
        enable_admission=bool(knobs.get("enable_admission", True)),
        enable_eviction=bool(knobs.get("enable_eviction", True)),
        enforce_tier_caps=bool(knobs.get("enforce_tier_caps", True)),
    )
    result = plan_window(window, config)
    return result.as_dataframes()


class PlannerHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: dict[str, Any]):
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self._send(400, {"error": f"invalid json: {e}"})
            return
        if self.path == "/get_plan":
            try:
                plan_df, evict_df, admission_df = plan_from_payload(payload)
                body = {
                    "plan": plan_df.to_dict(orient="records"),
                    "evict": evict_df.to_dict(orient="records"),
                    "admission": admission_df.to_dict(orient="records"),
                }
                self._send(200, body)
            except Exception as e:
                self._send(500, {"error": str(e)})
        elif self.path == "/report":
            # Accept perf counters and acknowledge
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})


def serve(host: str = "0.0.0.0", port: int = 8080):  # nosec B104
    httpd = HTTPServer((host, port), PlannerHandler)
    httpd.serve_forever()

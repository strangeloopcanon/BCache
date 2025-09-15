from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Tuple

import pandas as pd

from bodocache.planner.scheduler import run_window


def df_from_json(obj: Any, columns: list[str]) -> pd.DataFrame:
    if obj is None:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(obj, columns=columns)


def plan_from_payload(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    req = df_from_json(payload.get("requests"), [
        "req_id","node","model_id","model_version","prefix_id","layer","page_start","page_end","tier_src","tier_dst","deadline_ms","page_bytes","tenant","est_fill_ms"
    ])
    heat = df_from_json(payload.get("heat"), ["layer","page_id","decay_hits","tenant_weight"])
    tiers = df_from_json(payload.get("tier_caps"), ["tier","bandwidth_caps","free_bytes"])
    tenant_caps = df_from_json(payload.get("tenant_caps"), ["tenant","tier","bandwidth_caps"])
    lats = df_from_json(payload.get("layer_lat"), ["layer","lat_ms"])
    now_ms = int(payload.get("now_ms", 0))
    knobs = payload.get("knobs", {})

    plan_df, evict_df, admission_df = run_window(
        req, heat, tiers, tenant_caps, lats, now_ms,
        pmin=float(knobs.get("pmin", 1.0)),
        umin=float(knobs.get("umin", 0.0)),
        min_io_bytes=int(knobs.get("min_io_bytes", 512*1024)),
        alpha=float(knobs.get("alpha", 1.0)),
        beta=float(knobs.get("beta", 0.0)),
        window_ms=int(knobs.get("window_ms", 20)),
        max_ops_per_tier=int(knobs.get("max_ops_per_tier", 64)),
        enable_admission=bool(knobs.get("enable_admission", True)),
        enable_eviction=bool(knobs.get("enable_eviction", True)),
        enforce_tier_caps=bool(knobs.get("enforce_tier_caps", True)),
    )
    return plan_df, evict_df, admission_df


class PlannerHandler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: Dict[str, Any]):
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


def serve(host: str = "0.0.0.0", port: int = 8080):
    httpd = HTTPServer((host, port), PlannerHandler)
    httpd.serve_forever()


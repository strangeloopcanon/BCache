from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Any

import yaml


DEFAULTS = {
    'window_ms': 20,
    'min_io_bytes': 524288,
    'max_ops_per_tier': 64,
    'thresholds': {'pmin': 1.0, 'umin': 0.0},
    'popularity': {'alpha': 1.0, 'beta': 0.0},
    'ab_flags': {
        'enable_prefix_fanout': True,
        'enable_tenant_credits': True,
        'enable_admission': True,
        'enable_eviction': True,
        'enable_overlap': True,
        'enforce_tier_caps': True,
    },
    'tenant_credits_bytes': 32 * 1024 * 1024,
}


def load_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def merge_dict(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def load_config(runtime_path: str = 'configs/runtime.yaml', staged_path: str | None = None) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if staged_path and os.path.exists(staged_path):
        cfg = merge_dict(cfg, load_yaml(staged_path))
    cfg = merge_dict(cfg, load_yaml(runtime_path))
    return cfg


# Typed config wrappers for ergonomics and safety
@dataclass
class Thresholds:
    pmin: float = 1.0
    umin: float = 0.0


@dataclass
class Popularity:
    alpha: float = 1.0
    beta: float = 0.0


@dataclass
class ABFlags:
    enable_prefix_fanout: bool = True
    enable_tenant_credits: bool = True
    enable_admission: bool = True
    enable_eviction: bool = True
    enable_overlap: bool = True
    enforce_tier_caps: bool = True


@dataclass
class RuntimeConfig:
    window_ms: int = 20
    min_io_bytes: int = 524288
    max_ops_per_tier: int = 64
    thresholds: Thresholds = field(default_factory=Thresholds)
    popularity: Popularity = field(default_factory=Popularity)
    ab_flags: ABFlags = field(default_factory=ABFlags)
    tenant_credits_bytes: int = 32 * 1024 * 1024


def _get(d: Dict[str, Any], key: str, default):
    return d.get(key, default)


def load_config_typed(runtime_path: str = 'configs/runtime.yaml', staged_path: str | None = None) -> RuntimeConfig:
    raw = load_config(runtime_path=runtime_path, staged_path=staged_path)
    thresholds = Thresholds(
        pmin=float(_get(_get(raw, 'thresholds', {}), 'pmin', 1.0)),
        umin=float(_get(_get(raw, 'thresholds', {}), 'umin', 0.0)),
    )
    popularity = Popularity(
        alpha=float(_get(_get(raw, 'popularity', {}), 'alpha', 1.0)),
        beta=float(_get(_get(raw, 'popularity', {}), 'beta', 0.0)),
    )
    ab_flags = ABFlags(
        enable_prefix_fanout=bool(_get(_get(raw, 'ab_flags', {}), 'enable_prefix_fanout', True)),
        enable_tenant_credits=bool(_get(_get(raw, 'ab_flags', {}), 'enable_tenant_credits', True)),
        enable_admission=bool(_get(_get(raw, 'ab_flags', {}), 'enable_admission', True)),
        enable_eviction=bool(_get(_get(raw, 'ab_flags', {}), 'enable_eviction', True)),
        enable_overlap=bool(_get(_get(raw, 'ab_flags', {}), 'enable_overlap', True)),
        enforce_tier_caps=bool(_get(_get(raw, 'ab_flags', {}), 'enforce_tier_caps', True)),
    )
    return RuntimeConfig(
        window_ms=int(_get(raw, 'window_ms', 20)),
        min_io_bytes=int(_get(raw, 'min_io_bytes', 524288)),
        max_ops_per_tier=int(_get(raw, 'max_ops_per_tier', 64)),
        thresholds=thresholds,
        popularity=popularity,
        ab_flags=ab_flags,
        tenant_credits_bytes=int(_get(raw, 'tenant_credits_bytes', 32 * 1024 * 1024)),
    )

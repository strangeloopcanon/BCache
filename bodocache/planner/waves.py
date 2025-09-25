from __future__ import annotations

from typing import List, Tuple, TypedDict, Optional, NamedTuple, Sequence

import math
import pandas as pd


class TMemLayout(TypedDict):
    columns: int
    phases: int
    double_buffer: bool
    stage_n: int


class WaveSpec(TypedDict):
    """Runtime contract for a single execution wave."""

    pack_order: List[int]
    tile_order: List[Tuple[int, int]]
    bm: int
    bn: int
    bk: int
    cluster_shape: Tuple[int, int]
    tmem_layout: TMemLayout
    io_extents: List[Tuple[str, int, int]]
    swap_window: Tuple[int, int]


class TileConfig(NamedTuple):
    bm: int
    bn: int
    bk: int
    stage: int
    cluster: Tuple[int, int]


# A curated whitelist of tiles mirrored by bw-runtime. Update in lock-step with runtime.
DEFAULT_TILE_CONFIGS: Tuple[TileConfig, ...] = (
    TileConfig(128, 128, 64, 2, (2, 1)),
    TileConfig(128, 256, 64, 2, (2, 1)),
    TileConfig(256, 128, 64, 2, (2, 1)),
    TileConfig(128, 128, 128, 3, (2, 1)),
)


def _resolve_whitelist(whitelist: Optional[Sequence[TileConfig]]) -> Tuple[TileConfig, ...]:
    if whitelist and len(whitelist):
        return tuple(whitelist)
    return DEFAULT_TILE_CONFIGS


def _dtype_bytes(dtype: str) -> int:
    d = dtype.lower()
    if d in ("float16", "fp16", "bfloat16", "bf16"):
        return 2
    if d in ("float32", "fp32"):
        return 4
    # default conservative
    return 2


def _ensure_tile_whitelist(
    dtype: str,
    configs: Sequence[TileConfig],
) -> TileConfig:
    """Return the first tile config that satisfies dtype granularity requirements."""

    if not configs:
        raise ValueError("Tile whitelist is empty; cannot select WaveSpec shape.")

    bpe = _dtype_bytes(dtype)
    for cfg in configs:
        if (cfg.bk * bpe) % 32 == 0:
            return cfg

    raise ValueError(
        f"No tile config satisfies 32B K granularity for dtype={dtype};"
        " provide a whitelist with compliant bk values."
    )


def _select_tile_config(
    dtype: str,
    *,
    shapes: Optional[List[Tuple[int, int, int]]] = None,
    whitelist: Optional[Sequence[TileConfig]] = None,
    default_cluster: Tuple[int, int] = (2, 1),
    default_stage: int = 2,
) -> TileConfig:
    """Pick a tile configuration for the wave respecting the whitelist contract."""

    if whitelist and len(whitelist):
        return _ensure_tile_whitelist(dtype, whitelist)

    if shapes:
        converted = tuple(
            TileConfig(int(bm), int(bn), int(bk), int(default_stage), (int(default_cluster[0]), int(default_cluster[1])))
            for bm, bn, bk in shapes
        )
        return _ensure_tile_whitelist(dtype, converted)

    return _ensure_tile_whitelist(dtype, DEFAULT_TILE_CONFIGS)


def validate_wave_spec(
    spec: WaveSpec,
    *,
    dtype: str = "float16",
    whitelist: Optional[Sequence[TileConfig]] = None,
) -> None:
    """Raise ValueError if the wave spec violates the runtime contract."""

    required = [
        "pack_order",
        "tile_order",
        "bm",
        "bn",
        "bk",
        "cluster_shape",
        "tmem_layout",
        "io_extents",
        "swap_window",
    ]
    for field in required:
        if field not in spec:
            raise ValueError(f"WaveSpec missing required field '{field}'")

    whitelist_lookup = _resolve_whitelist(whitelist)
    bm = int(spec["bm"])
    bn = int(spec["bn"])
    bk = int(spec["bk"])
    cx, cy = (int(spec["cluster_shape"][0]), int(spec["cluster_shape"][1]))

    layout = spec["tmem_layout"]
    for key in ("columns", "phases", "double_buffer", "stage_n"):
        if key not in layout:
            raise ValueError(f"tmem_layout missing '{key}' field")
    stage_n = int(layout.get("stage_n", -1))

    matched = False
    for cfg in whitelist_lookup:
        if (
            bm == cfg.bm
            and bn == cfg.bn
            and bk == cfg.bk
            and cx == cfg.cluster[0]
            and cy == cfg.cluster[1]
            and stage_n == cfg.stage
        ):
            matched = True
            break

    if not matched:
        raise ValueError(
            "WaveSpec shape/cluster not in whitelist; ensure planner uses shared tile list"
        )

    bpe = _dtype_bytes(dtype)
    if (bk * bpe) % 32 != 0:
        raise ValueError(
            f"WaveSpec bk={spec['bk']} fails tensor core granularity for dtype={dtype}"
        )

    sb, se = spec["swap_window"]
    if int(sb) < 0 or int(se) <= int(sb):
        raise ValueError(
            f"Invalid swap window ({sb}, {se}); must satisfy 0 <= begin < end"
        )


def _build_swizzle(rows: int, cols: int) -> List[Tuple[int, int]]:
    """Simple 2D swizzle: row-major with odd-row reversal (snake)."""
    order: List[Tuple[int, int]] = []
    for r in range(rows):
        if r % 2 == 0:
            for c in range(cols):
                order.append((r, c))
        else:
            for c in reversed(range(cols)):
                order.append((r, c))
    return order


def _factorize_req_ids(req_col: pd.Series) -> List[int]:
    try:
        return [int(x) for x in req_col.tolist()]
    except Exception:
        codes, _ = pd.factorize(req_col.astype(str), sort=False)
        return [int(x) for x in codes.tolist()]


def build_wave_specs(
    plan_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    *,
    window_ms: int,
    dtype: str = "float16",
    shapes: Optional[List[Tuple[int, int, int]]] = None,
    default_cluster: Tuple[int, int] = (2, 1),
    tile_configs: Optional[Sequence[TileConfig]] = None,
) -> List[WaveSpec]:
    """Derive one WaveSpec per (node, tier_dst) group from a plan.

    - pack_order favors grouping by pcluster then earliest deadlines
    - tile_order encodes a snake swizzle over a tile grid sized to op count
    - bm/bn/bk (plus stage_hint/cluster) are selected from a whitelist satisfying 32B K granularity
    - io_extents reflect the coalesced ranges from the plan
    - swap_window guards weight swaps to the post-prefetch region of the wave
    """
    if plan_df is None or len(plan_df) == 0:
        return []

    cfg = _select_tile_config(
        dtype,
        shapes=shapes,
        whitelist=tile_configs,
        default_cluster=default_cluster,
    )
    tmem_layout: TMemLayout = {
        "columns": 8,
        "phases": 4,
        "double_buffer": True,
        "stage_n": int(cfg.stage),
    }
    waves: List[WaveSpec] = []
    by = [c for c in ["node", "tier_dst"] if c in plan_df.columns]
    for _, g in (plan_df.groupby(by, sort=False) if by else [(None, plan_df)]):
        # io_extents: one per planned coalesced op
        io_extents: List[Tuple[str, int, int]] = []
        for r in g.itertuples(index=False):
            layer = int(getattr(r, "layer", 0))
            start_pid = int(getattr(r, "start_pid", 0))
            end_pid = int(getattr(r, "end_pid", -1))
            if end_pid >= start_pid:
                io_extents.append((str(layer), start_pid, end_pid))

        # pack_order: use request ids, grouped by pcluster when available
        if not requests_df.empty:
            req = requests_df.copy()
            # Limit to layers participating in this group to keep relevance
            if "layer" in req.columns and "layer" in g.columns:
                layers = set(int(x) for x in g["layer"].unique().tolist())
                req = req[req["layer"].isin(layers)]
            sort_cols = [c for c in ["pcluster", "deadline_ms"] if c in req.columns]
            if sort_cols:
                ascending = [True] * len(sort_cols)
                req = req.sort_values(by=sort_cols, ascending=ascending)
            pack_order = _factorize_req_ids(req.get("req_id", pd.Series(range(len(req)))))
        else:
            pack_order = []

        # tile grid sized to op count (sqrt heuristic)
        tiles = max(1, len(io_extents))
        rows = int(math.floor(math.sqrt(tiles)))
        cols = int(math.ceil(tiles / max(1, rows)))
        # ensure full coverage
        if rows * cols < tiles:
            rows = max(rows, 1)
            while rows * cols < tiles:
                cols += 1
        tile_order = _build_swizzle(rows, cols)[:tiles]

        prefetch_tokens = int(len(io_extents))
        compute_tiles = int(max(1, len(tile_order)))
        swap_window = (prefetch_tokens, prefetch_tokens + compute_tiles)

        wave: WaveSpec = {
            "pack_order": pack_order,
            "tile_order": tile_order,
            "bm": int(cfg.bm),
            "bn": int(cfg.bn),
            "bk": int(cfg.bk),
            "cluster_shape": (int(cfg.cluster[0]), int(cfg.cluster[1])),
            "tmem_layout": dict(tmem_layout),
            "io_extents": io_extents,
            "swap_window": swap_window,
        }
        validate_wave_spec(wave, dtype=dtype, whitelist=tile_configs)
        waves.append(wave)
    return waves

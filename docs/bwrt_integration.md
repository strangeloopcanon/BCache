# BCache ↔ bw-runtime Integration Handoff

**Summary**

Goal: Let BCache emit a WaveSpec that bw-runtime executes via a narrow ABI, while hotweights coordinates swap-safe weight updates. Keep BCache’s planner and data plane; bw-runtime provides a GPU-ready device runtime with CPU fallback for now.

This handoff document captures the actionable asks, adapter code, proto schema, tests, and acceptance criteria that mirror the upstream bw-runtime change set (https://github.com/strangeloopcanon/bw-runtime).

---

## Planner Requirements

1. **Enforce K granularity**  
   Only emit shapes where K meets tensor-core granularity (e.g., BF16/FP16 → `K * dtype_bytes % 32 == 0`).

2. **Tile/cluster whitelist**  
   Use a tuned shortlist of `(BM, BN, BK, StageN, cluster)` entries. bw-runtime mirrors this list and asserts on it.

3. **Swap windows**  
   Emit `swap_begin ≤ token < swap_end` aligned to wave boundaries. Avoid swap windows that intersect the heavy TMA portion of a wave.

4. **Swizzle metadata**  
   Preferred: emit a concrete traversal via `tile_order_flat` (flattened `(m, n)` tile pairs).  
   Fallback: provide a coarse swizzle knob (e.g., `swizzle_group_size`) so runtime can synthesize a traversal.

5. **Planned field additions (next pass)**  
   - dtype and tensor strides (TMA descriptor hints)  
   - tail policy for partial tiles (pad / drop / peel)  
   - stage hint override (runtime may downgrade if unsafe)

---

## WaveSpec Contract (MVP)

Fields to emit **now**:

- `pack_order`: request ids ordered for L2 reuse.
- `tile_order`: list of `(tile_m, tile_n)` pairs defining thread-block swizzle.
- `bm`, `bn`, `bk`: tile shape satisfying tensor-core granularity.
- `cluster_shape`: `(x, y)` CTA cluster.
- `tmem_layout`: dict with `{columns, phases, double_buffer, stage_n}`.
- `io_extents`: list of `(layer:str, start_pid:int, end_pid:int)` ranges.
- `swap_window`: `(begin:int, end:int)` tokens where swap is safe.

### Reference Proto (`proto/bwrt/wavespec.proto`)

```proto
syntax = "proto3";
package bwrt;

message TileCoord {
  int32 m = 1;
  int32 n = 2;
}

message ClusterShape {
  int32 x = 1;
  int32 y = 2;
}

message TMemLayout {
  int32 columns = 1;
  int32 phases = 2;
  bool double_buffer = 3;
  int32 stage_n = 4;
}

message IoExtent {
  string layer = 1;
  int32 start_pid = 2;
  int32 end_pid = 3;
}

message WaveSpec {
  repeated int32 pack_order = 1;
  repeated TileCoord tile_order = 2;

  int32 bm = 3;
  int32 bn = 4;
  int32 bk = 5;

  ClusterShape cluster_shape = 6;
  TMemLayout tmem_layout = 7;
  repeated IoExtent io_extents = 8;

  int32 swap_begin = 9; // inclusive
  int32 swap_end   = 10; // exclusive
}
```

To regenerate Python stubs:

```bash
python -m pip install protobuf grpcio-tools
python -m grpc_tools.protoc -I proto --python_out=. proto/bwrt/wavespec.proto
```

---

## Runtime Adapter (Python)

`bodocache/adapters/bwrt_adapter.py` provides the production adapter. It:

- Prefers the pybind11 module (`bwrt._bwrt`) when present, else falls back to the ctypes binding.
- Normalises `tile_order` pairs, `cluster_shape`, and `swap_window` into the layout expected by each runtime flavour (adding legacy fields like `swap_begin`/`swap_end` for ctypes).
- Accepts raw device pointers or NumPy/CuPy/Torch arrays via `_ptr_from_obj`.
- Exposes `submit_and_wait` and `set_weights`, mirroring the runtime contract.

---

## Validation & Instrumentation

- Consume runtime metrics on each wave (or every N waves) to tune planner policies:  
  - low `wgmma_active` → consider larger `BM/BN`  
  - low `tma_occ` → increase prefetch/coalesced extents  
  - low `l2_hit` → widen swizzle group or emit richer `tile_order_flat`
- Keep ring occupancy moderate (~30–60%) to avoid TMA backpressure starving WGMMA.
- Share whitelist updates so runtime can assert on identical sets.

---

## Minimal Tests To Add

1. **Valid WaveSpec executes** (CPU mode): given `{bm,bn,bk,cluster_x/y,swap_begin/end}` with valid K + whitelist shape → `submit_and_wait` succeeds and produces correct `C` matrix.
2. **Swap window enforcement:** `set_weights` succeeds inside window; fails outside after a tight window is emitted.
3. **Invalid K rejected:** planner rejects / raises on non-granular K and falls back to a valid shape.

---

## Acceptance Criteria

- Valid WaveSpecs pass validation; invalid ones fail fast with descriptive errors.
- Swap windows enforced by bw-runtime; planner emits windows aligned to wave boundaries.
- Metrics are ingested and logged for planner feedback (mocked or real).
- Only whitelisted shapes are used (planner asserts; runtime asserts).

---

## Notes

- The runtime package exposes both a ctypes wrapper and an optional pybind11 module. The adapter prefers pybind11 automatically when available.
- Environment variables:
  - `BWRT_LIB_PATH` → shared library search path when ctypes is used.
- Planned follow-up request will extend the contract with dtype/stride metadata once device bring-up begins.

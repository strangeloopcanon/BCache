BCache v0.1.0 — Initial GitHub Release

Highlights
- Bodo-powered planner with pure-Python fallback and clear, modular pipeline.
- Multistream simulator with overlap and priority hints; segmented file backend; NodeAgent executor.
- MinHash-based prefix clustering with blake3→blake2b fallback.
- Telemetry logger that appends CSV snapshots per window.

Important Notes
- Python import namespace remains `bodocache` (e.g., `from bodocache.planner.scheduler import run_window`).
- Project brand is “BCache”. CLI entrypoints: `bcache-sim` (preferred) and `bodocache-sim` (alias).
- Bodo JIT can impose compile-time overhead/memory on first runs. Use pure-Python mode when desired.

Install
```bash
pip install -e .
```

Run the Simulator
```bash
# With Bodo JIT (default)
bcache-sim

# Pure-Python mode (lighter, no JIT)
BODOCACHE_PURE_PY=1 bcache-sim
```

CLI Examples
```bash
# Increase min I/O size and disable tier caps
bcache-sim --min-io 1048576 --no-enforce-tier-caps

# Adjust thresholds/weights
bcache-sim --pmin 0.5 --umin -1.0 --alpha 1.0 --beta 0.0

# Toggle features off
bcache-sim --disable-prefix-fanout --disable-tenant-credits \
           --disable-admission --disable-eviction --disable-overlap
```

Testing
```bash
# Fast path (pure Python)
BODOCACHE_PURE_PY=1 pytest -q

# Bodo JIT path (ensure sufficient memory)
pytest -q
```

Known Caveats
- Some DataFrame APIs fall back to Pandas under Bodo; this is expected and may log warnings.
- The data plane is a Python simulator; production should use a C++/CUDA backend.

Validation Snapshot
- Full test suite with JIT: 11 passed (first-run JIT adds latency)
- End-to-end sim (JIT): ops≈27, avg_io≈1.9MB, prefetch_timeliness≈0.85, NodeAgent bytes≈51.5MB


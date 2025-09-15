# BCache: A Bodo-Powered Hierarchical KV Cache Planner

## Motivation

Modern Large Language Models (LLMs) require a large Key-Value (KV) cache to handle long contexts, such as detailed documents or long-running conversations. When this cache exceeds the GPU's limited memory, it must be paged to slower CPU memory or storage, creating a significant I/O bottleneck. As a result, the GPU often sits idle, waiting for data to be transferred.

BCache is a prototype designed to solve this problem by treating KV cache management as a logistics challenge. It provides an intelligent, high-performance control plane that orchestrates data movement efficiently, maximizing hardware utilization and reducing latency.

## The BCache Approach

BCache's architecture separates the planning logic ("the brain") from the underlying data transfer mechanism ("the brawn").

*   **The Planner (The Brain):** The core of BCache is a sophisticated planner, written in high-level Python with Pandas. It looks at all pending requests holistically to create an optimal data transfer plan. It does this by scoring requests by popularity and urgency, identifying opportunities to group requests for similar data (prefix clustering), and merging many small transfers into large, efficient I/O operations (coalescing).

*   **The Bodo Advantage:** Normally, a planner this complex would be too slow if written in Python. BCache solves this by using **Bodo**, a platform that JIT-compiles the high-level Python and Pandas code into highly-optimized, parallel native code. This provides the performance of low-level C++ with the development speed and readability of Python.

*   **The Data Plane (The Brawn):** The planner's decisions are executed by a separate data plane. This component is intentionally simple and decoupled, responsible only for executing the copy operations given to it. In this prototype, it is a Python-based simulator, but it is designed to be replaced by a high-performance C++/CUDA implementation for production use.

## Where BCache Shines: Large-Scale Inference

BCache is designed for workloads that involve many forward passes with a frozen or slowly changing model. It is **not** intended for standard model training where weights change on every step.

The ideal use cases are:

*   **RLHF/RLAIF Rollouts:** An actor model is used to generate millions of responses for a reward model to score. This is a massive-scale inference task with significant opportunity for prefix caching.
*   **LLM-based Evaluation:** Running a fixed model over a very large evaluation dataset to compute metrics.
*   **Teacher Models for Distillation:** A large, fixed "teacher" model is used to generate training data for a smaller "student" model, which involves countless forward passes.
*   **Synthetic Data Generation:** Using a model to generate large amounts of structured data.
*   **Multi-tenant, High-Throughput Serving:** In environments with many users and long-context requests, BCache can optimize I/O and enforce resource limits across tenants.

In these scenarios, the KV cache is stable, and the benefits of a smart, Bodo-powered planner that can orchestrate I/O across a large cluster are enormous.

### Why These Scenarios Excel with BCache

These use cases represent the "sweet spot" where BCache's mathematical optimization delivers transformational results:

**🎯 Shared Context Patterns**: In RLHF rollouts and evaluation campaigns, requests often share significant portions of their context (system prompts, conversation history, document chunks). BCache's MinHash clustering identifies these semantic similarities, turning what appears as "different" requests into cacheable opportunities.

**⚡ Predictable Access Patterns**: Teacher models and synthetic data generation create systematic memory access patterns. BCache learns these patterns and can prefetch with high accuracy, eliminating the "cold start" penalty that plagues traditional caches.

**🏗️ Natural Data Hierarchy**: Multi-tenant serving creates clear "hot," "warm," and "cold" data tiers. BCache's multi-tier optimization ensures frequently accessed KV pages stay in fast memory while moving cold data to cheaper storage.

**📊 Batch Processing Advantages**: Unlike real-time chat where every millisecond matters, these scenarios can leverage BCache's 20ms planning windows to make sophisticated optimization decisions that wouldn't be possible in ultra-low-latency environments.

### The Mathematical Advantage

Traditional caching treats KV cache as simple key-value pairs. BCache understands that in LLM inference:

1. **Semantic Similarity ≠ Exact Match**: Two prompts can share most of their KV cache while being "different" requests
2. **Access Patterns Are Learnable**: Stable models create predictable memory access sequences
3. **Fairness Can Be Quantified**: Multi-tenant resource allocation is a solvable optimization problem
4. **I/O Can Be Orchestrated**: Thousands of small random reads can become large sequential transfers

This transforms caching from a reactive "store what was used" to a proactive "predict what will be needed" system.

## Performance Characteristics

**Before BCache**: GPU utilization often drops to 30-40% as models wait for KV cache transfers between storage tiers
**After BCache**: Mathematical optimization keeps GPUs fed with data, maintaining high utilization

**Key Optimizations**:
- **I/O Coalescing**: Transforms many small random reads into fewer, larger sequential transfers
- **Semantic Clustering**: Identifies shared context across requests that traditional caches miss
- **Predictive Prefetching**: Learns access patterns to preload data before it's needed
- **Fair Resource Allocation**: Mathematical guarantees for multi-tenant environments

**Planning Performance**: Complex optimization decisions made in ~20ms using Bodo JIT compilation

## Features

*   **Bodo-JIT Compiled Planner:** The core planning logic is compiled with Bodo for maximum performance.
*   **Readable Planner Pipeline:** The core planning logic is broken into a clear, four-stage pipeline for better readability and maintenance.
*   **I/O Coalescing:** The planner identifies and merges contiguous page requests into large, efficient I/O operations.
*   **Popularity and Urgency Scoring:** The planner uses a scoring system to prioritize requests based on their importance and deadline.
*   **Advanced Prefix Clustering:** Uses MinHash LSH to group requests with semantically similar prefixes, enabling more efficient I/O coalescing.
*   **Tenant-based Credit System:** Allocates resources based on tenant-specific policies.
*   **Automated Policy Tuner:** Includes a `replay_tuner.py` script to automatically sweep through policy parameters and find the optimal configuration for a given workload.
*   **Realistic Performance Simulation:** The agent simulator models multiple, parallel copy streams and accounts for planner-provided overlap hints.
*   **Pluggable Storage Backends:** The storage backend can be easily replaced to support different storage systems.
*   **Pure Python Fallback:** The planner can run in a pure Python mode if Bodo is not available.

## Quick Start

### Install (editable)

BCache is currently distributed as source. Install in editable mode:

```bash
pip install -e .
```

### Prerequisites

- Python 3.11+
- Pandas, NumPy, PyYAML (installed by `pip install -e .`)
- Optional: Bodo for JIT-compiled planner paths (pure-Python mode works without it). See [Bodo vs Pure-Python Mode](#bodo-vs-pure-python-mode).
- Optional: `blake3` for faster hashing (falls back to `blake2b`).

### Run the Simulation

```bash
bcache-sim  # or: bodocache-sim
```

This will run a simulation of the planner with a synthetic workload. You should see output similar to this, indicating that the planner is creating coalesced I/O plans:

```
Plan summary:
  ops=18 avg_io=2275.6KB total=40.00MB
  mean_fanout=1.00 max_fanout=1
  node  tier_src  tier_dst  pcluster  layer  run_id   bytes   deadline_ms  fanout  overlap  priority
node-0         0         1        12      1       1  524288 1757826505314       1        1       8.8
node-0         0         1        31      1       1 4194304 1757826505334       1        1      32.0
...
```

### Experimenting with the Planner

You can easily experiment with the planner's behavior by modifying the configuration files:

*   `configs/runtime.yaml`: Controls the simulation parameters and planner knobs like `min_io_bytes` and `enforce_tier_caps`.
*   `configs/policies.yaml`: Contains detailed policy settings like popularity (`pmin`), urgency (`umin`), and scoring weights (`alpha`, `beta`).

Change a value in one of these files and re-run the simulation to see how it affects the plan.

### Bodo vs Pure-Python Mode

- With Bodo JIT (default): ensure Bodo is installed and licensed. First run may compile and use more memory/CPU.
- Pure-Python mode (lighter for dev/tests): set `BODOCACHE_PURE_PY=1` to bypass JIT and use the Python implementations.

Examples:

```bash
# Pure-Python run of the simulator (no JIT)
BODOCACHE_PURE_PY=1 bcache-sim

# Run with Bodo JIT (unset the flag)
bcache-sim
```

### CLI Overrides Examples

Adjust planner knobs at runtime using CLI flags:

```bash
# Increase min I/O size and disable tier-cap enforcement
bcache-sim --min-io 1048576 --no-enforce-tier-caps

# Tune popularity/urgency thresholds
bcache-sim --pmin 0.5 --umin -1.0 --alpha 1.0 --beta 0.0

# Toggle features
bcache-sim --disable-prefix-fanout --disable-tenant-credits \
              --disable-admission --disable-eviction --disable-overlap

Note: The Python package namespace remains `bodocache` for imports (e.g., `from bodocache.planner.scheduler import run_window`). The project name and CLI brand are “BCache”.
```

For more advanced tuning, you can use the replay tuner to find the optimal policy parameters for a given workload:

```bash
python -m scripts.replay_tuner
```

This will sweep through various parameter combinations and print the best-performing configurations, which it also saves to `configs/staged.yaml`.

### Testing

This project uses `pytest` for testing. To avoid heavy JIT during tests on dev/CI, prefer the pure-Python mode:

```bash
pip install pytest
BODOCACHE_PURE_PY=1 pytest -q
```

To run tests with Bodo JIT, unset `BODOCACHE_PURE_PY` and ensure sufficient memory.

## Project Layout

```
├── bodocache/
│   ├── planner/      # The Bodo-compiled planner and policy logic.
│   │   ├── scheduler.py  # Main planner entrypoint and Bodo-JIT core.
│   │   └── pipeline.py   # Readable, pure-Python implementation of the planner stages.
│   ├── agent/        # The Node Agent (Python, with native CUDA/HIP/L0 backends).
│   └── adapters/     # Pluggable storage backends.
├── configs/          # Configuration files for policies and runtime.
├── proto/            # gRPC service definitions for planner-agent communication.
├── scripts/          # Simulation and utility scripts.
└── tests/            # Unit tests.

### Integrations

- Adapters for engines are provided under `bodocache/integrations/`:
  - `vllm_adapter.py` and `sglang_adapter.py` expose a simple `prefetch()` API to drive BCache from vLLM/SGLang decode loops.
  - The adapters work with the current simulator and are forward-compatible with native GPU copy engines (CUDA/HIP/Level Zero).
  - See `docs/integrations.md` for build and usage details, including pinned-buffer staging and device pointer capsules.

### Native Data Plane (CUDA/HIP)

- Build the native copy engine module under `native/`:
  - CUDA: `cmake -S native -B build-cuda -DUSE_CUDA=ON && cmake --build build-cuda -j`
  - ROCm/HIP: `cmake -S native -B build-hip -DUSE_HIP=ON && cmake --build build-hip -j`
  - Ensure `pybind11`, CUDA or ROCm toolchains are installed in your environment.
  - Add the build directory to `PYTHONPATH` or copy the resulting `bodocache_agent_copy_engine` module into your Python path.

- Quick microbench:
  - `python scripts/microbench_copy.py` (optionally uses PyTorch CUDA if available to allocate a GPU destination buffer).
```

## Current Limitations

This project is a v0 prototype and has the following limitations:

1.  **Native Data Plane Footguns:** Native copy engines (CUDA/HIP/L0) are provided but require building and correct driver/toolchain setups. Advanced features (e.g., GPUDirect) are not enabled by default and vary by vendor.
2.  **Engine Glue Required:** vLLM/SGLang integrations are provided as adapters and hooks; you still need to wire `build_requests` and `dest_resolver` to your engine’s KV manager.
3.  **Synthetic Storage:** Segment backends are file-based. The optional `io_uring` reader improves throughput but production deployments should map to your actual storage stack.

## Roadmap to Production

To turn this prototype into a production-ready system, the following components need to be built. The current implementation uses a **simulated data plane** in Python; the steps below describe how to build the real, high-performance data plane.

### Step 1: Build the Native Data Plane ("The Brawn")

This component is responsible for the fast, asynchronous movement of data from CPU memory to GPU memory.

-   C++/Python bridge implemented via `pybind11` under `native/`, producing `bodocache_agent_copy_engine`.
-   Pinned memory: `acquire_host_buffer(nbytes)` returns a pinned, writable memoryview (CUDA/HIP/L0 backends).
-   Async copies: `submit(ops, callback)` enqueues multi-stream async H2D copies and invokes Python callbacks upon completion.
-   Multi-vendor: build flags for NVIDIA (CUDA), AMD (HIP), and Intel (Level Zero).

### Step 2: Node Agent and Storage

-   `NodeAgent` detects a native engine (if installed), allocates pinned buffers, reads coalesced page ranges directly into them, and enqueues device copies.
-   Optional `io_uring` reader (`-DUSE_URING=ON`) improves file→pinned throughput (`bodocache.adapters.uring_backend`).

### Step 3: Integrate with Real Inference Engines

To be useful, BCache must be connected to an actual LLM inference server.

-   Use adapters in `bodocache/integrations/` with `VLLMHook`/`SGLangHook` to drive prefetch without invasive engine changes.
-   See `docs/vllm_integration.md` and `docs/sglang_integration.md` for glue patterns and examples.

### Step 4: Benchmark and Tune

-   With the full system integrated, create a benchmark suite that simulates a realistic, real-time workload.
-   Use this benchmark to tune the Bodo planner's policies (e.g., popularity scores, eviction strategies, overlap depth) to optimize for key metrics like Time To First Token (TTFT) and overall throughput.

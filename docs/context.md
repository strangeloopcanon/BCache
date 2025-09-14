# Context and Future Work

This document provides additional context for the BodoCache project, including a discussion of alternative approaches, an overview of related work, and a list of planned enhancements.

## Alternatives

*   **Engine-local scheduling only:** Continuous batching + chunked prefill (vLLM, Sarathi-Serve). Good on single nodes, short/medium contexts. No cluster-level admission, no cross-node prewarm.
*   **Data-plane acceleration:** NIXL/Dynamo or GDS reduce copy overhead and pick transfer backends, but they don’t do global popularity, coalescing across tenants, or deadline-aware prefetch.
*   **Storage-centric designs:** LMCache + Mooncake/3FS expose offload and fast RDMA pulls; you can run with minimal planning, but top setups still add a scheduler to meet SLOs.
*   **Do nothing special:** Works if locality is high and contexts are short; many report little or no gain from hierarchical cache absent smart scheduling.

## What leading labs/systems do today

*   **LMSYS / SGLang:** HiCache + Strata. Strata adds GPU-assisted I/O and an explicit cache-aware scheduler; shows large TTFT and bandwidth improvements over vLLM+LMCache and TRT-LLM on long-context loads. Also prior “cache-aware load balancer” gains in SGLang v0.4.
*   **Moonshot (MOONCAKE):** Disaggregated KV with a global “Conductor” scheduler pairing prefill/decoding, managing replicas, and overlapping KV transfers; RDMA transfer engine saturates multi-NIC paths.
*   **vLLM stack:** Engine-local scheduler with continuous batching and chunked prefill; for offloading it integrates LMCache as the KV tier. No built-in cross-node global planner.
*   **NVIDIA Dynamo/NIXL ecosystem:** Unified transfer library with a policy engine to pick UCX/GDS/S3 paths; complements but doesn’t replace a cache planner.
*   **DeepSeek 3FS:** High-throughput KV clients and metadata layout; storage is fast, but scheduling remains a separate concern.

## Planned enhancements (toward SOTA)

*   **Prefix-aware fan-out:** add `prefix_id`/minhash buckets to planner groups to emit one op per cluster per node.
*   **Per-tenant credits:** token-bucket caps integrated in JIT to bound write-through/IO usage and enforce SLOs.
*   **Admission (selective write-through):** JIT stage based on decay/reuse probability and credits.
*   **Eviction:** coldest-first with churn guardrails using decay heat and page sizes, in JIT.
*   **Layer-aware overlap (N/N+):** JIT picking overlap depth per layer; late-copy aborts by predicted tardiness.
*   **Tie-in stream hints:** extend proto with stream/tier hints and `prefix_cluster_id`.
*   **Replay + tuner:** Bodo-based trace replay to auto-tune thresholds, caps, and overlap depth; A/B feature flags.
*   **Storage adapters:** NIXL/Mooncake/3FS clients with coalesced list-IO.

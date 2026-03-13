# Continuous Batching: Planning Document

This document covers the design, implementation plan, and rollout strategy for adding
continuous batching to llamacpp-server. For a high-level overview of all parallelism
features, see [PARALLELISM.md](PARALLELISM.md).

## Current Architecture (Separate Contexts)

```
Request 1 → Context 1 (own KV cache) → [tokenize → prefill → decode loop] → response
Request 2 → Context 2 (own KV cache) → [tokenize → prefill → decode loop] → response
Request 3 → Context 3 (own KV cache) → [tokenize → prefill → decode loop] → response
```

Each gRPC `Predict` call creates an independent llama.cpp context. The model weights
are shared read-only, but each context has its own KV cache and runs its own decode
loop. This is simple and correct, but each `llama_decode` call is a separate GPU
forward pass.

## Target Architecture (Shared Context)

```
Request 1 ─┐                                           ┌→ response 1
Request 2 ─┤→ Shared Context → Scheduler → Batch Loop ─┤→ response 2
Request 3 ─┘                                           └→ response 3
```

All requests share one context. The scheduler builds a combined batch with tokens from
all active requests, calls `llama_decode` once, then samples each request's next token
at its specific index in the batch.

## Expected Benefits

1. **GPU inference with concurrent requests** — Instead of N separate forward passes,
   one batched forward pass processes all N requests. GPU utilization goes from 1/N to
   ~100%. This can yield 2-8x throughput improvements depending on model size and
   concurrency level.

2. **Reduced VRAM usage** — One shared KV cache is more memory-efficient than N
   independent caches, because the scheduler can right-size the cache and defragment it.

3. **Chunked prefill** — Long prompts are split into chunks and interleaved with
   decode tokens from other requests, preventing prefill stalls. Without this, a single
   large prompt blocks all other requests.

**Minimal benefit when:**

- Running CPU-only inference (forward pass is the bottleneck regardless).
- Concurrency is always 1 (no batching opportunity).
- Requests are rare and never overlap.

## Mode Strategy: Replace or Coexist?

**Continuous batching should eventually fully supersede the current separate-context
approach**, but should be developed and rolled out as a separate opt-in mode first.

### Rationale for superseding

Continuous batching with `n_parallel=1` degenerates to functionally the same behavior
as the current separate-context mode: one request, one decode loop, one KV cache region.
It is a **strict superset** of the current approach:

- It handles single-request workloads identically (one slot, one decode per tick).
- It handles concurrent workloads strictly better (batched forward passes, shared KV
  cache, chunked prefill).
- It is actually *better* even at n_parallel=1: the shared context is pre-allocated once
  and reused across requests, avoiding per-request context creation/teardown overhead.

Maintaining two parallel code paths long-term would double the bug surface and testing
burden without providing meaningful benefit for any workload.

### Risks of a direct switch

- The scheduler is a complex new component — bugs would affect all in-flight requests
  simultaneously (unlike separate contexts, where a bug in one request is isolated).
- KV cache management (sequence eviction, defragmentation) is tricky to get right.
- The new code needs to be battle-tested before becoming the default.

### Rollout plan

| Phase | Description | Mode |
|-------|-------------|------|
| **1 — Current** | Separate-context parallelism with semaphore concurrency control | Default (only mode) |
| **2 — Opt-in** | Continuous batching behind `--continuous-batching` flag | Explicit opt-in |
| **3 — Default** | Continuous batching becomes the default; separate-context available via `--legacy-mode` | Default with fallback |
| **4 — Removal** | Separate-context code path removed | Only mode |

Phase 2 → 3 transition should happen after the new mode has been tested in production
with real workloads and no regressions are observed.

## Implementation Requirements

### 1. New C API Bindings

The current bindings use `llama_batch_get_one()`, which creates a single-sequence
batch. Continuous batching needs `llama_batch_init()` with per-token metadata:

```c
// Current (single-sequence, simplified batch)
llama_batch_get_one(tokens, n_tokens)

// Needed (multi-sequence batch with per-token metadata)
llama_batch batch = llama_batch_init(n_tokens, 0, n_seq_max);
batch.token[i]    = token_id;
batch.pos[i]      = position;
batch.seq_id[i]   = &seq_ids;
batch.n_seq_id[i] = 1;
batch.logits[i]   = should_output;  // only last token per sequence
batch.n_tokens    = actual_count;
```

New Go bindings needed:
- `BatchInit(nTokens, embd, nSeqMax int) *Batch` — allocates a batch with per-token
  arrays.
- Per-token setters: `SetToken(i, token)`, `SetPos(i, pos)`, `SetSeqId(i, seqId)`,
  `SetLogits(i, logits)`.

### 2. KV Cache Sequence Management

After a request completes, its KV cache entries must be freed. The llama.cpp memory
API provides:
- `llama_memory_seq_pos_max(mem, seq_id)` — query sequence position (already bound).
- Sequence removal/clearing functions (need new bindings for the b8292+ memory API).

### 3. Inference Engine (Scheduler)

A central goroutine that runs a tick loop:

```
Phase 1 — Collect: gather pending tokens from all active slots
Phase 2 — Build batch: decode tokens first (priority), then prefill chunks
Phase 3 — Forward pass: single llama_decode call
Phase 4 — Sample: llama_sampler_sample at each slot's batch index
Phase 5 — Dispatch: send generated tokens back to each slot's gRPC stream
Phase 6 — Repeat while any slot is active
```

This replaces the current per-request decode loop in `predict_cmd.go` with a
centralized scheduler. The scheduler should live in a new package (e.g.,
`internal/engine/`).

### 4. Slot Management

Each slot tracks:
- State: `idle` → `prefilling` → `generating` → `idle`
- `seq_id`: unique sequence identifier in the KV cache
- Position counter for token positions
- Sampler chain (per-slot, for independent sampling configuration)
- Response channel back to the gRPC handler

### 5. Context Parameters

The shared context would need:
- `n_ctx` = total KV cache budget across all slots (e.g., `n_parallel * per_slot_ctx`)
- `n_seq_max` = maximum number of concurrent sequences
- `n_batch` = maximum tokens per decode call

## Effort Estimate

| Component | Complexity | Estimated effort |
|-----------|-----------|-----------------|
| Batch API bindings (BatchInit, per-token setters) | Low | 1-2 days |
| KV sequence management bindings | Low | 1 day |
| Inference engine / scheduler | High | 3-5 days |
| Slot management + gRPC integration | Medium | 2-3 days |
| Testing + debugging | Medium | 2-3 days |
| **Total** | | **~2 weeks** |

## Testing Strategy

All correctness-critical logic (scheduler, slot lifecycle, batch construction, sequence
isolation, chunked prefill) is hardware-independent and fully testable on CPU. GPU
testing is only needed to validate throughput improvements.

### Correctness Tests (CPU)

These tests verify that the scheduler and slot management work correctly. They extend
the existing `run-paralleltest` infrastructure with a `--continuous-batching` variant.

| Test | What it verifies |
|------|-----------------|
| N parallel requests, greedy sampling, compare output to N sequential requests | No cross-contamination between sequences (seq_id isolation) |
| N requests with varied prompt lengths (short + long) | Chunked prefill interleaving works correctly |
| 2N requests with N slots | Backpressure: excess requests queue and complete after slots free |
| Cancel request mid-generation | Slot cleanup, KV cache entries freed for that seq_id |
| Rapid fire: many short requests cycling through slots | Slot reuse correctness, no stale state |
| Sustained load over several minutes | No memory leaks, no KV cache fragmentation buildup |
| Single slot (n_parallel=1) | Degenerates to sequential behavior, same output as separate-context mode |

All of these run on any CPU-only development machine. Greedy sampling makes outputs
deterministic, so comparing continuous-batching output against separate-context output
for the same prompts is a strong correctness oracle.

### Performance Tests (GPU)

These tests validate that batching delivers the expected throughput improvement over the
separate-context baseline. They require a GPU.

| Test | What it measures |
|------|-----------------|
| Throughput: N concurrent requests, continuous batching vs separate-context | Tokens/sec improvement from batched forward passes |
| Latency under load: time-to-first-token and inter-token latency at varying N | Chunked prefill prevents latency spikes |
| VRAM usage: continuous batching vs N separate contexts | Memory efficiency of shared KV cache |
| Scaling: throughput as N increases from 1 to max slots | Batching benefit curve, saturation point |

### Hardware Requirements

| Level | Hardware | Purpose |
|-------|----------|---------|
| **Development / CI** | CPU only | All correctness tests. No GPU needed. |
| **Minimal GPU** | 1 GPU, 6+ GB VRAM (e.g. GTX 1060, T4) | Validate batching throughput benefit with a small model (1-3B Q4). Cloud: AWS `g4dn.xlarge` (~$0.50/hr). |
| **Recommended GPU** | 1-2 GPUs, 24+ GB VRAM each (e.g. RTX 3090/4090, A10G) | Production-representative testing with 7B-13B models, 8-16 concurrent slots, varied workloads. |
| **Full validation** | 2+ GPUs | Continuous batching combined with PP/TP. Sustained soak tests (hours) for memory leaks and KV cache fragmentation. Pre-release only. |

### Test Implementation Plan

1. **Extend `grpcclienttest`** with a `--continuous-batching` flag that tells the server
   to use the new mode. The existing `parallel` test mode already launches N concurrent
   requests and validates responses — the same harness works for both modes.

2. **Add a comparison test mode** (`--test-mode compare`) that runs the same set of
   prompts first in separate-context mode, then in continuous-batching mode, and
   asserts identical outputs (greedy sampling required).

3. **Add a `run-batchtest` Makefile target** mirroring `run-paralleltest` but with
   `--continuous-batching` enabled on the server side.

4. **GPU benchmarks** as a separate Makefile target (`run-benchmarks`) that reports
   throughput numbers for both modes. Not part of CI — run manually before releases.

## Reference Implementations

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/server.cpp) —
  C++ implementation with slot management, continuous batching, and chunked prefill.
- [LlamaCppEx ADR 006](https://hexdocs.pm/llama_cpp_ex/006-continuous-batching.html) —
  Design document for continuous batching with Elixir NIFs, directly applicable to
  our Go architecture.

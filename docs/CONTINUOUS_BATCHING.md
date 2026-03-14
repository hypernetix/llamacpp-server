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

- Concurrency is always 1 (no batching opportunity).
- Requests are rare and never overlap.

### Measured Results (CPU-only, Mistral-7B Q4_0, 4 concurrent requests)

| Metric | Separate-context | Continuous batching | Improvement |
|--------|-----------------|-------------------|-------------|
| Wall-clock time | 162.4s | 82.5s | **2.0x faster** |
| Aggregate throughput | 3.26 tok/s | 5.85 tok/s | **1.8x higher** |
| Per-request throughput | 0.82–0.85 tok/s | 1.1–1.8 tok/s | ~1.5x higher |
| KV cache memory | 4 × 4096 contexts | 1 × 4096 shared | ~4x less |

Even on CPU — where the forward pass is inherently sequential — continuous batching
delivers a 2x wall-clock speedup because one batched `llama_decode` call avoids the
overhead of 4 separate context switches and forward passes. GPU workloads should see
proportionally larger gains since the GPU can utilise the batch parallelism directly.

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

| Phase | Description | Mode | Status |
|-------|-------------|------|--------|
| **1** | Separate-context parallelism with semaphore concurrency control | Default (only mode) | Done |
| **2** | Continuous batching behind `--continuous-batching` flag | Explicit opt-in | **Done** — validated on CPU with 2x speedup |
| **3** | Continuous batching becomes the default; separate-context available via `--legacy-mode` | Default with fallback | Planned |
| **4** | Separate-context code path removed | Only mode | Planned |

### Phase 3 — Make Continuous Batching the Default

**Prerequisites** (before Phase 3 can begin):

- [ ] Docker CI integration for continuous batching tests
- [ ] Comparison test mode (`--test-mode compare`): same prompts in both modes,
      assert identical greedy outputs
- [ ] Single-slot test (`n_parallel=1`): confirm identical output to separate-context
- [ ] Backpressure test: 2N requests with N slots, all complete correctly
- [ ] GPU validation (if hardware available): confirm throughput scaling

**Implementation steps:**

1. **Flip the default**: `--continuous-batching` becomes the default when
   `--n-parallel` is set (or defaulted). The old mode becomes `--legacy-mode`.
2. **`--n-parallel` defaults to 1** (was 0=unlimited). With CB, unlimited doesn't
   make sense — each slot consumes KV cache budget. Users configure based on their
   context size and expected concurrency.
3. **`--ctx-size` semantics**: with CB, this is the *total* KV cache across all
   slots. Per-slot budget = `ctx-size / n-parallel`. Document this clearly.
4. **Update Docker configs**: remove `--continuous-batching` from Docker entrypoints
   (it's now default). Add `--legacy-mode` only if needed for compatibility testing.
5. **Update CI**: all integration tests run against CB by default. Add one legacy-mode
   run to prevent regressions during the transition period.
6. **Update documentation**: README, PARALLELISM.md, all examples.
7. **Deprecation log**: when `--legacy-mode` is used, emit a startup warning that the
   flag will be removed in Phase 4.

**Estimated effort**: 2–3 days (mostly testing and documentation).

### Phase 4 — Remove Separate-Context Code Path

**Prerequisites** (before Phase 4 can begin):

- [ ] Phase 3 has been running in production without regressions
- [ ] No users relying on `--legacy-mode` for correctness (only for comparison testing)

**Implementation steps:**

1. Remove `--legacy-mode` / `--continuous-batching` flags (CB is the only mode).
2. Delete `internal/llmservice/predict_cmd.go` (the per-request decode loop).
3. Delete `internal/inference/predictions_manager.go` (the semaphore-based manager).
4. Simplify `internal/llmservice/service.go` — directly create `engine.Engine`,
   no conditional branching.
5. `PredictOptions` loses `ContinuousBatching` and `NParallel` fields (NParallel
   moves to engine options only).
6. Remove `run-paralleltest` Makefile target (replaced by `run-batchtest`).
7. Update all tests and documentation.

**Files deleted:**
- `internal/llmservice/predict_cmd.go` (~300 lines)
- `internal/inference/predictions_manager.go` (~100 lines)

**Files simplified:**
- `internal/llmservice/service.go` — remove conditional engine creation
- `cmd/llamacppserver/main.go` — remove `--continuous-batching` and `--legacy-mode`
- `Makefile` — merge `run-batchtest` into `run-paralleltest`

**Estimated effort**: 1 day.

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
- Sequence removal/clearing functions (need new bindings for the b8323+ memory API).

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

1. **`--continuous-batching` flag on `llamacppclienttest`** — When present, the spawned
   server starts with `--continuous-batching --n-parallel N`. When connecting to an
   external server (`--port`), the flag has no effect (the server is already configured).
   **Status: Done.**

2. **`run-batchtest` Makefile target** — Mirrors `run-paralleltest` but passes
   `--continuous-batching` to the client test, which forwards it to the spawned server.
   Uses greedy sampling (`Temperature: 0.0`) for deterministic output.
   **Status: Done.**

   ```bash
   # Run with default 4 slots
   make run-batchtest MODEL_PATH=/path/to/model.gguf

   # Run with 8 slots
   make run-batchtest MODEL_PATH=/path/to/model.gguf PARALLEL_N=8
   ```

3. **Comparison test mode** (`--test-mode compare`) — Run the same prompts first in
   separate-context mode, then in continuous-batching mode, and assert identical outputs
   (greedy sampling required). **Status: Not started (future).**

4. **Docker CI** — Add CB test services to docker-compose.ci.yml and integration test
   scripts. **Status: Deferred** until the engine is validated with manual testing.

5. **GPU benchmarks** as a separate Makefile target (`run-benchmarks`) that reports
   throughput numbers for both modes. Not part of CI — run manually before releases.
   **Status: Not started (future).**

## Implementation Progress

### Phase 2 — Opt-in Continuous Batching

| Component | Status | Notes |
|-----------|--------|-------|
| Batch API bindings (`BatchInit`, `Add`, per-token setters) | **Done** | `internal/bindings/llamacpp.go` — `BatchInit(nTokens, embd, nSeqMax)`, `Add(token, pos, seqId, logits)`, `Clear()`, individual setters |
| KV sequence management bindings | **Done** | `Memory` type with `Clear`, `SeqRm`, `SeqCp`, `SeqKeep`, `SeqAdd`, `SeqPosMin`, `SeqPosMax` |
| Inference engine / scheduler | **Done** | `internal/engine/engine.go` — tick loop: build batch → decode → sample → dispatch |
| Slot management | **Done** | `internal/engine/slot.go` — idle/prefilling/generating state machine, per-slot sampler chains |
| Service integration (`--continuous-batching` flag) | **Done** | `--continuous-batching` CLI flag, `llmservice.NewService` creates `engine.Engine` when enabled |
| Correctness tests (CPU) | **Done** | `run-batchtest` validated: 4 concurrent requests, 2x wall-clock speedup vs separate-context |

### Bindings API Reference

**Batch** (`internal/bindings/llamacpp.go`):

```go
BatchInit(nTokens, embd, nSeqMax int) *Batch  // allocate multi-sequence batch
b.Add(token, pos, seqId int, logits bool)      // append token to batch
b.Clear()                                       // reset to empty (no dealloc)
b.SetToken(i, token int)                        // per-token setters
b.SetPos(i, pos int)
b.SetSeqId(i, seqId int)
b.SetLogits(i int, logits bool)
b.SetNTokens(n int)
b.Cap() int                                     // max capacity
b.NTokens() int                                 // current count
b.Free()                                        // deallocate owned batch
```

**Memory** (`internal/bindings/llamacpp.go`):

```go
ctx.Memory() *Memory                            // get KV cache memory handle
m.Clear(data bool)                              // clear all memory
m.SeqRm(seqId, p0, p1 int) bool                // remove sequence tokens in [p0, p1)
m.SeqCp(seqIdSrc, seqIdDst, p0, p1 int)        // copy sequence
m.SeqKeep(seqId int)                            // remove all except this sequence
m.SeqAdd(seqId, p0, p1, delta int)              // shift positions by delta
m.SeqPosMin(seqId int) int                      // smallest position (-1 if empty)
m.SeqPosMax(seqId int) int                      // largest position (-1 if empty)
```

### Phase 2 Validation Results

Tested with Mistral-7B-Instruct-v0.2 Q4_0 on CPU (Intel, no GPU), 4 concurrent
greedy-sampled requests, `--n-parallel 4 --ctx-size 4096`:

- **All 4 requests completed successfully** with coherent, non-contaminated responses.
- **2.0x wall-clock speedup** over separate-context mode (82.5s vs 162.4s).
- **1.8x aggregate throughput** (5.85 tok/s vs 3.26 tok/s).
- Slot lifecycle worked correctly: prefill → generate → EoG/max-tokens → idle.
- KV cache cleanup on slot finish confirmed (no stale state on reuse).

### Engine Architecture

The engine (`internal/engine/`) is a single-goroutine scheduler that implements
`inference.PredictionsManager`. It is activated by passing `--continuous-batching`
to the server.

```
                      ┌─────────────────────────────────────────────────┐
  Predict() ─────────►│              request channel                    │
  Predict() ─────────►│                                                 │
  Predict() ─────────►│                                                 │
                      └────────────────────┬────────────────────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────────────┐
                      │              Engine.run()                       │
                      │                                                 │
                      │  1. Assign requests to idle slots               │
                      │  2. Build batch:                                │
                      │     - generating slots → 1 decode token each   │
                      │     - prefilling slots → prompt chunk           │
                      │  3. llama_decode(batch)                         │
                      │  4. Sample at each slot's batch index            │
                      │  5. Stream tokens back / finish slots           │
                      │  6. Repeat while any slot is active             │
                      │                                                 │
                      │  Slots: [0] [1] [2] ... [N-1]                  │
                      │  States: idle / prefilling / generating         │
                      └─────────────────────────────────────────────────┘
```

**Files:**
- `engine.go` — `Engine` struct, `New`, `Predict`, `Stop`, run loop, tick cycle
- `slot.go` — `slot` struct, state machine, `request`/`requestResult` types
- `sampler.go` — `buildSamplerChain` helper (same logic as `predict_cmd.go`)

**Usage:**

```bash
# Enable continuous batching with 4 slots
llamacppserver --continuous-batching --n-parallel 4 --ctx-size 8192

# Without the flag, the server uses the original separate-context mode
llamacppserver --n-parallel 4 --ctx-size 4096
```

## Reference Implementations

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/server.cpp) —
  C++ implementation with slot management, continuous batching, and chunked prefill.
- [LlamaCppEx ADR 006](https://hexdocs.pm/llama_cpp_ex/006-continuous-batching.html) —
  Design document for continuous batching with Elixir NIFs, directly applicable to
  our Go architecture.

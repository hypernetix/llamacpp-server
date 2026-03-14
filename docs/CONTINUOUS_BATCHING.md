# Continuous Batching

This document describes the continuous batching engine in llamacpp-server — its
architecture, benefits, benchmark results, and testing strategy. For a high-level
overview of all parallelism features, see [PARALLELISM.md](PARALLELISM.md).

## What is Continuous Batching?

In a naive LLM server, each request gets its own inference context with a private
KV cache, and each `llama_decode` call processes tokens for only one request. Under
concurrency, this means N separate forward passes per generation step — each
underutilising the hardware.

Continuous batching (also called dynamic batching) uses a **single shared context**.
A scheduler collects tokens from all active requests and issues one `llama_decode` call
per tick, processing all requests simultaneously. Each request is isolated by a unique
sequence ID in the shared KV cache.

```
Request 1 ─┐                                           ┌→ response 1
Request 2 ─┤→ Shared Context → Scheduler → Batch Loop ─┤→ response 2
Request 3 ─┘                                           └→ response 3
```

This is the same technique used by llama.cpp's own HTTP server, vLLM (PagedAttention),
and SGLang (RadixAttention).

## Benefits

**Higher throughput under concurrency.** Instead of N separate forward passes, one
batched pass processes all N requests. GPU utilisation scales with concurrency instead
of being capped at 1/N. This typically yields 2-8x throughput improvements depending
on model size and request count.

**Lower memory usage.** One shared KV cache is more memory-efficient than N independent
caches. The scheduler reuses freed sequence slots immediately, avoiding the allocation
and teardown overhead of per-request contexts.

**Chunked prefill.** Long prompts are split into chunks and interleaved with decode
tokens from other requests. This prevents a single large prompt from stalling all other
in-flight generation, keeping time-to-first-token predictable.

**Minimal overhead at low concurrency.** With `n_parallel=1`, the engine degenerates to
sequential request processing with zero batching overhead. The shared context is
pre-allocated once and reused across requests, which is actually more efficient than
creating/destroying per-request contexts.

## Architecture

### Engine Overview

The engine (`internal/inferenceengine/`) is a single-goroutine scheduler that implements
the `PredictionsManager` interface. Incoming `Predict()` calls are serialised through a
request channel and dispatched to available slots.

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
                      │  4. Sample at each slot's batch index           │
                      │  5. Stream tokens back / finish slots           │
                      │  6. Repeat while any slot is active             │
                      │                                                 │
                      │  Slots: [0] [1] [2] ... [N-1]                  │
                      │  States: idle / prefilling / generating         │
                      └─────────────────────────────────────────────────┘
```

### Tick Loop

Each tick of the scheduler follows a strict phase order:

1. **Assign** — drain the request channel, assign pending requests to idle slots.
2. **Build batch** — decode tokens from generating slots get priority (one token each);
   remaining batch budget is filled with prefill chunks from prefilling slots.
3. **Forward pass** — single `llama_decode(batch)` call.
4. **Sample** — `llama_sampler_sample` at each generating slot's batch index to produce
   the next token.
5. **Dispatch** — stream generated tokens back to each slot's response channel. If a
   slot reaches EOS or `max_tokens`, it transitions to idle and its KV cache sequence
   is freed.
6. **Repeat** while any slot is active or requests are pending.

When all slots are idle and no requests are pending, the engine parks (blocks on the
request channel) until new work arrives.

### Slot State Machine

Each slot tracks the lifecycle of a single inference request:

```
idle ──► prefilling ──► generating ──► idle
              │                          ▲
              └──────────────────────────┘
                   (short prompts skip prefilling)
```

- **idle** — available for assignment.
- **prefilling** — processing prompt tokens in chunks. Transitions to generating when
  the full prompt is consumed.
- **generating** — autoregressive decoding, one token per tick. Finishes on EOS,
  `max_tokens`, or error.

Each slot owns its own sampler chain (configured per-request), sequence ID, position
counter, and response channel.

### Key Implementation Files

| File | Purpose |
|------|---------|
| `internal/inferenceengine/engine.go` | `Engine` struct, `PredictionsManager` interface, `PredictArgs`, `StreamFunc`, run loop |
| `internal/inferenceengine/slot.go` | `slot` struct, state machine, `request`/`requestResult` types |
| `internal/inferenceengine/sampler.go` | `buildSamplerChain` — constructs sampler chain from request args |

### Go Bindings

The engine uses Go bindings (CGO) for the llama.cpp C API:

**Batch API** — multi-sequence batch construction:

```go
BatchInit(nTokens, embd, nSeqMax int) *Batch
b.Add(token, pos, seqId int, logits bool)
b.Clear()
b.NTokens() int
b.Free()
```

**Memory API** — KV cache sequence management:

```go
ctx.Memory() *Memory
m.SeqRm(seqId, p0, p1 int) bool      // free sequence tokens in [p0, p1)
m.SeqCp(seqIdSrc, seqIdDst, p0, p1)  // copy sequence
m.SeqKeep(seqId int)                  // evict all except this sequence
m.SeqAdd(seqId, p0, p1, delta int)    // shift positions
```

## Benchmark Results

### CPU-only (Mistral-7B Q4_0, 4 concurrent requests)

Tested on Intel CPU, no GPU, `--n-parallel 4 --ctx-size 4096`, greedy sampling:

| Metric | Per-request contexts | Continuous batching | Improvement |
|--------|---------------------|---------------------|-------------|
| Wall-clock time | 162.4s | 82.5s | **2.0x faster** |
| Aggregate throughput | 3.26 tok/s | 5.85 tok/s | **1.8x higher** |
| Per-request throughput | 0.82–0.85 tok/s | 1.1–1.8 tok/s | ~1.5x higher |
| KV cache memory | 4 × 4096 contexts | 1 × 4096 shared | **~4x less** |

Even on CPU — where the forward pass is inherently sequential — continuous batching
delivers a 2x wall-clock speedup because one batched `llama_decode` call avoids the
overhead of 4 separate context switches and forward passes.

### Expected GPU results

GPU workloads should see proportionally larger gains because the GPU can exploit batch
parallelism directly. Based on published benchmarks from similar architectures:

- **2-4 concurrent requests**: 2-4x throughput improvement.
- **8-16 concurrent requests**: 4-8x throughput improvement (until VRAM-bound).
- **Time-to-first-token**: chunked prefill keeps TTFT stable under load, while
  per-request contexts show increasing TTFT as concurrency grows.

GPU benchmarks for llamacpp-server are planned but not yet collected (see Future Work).

## Comparison with llama.cpp's Own Server

llamacpp-server's continuous batching engine is architecturally similar to the one in
llama.cpp's built-in HTTP server (`examples/server/server.cpp`), sharing the same
underlying C API. The key differences are in the server layer, not the inference engine:

| Aspect | llamacpp-server | llama.cpp server |
|--------|----------------|-----------------|
| **Server language** | Go | C++ |
| **API protocols** | gRPC + HTTP/SSE | HTTP (OpenAI-compatible) |
| **Batch scheduling** | Go goroutine, tick loop | C++ thread, tick loop |
| **Slot state machine** | idle → prefilling → generating | idle → started → generating |
| **Sampler chain** | Per-slot, built from request args | Per-slot, built from request args |
| **KV cache management** | `llama_memory` API (b8323+) | `llama_kv_cache` / `llama_memory` API |
| **Chunked prefill** | Supported | Supported |
| **Speculative decoding** | Not yet | Supported |
| **Grammar-constrained sampling** | Not yet | Supported |
| **Prompt caching / reuse** | Not yet | Supported (system prompt caching) |
| **Embedding / reranking** | Not yet | Supported |

Both implementations use the same tick-driven scheduler pattern (decode-maximal
scheduling, chunked prefill, per-slot sampler chains). The main advantage of
llamacpp-server is its gRPC interface and Go ecosystem integration; llama.cpp's server
offers more advanced features due to its longer development history.

## Hardware Requirements

| Level | Hardware | Purpose |
|-------|----------|---------|
| **Development / CI** | CPU only | All correctness tests run on CPU. No GPU needed. |
| **Minimal GPU** | 1 GPU, 6+ GB VRAM (e.g. GTX 1060, T4) | Validate batching throughput with small models (1-3B Q4). Cloud: AWS `g4dn.xlarge` (~$0.50/hr). |
| **Recommended GPU** | 1-2 GPUs, 24+ GB VRAM each (e.g. RTX 3090/4090, A10G) | Production-representative testing: 7B-13B models, 8-16 concurrent slots. |
| **Full validation** | 2+ GPUs | CB combined with PP/TP. Sustained soak tests for memory leaks and KV cache fragmentation. |

## Testing

All correctness-critical logic (scheduler, slot lifecycle, batch construction, sequence
isolation) is hardware-independent and fully testable on CPU. Greedy sampling
(`Temperature: 0.0`) produces deterministic output, making test assertions reliable.

### Available Tests

| Test | Command | What it verifies |
|------|---------|-----------------|
| **Parallel inference** | `make run-paralleltest` | N concurrent requests complete without errors, no cross-contamination |
| **Backpressure** | `make run-backpressuretest` | 2N requests with N slots: excess requests queue and complete after slots free |
| **Docker CI (gRPC)** | `make docker-integration-test-ci` | End-to-end: model download → load → gRPC predict (single slot) |
| **Docker CI (HTTP)** | `make docker-integration-test-ci` | End-to-end: model download → load → HTTP/SSE predict (single slot) |
| **Docker CI (parallel)** | `make docker-integration-test-ci` | End-to-end: 4-slot server, 4 concurrent requests |

```bash
# Local parallel test (4 slots, 4 requests)
make run-paralleltest MODEL_PATH=/path/to/model.gguf

# Local parallel test (8 slots, 8 requests)
make run-paralleltest MODEL_PATH=/path/to/model.gguf PARALLEL_N=8

# Backpressure test (4 slots, 8 requests)
make run-backpressuretest MODEL_PATH=/path/to/model.gguf PARALLEL_N=4

# Full Docker CI (downloads model automatically)
make docker-integration-test-ci
```

### Correctness Test Matrix

| Test scenario | Status |
|--------------|--------|
| N parallel requests, greedy sampling | Done |
| 2N requests with N slots (backpressure / queuing) | Done |
| Docker CI: gRPC, HTTP, and parallel tests | Done |
| N requests with varied prompt lengths (chunked prefill) | Planned |
| Cancel request mid-generation | Planned |
| Rapid-fire short requests (slot reuse correctness) | Planned |
| Sustained load (memory leak detection) | Planned |

### Performance Test Matrix (GPU)

| Test scenario | Status |
|--------------|--------|
| Throughput: N concurrent requests at varying N | Planned |
| Latency: time-to-first-token under load | Planned |
| VRAM usage at varying slot counts | Planned |
| Throughput scaling curve (1 to max slots) | Planned |

## Future Work

| Feature | Priority | Description |
|---------|----------|-------------|
| **GPU benchmarks** | High | Collect throughput and latency numbers on GPU hardware to quantify the batching benefit beyond CPU. |
| **Speculative decoding** | Medium | Use a small draft model to propose tokens, verified by the main model. Can 2-3x single-request throughput. |
| **Grammar-constrained sampling** | Medium | Restrict output to a formal grammar (JSON, SQL, etc.) using llama.cpp's grammar sampler. |
| **Prompt caching** | Medium | Reuse KV cache entries for shared prompt prefixes (e.g. system prompts) across requests. |
| **KV cache defragmentation** | Low | Compact the KV cache when fragmentation from finished sequences degrades performance. |
| **Embedding endpoint** | Low | Expose llama.cpp's embedding mode for retrieval and reranking workloads. |

## Usage

```bash
# Default: 1 slot, sequential processing
llamacppserver --grpc-port 50052

# 4 concurrent slots, 8K total context (2K per slot)
llamacppserver --n-parallel 4 --ctx-size 8192

# 8 slots with flash attention on GPU
llamacppserver --ngpu 99 --flash-attn --n-parallel 8 --ctx-size 16384
```

## References

- [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/server.cpp) —
  C++ implementation with slot management, continuous batching, and chunked prefill.
- [vLLM: PagedAttention](https://arxiv.org/abs/2309.06180) — the seminal paper on
  paged KV cache management for efficient LLM serving.
- [LlamaCppEx ADR 006: Continuous Batching](https://hexdocs.pm/llama_cpp_ex/006-continuous-batching.html) —
  design document for continuous batching with Elixir NIFs.

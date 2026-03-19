# Go vs Rust llama.cpp Implementation: Feature Comparison

**Date**: March 19, 2026
**Revision**: 2.0 (supersedes [v0/go-vs-rust-llama-cpp-comparison.md](v0/go-vs-rust-llama-cpp-comparison.md) from October 2025)
**Purpose**: Evaluate whether transitioning from Go to Rust for llama.cpp bindings is justified

---

## Executive Summary

This document compares the current Go-based llamacpp-server with available Rust
llama.cpp bindings. Both sides have matured significantly since the October 2025
review:

- **This project** has grown from a gRPC-only server into a production-ready
  platform with an OpenAI-compatible API, continuous batching, multi-slot
  parallelism, flash attention, cross-platform builds, and Docker CI/CD.
- **utilityai/llama-cpp-rs** has become the clear Rust front-runner with 139
  releases, comprehensive bindings (embeddings, grammar, LoRA, chat templates),
  and an example OpenAI completions server.
- **mdrokz/rust-llama.cpp** and **edgenai/llama_cpp-rs** are effectively
  abandoned (no commits since mid-2024).

**Key Finding**: The gap at the *binding* level has narrowed — `llama-cpp-2` now
covers most of the llama.cpp C API surface. However, the gap at the *server*
level remains wide. This project provides a complete, tested, deployable
inference server; Rust alternatives provide libraries and examples.

**Recommendation**: **Stay with Go.** The binding-level parity does not offset
the 6+ months of server infrastructure, testing, and operational tooling that
would need to be rebuilt. Monitor `llama-cpp-2` for a mature server framework.

---

## Participants

| Project | Language | Type | Stars | Latest Release | Status |
|---------|----------|------|-------|----------------|--------|
| **hypernetix/llamacpp-server** (this) | Go 1.22 + CGO | Bindings + full server | — | continuous | Active |
| [utilityai/llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) | Rust | Bindings (`llama-cpp-2` crate) + example server | 500 | v0.1.139 (Mar 2026) | Active |
| [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp) | Rust | Thin bindings | 419 | v0.3.0 | Inactive since Jun 2024 |
| [edgenai/llama_cpp-rs](https://github.com/edgenai/llama_cpp-rs) | Rust | High-level async bindings | 243 | v0.2.2 (Nov 2023) | Inactive since Jun 2024 |

> **Note on consolidation**: The Rust ecosystem has effectively consolidated
> around `utilityai/llama-cpp-rs`. The other two projects are no longer
> maintained. This comparison therefore focuses primarily on `llama-cpp-2` as
> the Rust contender.

---

## Functional Comparison

### 1. Binding Features

What each project exposes from the llama.cpp C API.

| Feature | Go (this project) | Rust (`llama-cpp-2`) | Notes |
|---------|-------------------|----------------------|-------|
| **Model loading** | ✅ | ✅ | Both support GGUF |
| **Progress callbacks** | ✅ | ✅ | llama-cpp-2 added this |
| **Model parameters** (GPU layers, mmap, split mode, tensor split) | ✅ Full | ✅ Full | Parity |
| **Context management** (n_ctx, batch, threads, seq) | ✅ Full | ✅ Full | Parity |
| **Flash attention** | ✅ | ✅ | Both via llama.cpp flag |
| **Tokenization** (encode, decode, special tokens, BOS/EOS) | ✅ Full | ✅ Full | Parity |
| **Sampling** (temp, top-k, top-p, min-p, penalties, greedy) | ✅ Full chain | ✅ Sampler module | Parity |
| **Grammar-constrained generation** (GBNF) | ✅ In bindings | ✅ `GrammarError`, JSON schema | Rust has JSON-schema-to-grammar helper |
| **Batch inference** | ✅ | ✅ | Both |
| **KV cache management** | ✅ Full + quantization | ✅ | Go has KV quantization wired |
| **Embeddings** (`llama_encode`) | ❌ Not yet exposed | ✅ `EmbeddingsError` | Rust ahead |
| **LoRA adapters** | ❌ Not yet exposed | ✅ Init/Set/Remove | Rust ahead |
| **Chat templates** | ✅ ChatML in server | ✅ `LlamaChatTemplate` from model | Rust reads the model's template; Go hardcodes ChatML |
| **GGUF metadata inspection** | ✅ Basic (arch, params, size) | ✅ `MetaValError` | Rust has richer metadata access |
| **Speculative decoding** | ❌ | ❌ | Neither exposes this yet |
| **Custom logging** | ✅ | ✅ `send_logs_to_tracing` | Both |
| **Backend device enumeration** | ❌ | ✅ `list_llama_ggml_backend_devices` | Rust ahead |

**Binding summary**: `llama-cpp-2` has reached near-parity and pulls ahead on
embeddings, LoRA, chat templates from model metadata, and backend device
enumeration. The Go bindings lead on KV cache quantization wiring.

### 2. Server Features

What each project provides as a *deployable inference server*.

| Feature | Go (this project) | Rust (`llama-cpp-2`) |
|---------|-------------------|----------------------|
| **OpenAI `/v1/chat/completions`** | ✅ Streaming + non-streaming | ⚠️ Example `openai-server` |
| **OpenAI `/v1/completions`** | ✅ Streaming + non-streaming | ⚠️ Example |
| **OpenAI `/v1/models`** | ✅ | ❌ |
| **Custom HTTP+SSE API** | ✅ Full | ❌ |
| **gRPC API** | ✅ Full (Ping, LoadModel, Predict) | ❌ |
| **Model management** (load, cache, concurrent access) | ✅ Thread-safe manager | ❌ |
| **Model load progress streaming** | ✅ SSE progress events | ❌ |
| **Continuous batching** | ✅ Multi-slot scheduler | ❌ |
| **Multi-slot parallelism** (n_parallel) | ✅ Configurable slots | ❌ |
| **Pipeline parallelism** (multi-GPU layer split) | ✅ Wired | ❌ |
| **Tensor parallelism** (multi-GPU row split) | ✅ Wired | ❌ |
| **Graceful shutdown** (drain slots, SIGTERM) | ✅ | ❌ |
| **Health endpoint** | ✅ | ❌ |
| **Docker images** (CPU, Vulkan, CUDA planned) | ✅ Multi-variant | ❌ |
| **Docker CI/CD** (integration tests) | ✅ Multiple compose files | ❌ |
| **OpenAPI spec** | ✅ Two specs (custom + v1) | ❌ |
| **Cross-platform builds** (Win/Linux/macOS) | ✅ Unified Makefile | ⚠️ Cargo features |
| **Test coverage** (baseline, greedy, seeded, stress, parallel, backpressure, OpenAI SDK) | ✅ Comprehensive | ❌ |

**Server summary**: This is where the gap remains wide. `llama-cpp-2` provides
an example OpenAI server but not a production inference platform. This project
provides continuous batching, model management, multi-API support, Docker CI/CD,
and comprehensive test suites — none of which exist in the Rust ecosystem.

---

## Detailed Feature Matrix

| Category | Go (this project) | Rust (`llama-cpp-2`) | Winner |
|----------|-------------------|----------------------|--------|
| **Bindings** | | | |
| Model loading + params | ✅ Full | ✅ Full | Tie |
| Context + batch + flash attn | ✅ Full | ✅ Full | Tie |
| Tokenization | ✅ Full | ✅ Full | Tie |
| Sampling (chain, all strategies) | ✅ Full | ✅ Sampler module | Tie |
| Grammar / structured output | ✅ GBNF | ✅ GBNF + JSON schema | Rust |
| Embeddings | ❌ | ✅ | Rust |
| LoRA adapters | ❌ | ✅ | Rust |
| Chat template from model | ❌ (ChatML hardcoded) | ✅ | Rust |
| KV cache quantization | ✅ q8_0, q4_0 | ⚠️ Basic | Go |
| GGUF metadata | ✅ Basic | ✅ Richer | Rust |
| **Server** | | | |
| OpenAI-compatible API | ✅ Full | ⚠️ Example only | Go |
| gRPC API | ✅ Full | ❌ | Go |
| Continuous batching | ✅ | ❌ | Go |
| Multi-slot parallelism | ✅ | ❌ | Go |
| Model management | ✅ Thread-safe | ❌ | Go |
| Multi-GPU wiring | ✅ | ❌ | Go |
| Graceful shutdown | ✅ | ❌ | Go |
| **Operations** | | | |
| Docker images + CI | ✅ | ❌ | Go |
| OpenAPI specs | ✅ | ❌ | Go |
| Test suites | ✅ 7 test modes | ❌ | Go |
| Cross-platform Makefile | ✅ Win/Linux/macOS | N/A (Cargo) | Go |
| **Developer Experience** | | | |
| Release cadence | Continuous | 139 releases (fast) | Rust |
| llama.cpp version tracking | Manual (pinned version) | Submodule, fast follow | Rust |
| Build complexity | CGO + Makefile | `cargo build` | Rust |
| Type safety of bindings | ✅ Go wrappers | ✅ Rust ownership model | Rust |
| **Performance** | | | |
| Inference speed | ✅ llama.cpp | ✅ llama.cpp | Tie |
| FFI overhead | Minimal (CGO) | Minimal (Rust FFI) | Tie |
| Memory predictability | ✅ Good (GC is modern) | ✅ No GC | Rust (theoretical) |
| Concurrency model | Goroutines | async/Tokio | Tie |

---

## Non-Functional Comparison

| Aspect | Go (this project) | Rust (`llama-cpp-2`) |
|--------|-------------------|----------------------|
| **Ecosystem maturity** | Stable, production-proven | Rapidly maturing library |
| **Community size** | Go ecosystem is large | 500 stars, 71 contributors, growing |
| **Documentation** | OpenAPI specs, test output, roadmap | docs.rs API docs, examples |
| **Dependency count** | Minimal (go-flags, grpc, protobuf) | Minimal (llama-cpp-sys-2) |
| **Binary size** | Single static binary + llama.cpp .so | Single static binary + llama.cpp .so |
| **Memory safety** | Go GC + runtime.Pinner for CGO | Rust ownership, no GC |
| **Error handling** | `error` returns | `Result<T, E>` with typed errors |
| **Compile times** | Fast (~30s) | Slow (~2-5min with llama.cpp build) |
| **Debugging** | Easy (delve, printf) | Harder (lldb, complex lifetimes) |

---

## Migration Effort Analysis

### What Would Need to Be Rebuilt in Rust

Since October 2025, this project has grown considerably. The rebuild scope is
larger than previously estimated:

| Component | Go lines | Estimated Rust | Notes |
|-----------|----------|----------------|-------|
| Bindings layer | ~700 | ~0 (use `llama-cpp-2`) | Rust bindings now cover this |
| Inference engine (continuous batching, slot scheduler) | ~500 | ~600-800 | No Rust equivalent exists |
| Model management (concurrent load, cache, progress) | ~225 | ~300-400 | No Rust equivalent exists |
| HTTP server (custom + OpenAI APIs) | ~450 | ~500-700 | axum/actix, SSE, OpenAI types |
| gRPC server | ~200 | ~300 | tonic |
| OpenAPI specs | ~750 | ~750 | Reusable YAML |
| Docker infrastructure | ~300 | ~300 | Largely reusable |
| Test infrastructure (7 modes, Python SDK test) | ~500 | ~500 | Test logic reusable, code not |
| CLI + configuration | ~240 | ~200 | clap |
| **Total new Rust code** | | **~3,000-3,500** | |

**Development time**: 8-12 weeks for an experienced Rust developer (up from the
4-8 week estimate in October 2025, reflecting the server's growth).

**What is now free**: The binding layer. In October 2025, missing Rust bindings
were a significant cost item. `llama-cpp-2` now covers embeddings, LoRA, grammar,
chat templates, and sampling — features the Go bindings partially lack.

**What remains expensive**: The server infrastructure. Continuous batching,
multi-slot scheduling, model management with progress broadcasting, and the full
test suite have no Rust equivalents and would be the bulk of the migration work.

---

## Performance Considerations

### The Core Argument Hasn't Changed

Both Go and Rust call the **same llama.cpp C++ library** for inference. The
bottleneck is matrix multiplication on CPU/GPU, not the language of the server
wrapper.

| Metric | Go | Rust | Difference |
|--------|----|----|------------|
| Token generation | llama.cpp | llama.cpp | 0% |
| CGO/FFI call overhead | ~50-100ns per call | ~10-30ns per call | Irrelevant vs ~10ms per token |
| Memory (GC pauses) | <1ms with modern Go GC | None | Irrelevant for inference latency |
| Concurrency overhead | Goroutines (~2KB each) | Tokio tasks (~256B each) | Negligible |

**Measured throughput** on this project (SmolLM2-135M, CPU, Docker):
- 93 tok/s non-streaming
- 68 tok/s streaming (includes per-token SSE flush)

The streaming overhead is HTTP flushing, not Go. Rust would have the same cost.

**Expected improvement from Rust migration**: 0-2% in end-to-end latency, well
within measurement noise.

---

## When Would Rust Make Sense?

### Scenarios Favoring Migration

1. **A production-grade Rust inference server emerges** — if someone builds the
   equivalent of this project's continuous batching + model management on top of
   `llama-cpp-2`, migration becomes a swap rather than a rebuild.

2. **The binding-level features matter urgently** — if embeddings, LoRA, or
   model-native chat templates become critical before they're added to the Go
   bindings, one could argue for Rust. But adding these to Go is 1-2 weeks of
   work, not months.

3. **Team is Rust-native** — if the team writes Rust daily and Go is the
   unfamiliar language, the maintenance cost calculus flips.

4. **Extreme tail-latency requirements** — if p99 latency spikes from Go GC
   pauses (sub-millisecond) are unacceptable, Rust eliminates them. This is
   unlikely for LLM inference where token generation is ~10ms.

### Current Reality (March 2026)

| Condition | Met? |
|-----------|------|
| Production Rust inference server exists | ❌ |
| Binding features urgently needed in Rust only | ❌ (addable to Go) |
| Team is Rust-native | ❌ |
| GC pauses are a measured problem | ❌ |

**None of the trigger conditions are met.**

---

## What Changed Since October 2025

| Aspect | October 2025 | March 2026 |
|--------|-------------|------------|
| **This project** | gRPC-only, single-slot | gRPC + HTTP + OpenAI API, continuous batching, multi-slot, flash attn, Docker CI |
| **utilityai/llama-cpp-rs** | Moderate bindings | Comprehensive bindings (embeddings, LoRA, grammar, chat templates, 139 releases) |
| **mdrokz/rust-llama.cpp** | Early stage | Abandoned (no commits since Jun 2024) |
| **edgenai/llama_cpp-rs** | Active | Abandoned (no commits since Jun 2024) |
| **Binding gap** | Significant (Go ahead) | Narrowed (Rust ahead on some features) |
| **Server gap** | Large (Go ahead) | Wider (Go added much more server infrastructure) |
| **Net assessment** | Stay with Go | Stay with Go (stronger conviction) |

---

## Conclusion

The Rust llama.cpp binding ecosystem has matured impressively — `llama-cpp-2`
is well-maintained, feature-rich, and tracks llama.cpp releases closely. At the
binding level, it now leads on embeddings, LoRA, and chat template support.

However, **bindings are not a server**. This project's value lies in the
production infrastructure built on top of bindings: continuous batching,
multi-slot parallelism, model management, three API surfaces (gRPC, HTTP+SSE,
OpenAI-compatible), Docker CI/CD, and a comprehensive test suite. None of this
exists in the Rust ecosystem.

**Recommendation**: Stay with Go. Add the binding-level features that
`llama-cpp-2` has and this project lacks (embeddings, LoRA, model-native chat
templates) — these are already on the roadmap and represent days of work, not
months.

**Next review**: September 2026 (6 months), or sooner if a production-grade
Rust inference server emerges on top of `llama-cpp-2`.

---

**Document Version**: 2.0
**Last Updated**: March 19, 2026
**Previous Version**: [v0/go-vs-rust-llama-cpp-comparison.md](v0/go-vs-rust-llama-cpp-comparison.md) (October 7, 2025)

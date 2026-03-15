# Roadmap

Current state: the project supports dual gRPC + HTTP/SSE APIs, continuous batching
with multi-slot parallelism, pipeline and tensor parallelism, flash attention,
cross-platform builds (Windows/Linux/macOS), Docker CI/CD, and comprehensive
test coverage (baseline, greedy, seeded, stress, parallel, backpressure).
It is production-ready for CPU inference.

The next development directions focus on GPU support, API compatibility, and
production operations.

---

## 1. GPU Docker Images & Testing (high priority)

Docker images are currently CPU-only (`debian:bookworm-slim`). For production
GPU inference:

- **CUDA Docker image** — based on `nvidia/cuda`, linking against the CUDA
  llama.cpp release binaries
- **Vulkan Docker image** — for cross-vendor GPU support (AMD + NVIDIA) on Linux
- **Multi-GPU testing** — pipeline parallelism (`--split-mode layer`) and tensor
  parallelism (`--split-mode row`) are wired up but never tested under Docker/CI
- **GPU CI** — at minimum a self-hosted GPU runner; GitHub-hosted GPU runners
  are also an option

> **Note:** Vulkan is prioritized alongside CUDA because NVIDIA hardware is
> available in cloud, while AMD hardware is available locally. Vulkan covers both.

See [GPU_BUILD_STRATEGY.md](GPU_BUILD_STRATEGY.md) for the build matrix and
Docker variant strategy.

## 2. OpenAI-Compatible API

The current HTTP API works but uses a custom schema. An OpenAI-compatible layer
would enable drop-in use with existing client libraries:

- **`/v1/chat/completions`** — chat messages format, streaming via SSE with
  `choices[].delta`
- **`/v1/completions`** — raw completions (close to what already exists)
- **`/v1/models`** — list loaded models
- **Stop sequences** and **logprobs** — commonly used by client SDKs

This would let any OpenAI SDK client (Python `openai`, LangChain, LiteLLM, etc.)
connect directly. The `llmservice` layer already has the inference logic; it is
mostly API surface and response format work.

## 3. Observability & Production Operations

For real production deployments:

- **Prometheus metrics** — tokens/sec, request latency histograms, slot
  utilization, queue depth, KV cache usage, model load times
- **Structured JSON logging** — the current `SprintfLogger` works for
  development but production needs machine-parseable logs
- **Admin/status endpoint** — expose slot states, active requests, engine
  statistics (useful for load balancers and dashboards)
- **Request timeouts** — server-side deadline enforcement (not just client
  cancellation)

## 4. Advanced Inference Features

Features that llama.cpp supports but are not yet wired through:

- **Speculative decoding** — use a small draft model to accelerate generation
  from a large model (major throughput boost on GPU)
- **Prompt caching / prefix sharing** — when multiple requests share a common
  system prompt, reuse the KV cache prefix instead of re-processing it
- **Embeddings endpoint** — expose `llama_encode` for vector embeddings
  (useful for RAG pipelines)
- **Grammar-constrained generation** — GBNF grammars for structured output
  (JSON mode, function calling, etc.)

## 5. Multi-Model Improvements

The server already supports loading multiple models concurrently (the model
manager is keyed by path, and `Predict` routes to the correct model). However,
this has never been tested and has no resource governance:

- **Testing** — add multi-model integration tests (load two models, interleave
  requests)
- **Resource limits** — per-model memory budgets, maximum loaded models,
  eviction policy (LRU)
- **Per-model inference config** — currently all models share a single inference
  engine with fixed slot count and context size; allow per-model overrides
- **Model download** — download GGUF models from HuggingFace by name/URL
  instead of requiring local filesystem paths
- **Model info endpoint** — return metadata (parameter count, quantization,
  context length, etc.)

## 6. Deployment & Configuration

- **Helm chart / Kubernetes manifests** — GPU-aware scheduling, horizontal pod
  autoscaling based on slot utilization
- **Configuration file** — YAML/TOML config as an alternative to CLI flags
  (easier for complex deployments)
- **Graceful scaling** — drain slots before shutdown, request queuing with
  backpressure signals

---

## Recommended Sequence

| Phase | Focus | Rationale |
|-------|-------|-----------|
| **Next** | GPU Docker (CUDA + Vulkan) | Unlocks real production use cases |
| **Then** | OpenAI-compatible API | Highest adoption impact, moderate effort |
| **Then** | Prometheus metrics | Essential for operating in production |
| **Later** | Advanced features, multi-model, K8s | Driven by user demand |

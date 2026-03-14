# Inference Parallelism in llamacpp-server

This document describes the parallelism features available in llamacpp-server and how
they compare to other LLM inference solutions.

## Overview

llamacpp-server supports several forms of parallelism. They operate at different levels
and can be combined:

| Type | Level | Purpose | Hardware Requirement |
|------|-------|---------|---------------------|
| Thread parallelism | Intra-operator | Speed up a single matrix operation | Multi-core CPU |
| Pipeline parallelism (PP) | Inter-device | Split model layers across GPUs | Multi-GPU |
| Tensor parallelism (TP) | Inter-device | Split tensors across GPUs | Multi-GPU (same VRAM) |
| Continuous batching | Request-level | Process multiple requests in one forward pass | Any |
| Flash attention | Intra-operator | Fused attention kernel | Any (best on GPU) |

## Thread Parallelism

llama.cpp uses OpenMP (on Linux/macOS) or platform threading for parallelizing
matrix operations within a single decode call. Two thread counts are configurable:

- **`n_threads`** — threads used during autoregressive token generation (decoding one
  token at a time). Lower values may be sufficient since single-token decode is
  memory-bound.
- **`n_threads_batch`** — threads used during prompt processing (prefill). Higher values
  help here because batch prefill is compute-bound.

Setting either to `0` lets llama.cpp auto-detect based on available cores.

### Server flags

```
--threads N          Number of threads for generation (default: 0 = auto)
--threads-batch N    Number of threads for batch/prompt processing (default: 0 = auto)
```

### Guidance

- For CPU inference, `n_threads` ≈ number of performance cores is a good starting point.
- `n_threads_batch` can be set equal to or higher than `n_threads`.
- On systems with SMT/HyperThreading, using physical core count (not logical) often
  gives the best throughput.

## Pipeline Parallelism (PP)

Pipeline parallelism splits model layers across multiple GPUs. Each GPU holds a
contiguous subset of layers. During batch inference, while GPU 1 processes sub-batch A
on its layers, GPU 2 can process sub-batch B on its layers, forming a pipeline.

### How it works

1. The model is loaded with `n_gpu_layers` set to offload layers to GPU(s).
2. The layer-to-device assignment is controlled by `tensor_split` (proportional
   allocation) or defaults to even distribution.
3. For PP to actually pipeline (not just distribute), `n_batch` must be **larger**
   than `n_ubatch`:
   - `n_batch` — the total number of tokens in a logical batch.
   - `n_ubatch` — the micro-batch (compute batch) size for each device step.
   - When `n_batch > n_ubatch`, the batch is split into micro-batches that can be
     pipelined across devices.

### Server flags

```
--ngpu N             Number of GPU layers to offload (default: 99)
--split-mode layer   Layer-based split across GPUs (default)
--tensor-split P,P   Proportional GPU distribution (e.g. '0.6,0.4')
--batch-size N       Batch size / n_batch (default: 2048)
```

To enable pipeline parallelism with 2 GPUs, ensure `n_batch > n_ubatch` (the default
`n_ubatch` in llama.cpp is 512). With the server's default `batch-size=2048`, PP is
active when multiple GPUs are available.

### Performance notes

- PP scales nearly linearly with device count for batch processing.
- Single-token generation does not benefit from PP (there is no batch to pipeline).
- Setting `n_batch = n_ubatch` effectively disables PP.

## Tensor Parallelism (TP)

Tensor parallelism splits individual tensor operations across multiple GPUs. Each GPU
computes a portion of every layer, and results are combined via AllReduce.

### How it works

llama.cpp exposes TP via the `split_mode` model parameter:

- **`LLAMA_SPLIT_MODE_NONE` (0)** — use a single GPU only.
- **`LLAMA_SPLIT_MODE_LAYER` (1)** — split layers across GPUs (pipeline parallelism).
  This is the default.
- **`LLAMA_SPLIT_MODE_ROW` (2)** — split rows/tensors across GPUs. This is tensor
  parallelism. Each GPU computes a portion of every layer, then results are combined.

### Server flags

```
--split-mode row     Enable tensor parallelism (split rows across GPUs)
--tensor-split P,P   Proportional GPU distribution (e.g. '0.5,0.5')
--main-gpu N         Main GPU index for split-mode=none (default: 0)
--ngpu N             Number of GPU layers to offload (default: 99)
```

### Example usage

```bash
# Even 2-GPU tensor parallelism
llamacppserver --ngpu 99 --split-mode row --tensor-split 0.5,0.5

# Uneven split (GPU 0 has more VRAM than GPU 1)
llamacppserver --ngpu 99 --split-mode row --tensor-split 0.7,0.3

# Single GPU (disable multi-GPU)
llamacppserver --ngpu 99 --split-mode none --main-gpu 0
```

### Current limitations

As of llama.cpp b8323:

- TP works best with dense (non-MoE) models.
- Best results with 2 GPUs of equal VRAM (use `--tensor-split` to adjust for
  unequal GPUs).
- TP is not enabled by default because it is still considered experimental in upstream
  llama.cpp.
- Pipeline parallelism (`--split-mode layer`) remains the default and more mature
  approach for multi-GPU setups.

### When to use TP vs PP

| Scenario | Recommended |
|----------|-------------|
| 2 identical GPUs, latency-sensitive | TP (`--split-mode row`) |
| 2+ GPUs, throughput-oriented | PP (`--split-mode layer`) |
| Mixed GPU VRAM sizes | PP with `--tensor-split` |
| MoE models | PP only |
| Single GPU | `--split-mode none` |

## Continuous Batching

Continuous batching (also called dynamic batching) processes multiple concurrent
requests within a **single** llama.cpp context and forward pass. All requests share
one KV cache, with each request isolated by a unique sequence ID. A central scheduler
combines tokens from all active requests into a single `llama_decode` call per tick.

This is the same approach used by llama.cpp's own HTTP server and by production
inference engines like vLLM and SGLang.

### Benefits

- **Higher throughput** — one batched forward pass instead of N separate ones.
  GPU utilization scales with concurrency instead of being capped at 1/N.
- **Lower memory usage** — one shared KV cache instead of N independent caches.
  The scheduler right-sizes the cache and reuses freed slots.
- **Chunked prefill** — long prompts are split into chunks and interleaved with
  decode tokens from other requests, preventing latency spikes.
- **2x wall-clock speedup** measured on CPU with 4 concurrent requests
  (see [CONTINUOUS_BATCHING.md](CONTINUOUS_BATCHING.md) for details).

### Server flags

```
--n-parallel N       Number of concurrent inference slots (default: 1)
--ctx-size N         Total KV cache size; per-slot budget = ctx-size / n-parallel (default: 4096)
--batch-size N       Max tokens per decode call (default: 2048)
```

With `n_parallel=1` (default), inference is sequential — one request at a time.
Increase `--n-parallel` to enable concurrent request handling with batched decoding.

### Guidance

- **CPU-only**: Even `--n-parallel 2-4` provides measurable speedup because the
  batched `llama_decode` amortises per-call overhead.
- **GPU with ample VRAM**: Higher `--n-parallel` values (4-16) exploit GPU parallelism
  for near-linear throughput scaling.
- **Memory budget**: Total KV cache is `ctx-size` tokens, split evenly across slots.
  For F16 KV with a 7B model and 4096 total context, this is roughly 1-2 GB total.

For architecture details and benchmark results, see
[CONTINUOUS_BATCHING.md](CONTINUOUS_BATCHING.md).

## Flash Attention

Flash attention is a fused attention kernel that combines scale, mask, and softmax into
a single operation. It reduces memory bandwidth requirements and improves throughput,
especially for longer contexts.

### Server flags

```
--flash-attn         Enable flash attention (default: disabled)
```

### Guidance

- Enable flash attention for production workloads — it provides significant speedups
  with no quality impact.
- On CPU, flash attention still helps by reducing memory traffic.
- On GPU (CUDA, Metal), the speedup is more pronounced.
- Some very old hardware or unusual model architectures may not support it; in those
  cases, llama.cpp falls back gracefully.

## Context and Batch Size

### Server flags

```
--ctx-size N         Total KV cache across all slots (default: 4096)
--batch-size N       Batch size for prompt processing (default: 2048)
```

- `ctx-size` is the total KV cache budget. With `--n-parallel N`, each slot gets
  `ctx-size / N` tokens of context. Larger values support longer conversations but
  use more memory.
- `batch-size` controls how many tokens are processed at once during prompt prefill.
  Larger values improve prefill throughput but use more memory.

## Recommended Configurations

### CPU-only development/testing

```
llamacppserver --n-parallel 1 --ctx-size 2048 --batch-size 512 --threads 4
```

Conservative settings suitable for small models on a development machine.

### Single GPU production

```
llamacppserver --ngpu 99 --flash-attn --n-parallel 4 --ctx-size 4096 --batch-size 2048
```

Offload all layers to GPU, enable flash attention, allow 4 concurrent requests.

### Multi-GPU production (Pipeline Parallelism)

```
llamacppserver --ngpu 99 --flash-attn --split-mode layer --n-parallel 8 --ctx-size 8192 --batch-size 4096
```

With multiple GPUs, larger batch sizes enable pipeline parallelism. The model layers
are automatically distributed across available GPUs.

### Multi-GPU production (Tensor Parallelism)

```
llamacppserver --ngpu 99 --flash-attn --split-mode row --tensor-split 0.5,0.5 --n-parallel 4 --ctx-size 4096
```

With 2 identical GPUs and latency-sensitive workloads, tensor parallelism can reduce
per-request latency by splitting every layer across both GPUs.

## Comparison with Other Inference Solutions

llamacpp-server occupies a specific niche: a lightweight, self-contained Go server
with gRPC and HTTP+SSE interfaces, powered by llama.cpp's C inference engine. The
table below compares it with popular alternatives.

| Feature | llamacpp-server | llama.cpp server | vLLM | SGLang | TGI |
|---------|----------------|-----------------|------|--------|-----|
| **Language** | Go + C (CGO) | C++ | Python + C++ | Python + C++ | Rust + Python |
| **API** | gRPC + HTTP/SSE | HTTP (OpenAI-compat) | HTTP (OpenAI-compat) | HTTP (OpenAI-compat) | HTTP + gRPC |
| **Model formats** | GGUF | GGUF | HF, GPTQ, AWQ, GGUF | HF, GPTQ, AWQ | HF, GPTQ, AWQ |
| **Quantization** | GGUF (1.5–8 bit) | GGUF (1.5–8 bit) | GPTQ, AWQ, INT4/8 | GPTQ, AWQ, FP4/FP8 | GPTQ, AWQ |
| **Continuous batching** | Yes | Yes | Yes (PagedAttention) | Yes (RadixAttention) | Yes |
| **Pipeline parallelism** | Yes | Yes | Yes | Yes | Yes |
| **Tensor parallelism** | Yes (experimental) | Yes (experimental) | Yes | Yes | Yes |
| **Flash attention** | Yes | Yes | Yes | Yes | Yes |
| **CPU inference** | Excellent | Excellent | Limited | Limited | Limited |
| **GPU inference** | Good | Good | Excellent | Excellent | Excellent |
| **Dependencies** | Minimal (Go, llama.cpp) | Minimal (C++) | Python ecosystem, CUDA | Python ecosystem, CUDA | Rust toolchain, CUDA |
| **Best for** | Embedded, edge, gRPC integration | Local / CLI usage | GPU-heavy production | Agentic / chat workloads | Enterprise deployment |

### When to choose llamacpp-server

- **gRPC-native integration** — your stack already uses gRPC and you want native
  streaming without an HTTP adapter layer.
- **CPU or edge deployment** — llama.cpp is the most optimized engine for CPU inference
  and quantized models on resource-constrained hardware.
- **Minimal dependencies** — a single Go binary with no Python, no CUDA toolkit
  install, no virtual environments. Ideal for air-gapped or embedded environments.
- **GGUF model ecosystem** — you use quantized GGUF models from sources like
  Hugging Face or TheBloke.

### When to choose something else

- **Maximum GPU throughput** — vLLM's PagedAttention and SGLang's RadixAttention are
  heavily optimized for high-concurrency GPU workloads with HuggingFace model formats.
- **OpenAI API compatibility** — if your application expects an OpenAI-compatible
  endpoint, vLLM, SGLang, or llama.cpp's own server provide that out of the box.
- **Enterprise scale** — TGI and vLLM have more mature autoscaling, load balancing,
  and monitoring integrations for large-scale deployments.

## Testing Parallel Inference

The `llamacppclienttest` includes a `parallel` test mode that verifies the server handles
concurrent requests correctly:

```bash
# Run with default 4 concurrent requests
make run-paralleltest MODEL_PATH=/path/to/model.gguf

# Run with 8 concurrent requests
make run-paralleltest MODEL_PATH=/path/to/model.gguf PARALLEL_N=8

# Or directly:
./cmd/llamacppclienttest/llamacppclienttest --server ./cmd/llamacppserver/llamacppserver \
    --model /path/to/model.gguf --test-mode parallel --parallel-n 4
```

The parallel test:

1. Loads the model once on the server.
2. Launches N concurrent gRPC `Predict` requests with different prompts.
3. Waits for all to complete.
4. Reports per-request timing, token counts, and aggregate throughput.
5. Exits with code 0 if all requests succeeded, 1 if any failed.

Even on CPU-only systems, the test verifies that the server handles concurrent requests
without hangs, crashes, or data corruption.

## All Server Flags Reference

```
Model loading:
  --ngpu N             Number of GPU layers to offload (default: 99)
  --mmap               Use mmap for model loading
  --split-mode MODE    GPU split: none, layer (default), row (tensor parallelism)
  --main-gpu N         Main GPU index for split-mode=none (default: 0)
  --tensor-split P,P   GPU split proportions, comma-separated

Inference:
  --flash-attn         Enable flash attention
  --ctx-size N         Total KV cache across all slots (default: 4096)
  --batch-size N       Batch size for prompt processing (default: 2048)
  --threads N          Threads for generation, 0=auto (default: 0)
  --threads-batch N    Threads for batch processing, 0=auto (default: 0)

Concurrency:
  --n-parallel N       Number of concurrent inference slots (default: 1)

Network:
  --host ADDR          Bind address (default: 127.0.0.1)
  --grpc-port PORT     gRPC listen port (default: 50052)
  --http-port PORT     HTTP+SSE listen port (default: 8082)
```

## Further Reading

- [Continuous Batching — Architecture and Benchmarks](CONTINUOUS_BATCHING.md)
- [llama.cpp Pipeline Parallelism PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017)
- [llama.cpp Tensor Parallelism Discussion](https://github.com/ggml-org/llama.cpp/discussions/20252)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)
- [Flash Attention PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)
- [vLLM Project](https://github.com/vllm-project/vllm)
- [SGLang Project](https://github.com/sgl-project/sglang)

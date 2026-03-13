# Inference Parallelism in llama.cpp

This document describes the parallelism features available in llama.cpp and how they
are exposed in the llamacpp-server gRPC server.

## Overview of Parallelism Types

llama.cpp supports several forms of parallelism. They operate at different levels and
can be combined:

| Type | Level | Purpose | Hardware Requirement |
|------|-------|---------|---------------------|
| Thread parallelism | Intra-operator | Speed up a single matrix operation | Multi-core CPU |
| Pipeline parallelism (PP) | Inter-device | Split model layers across GPUs | Multi-GPU |
| Tensor parallelism (TP) | Inter-device | Split tensors across GPUs | Multi-GPU (same VRAM) |
| Continuous batching | Request-level | Process multiple requests concurrently | Any |
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

### Status in llamacpp-server

PP is automatically enabled by llama.cpp when the model is distributed across multiple
GPUs via `--ngpu`. The server exposes:

```
--ngpu N             Number of GPU layers to offload (default: 99)
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

### Status

As of llama.cpp b8292, TP is available via the "split mode tensor" implementation
(merged in early 2026 via PR #19378). Current limitations:

- Only works with dense (non-MoE) models.
- Requires exactly 2 GPUs with equal VRAM.
- Must be enabled with `--split-mode tensor` when using llama.cpp's own server.

### Status in llamacpp-server

The Go bindings expose `SetTensorSplit()` on `ModelParams` for proportional GPU
distribution. Full TP split-mode configuration is not yet directly exposed via server
flags, as TP is still considered experimental in upstream llama.cpp.

If you need TP, set `tensor_split` proportions programmatically or use llama.cpp's
native tools to benchmark before integrating.

## Concurrent Request Handling (Slots)

llamacpp-server supports handling multiple concurrent inference requests. Each request
gets its own llama.cpp context (with independent KV cache), while the loaded model is
shared read-only across all contexts.

### How it works

1. A gRPC `Predict` request arrives.
2. The predictions manager checks the concurrency limit.
3. If a slot is available, a new llama.cpp context is created for this request.
4. The prediction runs independently of other in-flight predictions.
5. On completion, the context is freed and the slot is released.

### Server flags

```
--n-parallel N       Max concurrent inference requests (default: 0 = unlimited)
```

- `0` means no limit — the server will accept and run as many concurrent requests as
  arrive. This is suitable when requests are infrequent or the system has ample
  resources.
- Setting `N > 0` limits concurrent predictions via a semaphore. Excess requests queue
  and wait for a slot to become available.

### Guidance

- **CPU-only**: Set `--n-parallel 1` or a small number. Multiple concurrent contexts
  on CPU will compete for cores and cause thread contention, reducing per-request
  throughput. However, the server will still correctly handle queued requests.
- **GPU with ample VRAM**: Higher `--n-parallel` values work well. Each context
  allocates its own KV cache, so VRAM usage scales with `n_parallel * ctx_size`.
- **Memory budget**: Each slot allocates `ctx_size` tokens of KV cache. For F16 KV
  with a 7B model and 4096 context, this is roughly 1-2 GB per slot.

### Difference from llama.cpp server

llama.cpp's own HTTP server (`llama-server`) uses a single shared context with
multiple "slots" and continuous batching to process multiple requests in a single
`llama_decode` call. llamacpp-server uses separate contexts per request, which is
simpler but does not benefit from shared-context continuous batching.

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
--ctx-size N         Context window size per slot (default: 4096)
--batch-size N       Batch size for prompt processing (default: 2048)
```

- `ctx-size` determines the maximum sequence length each inference slot can handle.
  Larger values support longer conversations but use more memory per slot.
- `batch-size` controls how many tokens are processed at once during prompt prefill.
  Larger values improve prefill throughput but use more memory.

## Recommended Configurations

### CPU-only development/testing

```
grpcserver --n-parallel 1 --ctx-size 2048 --batch-size 512 --threads 4
```

Conservative settings suitable for small models on a development machine.

### Single GPU production

```
grpcserver --ngpu 99 --flash-attn --n-parallel 4 --ctx-size 4096 --batch-size 2048
```

Offload all layers to GPU, enable flash attention, allow 4 concurrent requests.

### Multi-GPU production (Pipeline Parallelism)

```
grpcserver --ngpu 99 --flash-attn --n-parallel 8 --ctx-size 8192 --batch-size 4096
```

With multiple GPUs, larger batch sizes enable pipeline parallelism. The model layers
are automatically distributed across available GPUs.

## Testing Parallel Inference

The `grpcclienttest` includes a `parallel` test mode that verifies the server handles
concurrent requests correctly:

```bash
# Run with default 4 concurrent requests
make run-paralleltest MODEL_PATH=/path/to/model.gguf

# Run with 8 concurrent requests
make run-paralleltest MODEL_PATH=/path/to/model.gguf PARALLEL_N=8

# Or directly:
./cmd/grpcclienttest/grpcclienttest --server ./cmd/grpcserver/grpcserver \
    --model /path/to/model.gguf --test-mode parallel --parallel-n 4
```

The parallel test:

1. Loads the model once on the server.
2. Launches N concurrent gRPC `Predict` requests with different prompts.
3. Waits for all to complete.
4. Reports per-request timing, token counts, and aggregate throughput.
5. Exits with code 0 if all requests succeeded, 1 if any failed.

Even on CPU-only systems, the test verifies that the server handles concurrent requests
without hangs, crashes, or data corruption. Requests may be serialized or run in
parallel depending on server configuration and system resources.

## Further Reading

- [llama.cpp Pipeline Parallelism PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017)
- [llama.cpp Tensor Parallelism Discussion](https://github.com/ggml-org/llama.cpp/discussions/20252)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md)
- [Flash Attention PR #5021](https://github.com/ggml-org/llama.cpp/pull/5021)

# GPU Build Strategy

How to organize building and testing the project across multiple GPU platforms
without an explosion of OS × GPU build variants.

## Key Insight: The Go Code Never Changes

The critical realization is that the Go source, CGO bindings, and the
`libllama` API are **identical** across all GPU variants. What differs is only:

1. **Which llama.cpp binary package is downloaded** (`make prepare`)
2. **Which GPU backend `.so` files ship at runtime** (`libggml-vulkan.so`, etc.)
3. **Which Docker base image provides GPU drivers** (runtime stage only)

The Go binary links against `libllama.so` + `libggml.so` + `libggml-base.so`,
which expose the same ABI regardless of whether the build includes CUDA, Vulkan,
or CPU-only backends. GPU backends are loaded as **dynamic plugins** at runtime.

This means: one `make build` per platform, with `make prepare GPU_VARIANT=X`
selecting which llama.cpp binaries to download.

## llama.cpp Release Matrix

From a typical [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b8292):

| Platform | Variant | Archive name |
|----------|---------|--------------|
| **Linux x64** | CPU | `llama-bXXXX-bin-ubuntu-x64.tar.gz` |
| **Linux x64** | Vulkan (NVIDIA + AMD) | `llama-bXXXX-bin-ubuntu-vulkan-x64.tar.gz` |
| **Linux x64** | ROCm 7.2 (AMD optimized) | `llama-bXXXX-bin-ubuntu-rocm-7.2-x64.tar.gz` |
| **macOS arm64** | Metal (built-in) | `llama-bXXXX-bin-macos-arm64.tar.gz` |
| **macOS x64** | CPU | `llama-bXXXX-bin-macos-x64.tar.gz` |
| **Windows x64** | CPU | `llama-bXXXX-bin-win-cpu-x64.zip` |
| **Windows x64** | CUDA 12.4 | `llama-bXXXX-bin-win-cuda-12.4-x64.zip` |
| **Windows x64** | CUDA 13.1 | `llama-bXXXX-bin-win-cuda-13.1-x64.zip` |
| **Windows x64** | Vulkan (NVIDIA + AMD) | `llama-bXXXX-bin-win-vulkan-x64.zip` |
| **Windows x64** | HIP/ROCm (AMD) | `llama-bXXXX-bin-win-hip-radeon-x64.zip` |
| **Windows x64** | SYCL (Intel) | `llama-bXXXX-bin-win-sycl-x64.zip` |

**Notable:** There is no Linux CUDA binary. On Linux with NVIDIA GPUs, the
**Vulkan** build is used (Vulkan works on NVIDIA, AMD, and Intel GPUs).
On Windows, dedicated CUDA builds exist because CUDA outperforms Vulkan there.

macOS Apple Silicon includes Metal support automatically — no separate variant.

## What We Actually Need

Not all 20+ release assets are relevant. The practical matrix:

### Native Builds (Local Development)

Build for **your** platform only. The Makefile auto-detects OS and GPU:

| Your setup | What `make prepare` downloads |
|------------|-------------------------------|
| Linux + no GPU | ubuntu-x64 (CPU) |
| Linux + NVIDIA | ubuntu-vulkan-x64 |
| Linux + AMD | ubuntu-vulkan-x64 (or ROCm if optimized AMD is needed) |
| macOS Apple Silicon | macos-arm64 (Metal built-in) |
| Windows + no GPU | win-cpu-x64 |
| Windows + NVIDIA | win-cuda-12.4-x64 (auto-detected via `nvcc`) |
| Windows + AMD | win-vulkan-x64 |

You never build all variants locally. Just `make all` on your machine.

### Docker Images (Production)

Docker is Linux-only, so the matrix is compact — only **3 variants**:

| Variant | llama.cpp binary | Base image | GPU access |
|---------|------------------|------------|------------|
| `cpu` | ubuntu-x64 | `debian:bookworm-slim` | None |
| `vulkan` | ubuntu-vulkan-x64 | `debian:bookworm-slim` + `libvulkan1` | `--device /dev/dri` |
| `rocm` | ubuntu-rocm-7.2-x64 | `rocm/dev-ubuntu-22.04:6.x` | `--device /dev/kfd --device /dev/dri` |

The Vulkan image works with **both** NVIDIA and AMD GPUs, making it the
recommended default for GPU inference in Docker.

### CI Matrix

| Job | Platform | GPU | Purpose |
|-----|----------|-----|---------|
| Build + test | Ubuntu | CPU | Functional correctness (current) |
| Build + test | Windows | CPU | Cross-platform (current) |
| Build + test | macOS | CPU/Metal | Cross-platform (current) |
| Build only | Ubuntu | Vulkan | Verify Vulkan Docker image builds |
| Build only | Ubuntu | ROCm | Verify ROCm Docker image builds |
| Build + test | GPU runner | Vulkan or CUDA | GPU inference correctness (when available) |

**"Build only" means**: `docker build` succeeds and the server starts, but no
GPU inference test (requires actual GPU hardware on the runner).

## Implementation

### Phase 1: `GPU_VARIANT` Makefile Variable (done)

`GPU_VARIANT` controls which llama.cpp binary package is downloaded:

```makefile
GPU_VARIANT ?= auto   # auto, cpu, vulkan, rocm, cuda12, cuda13
```

- `auto` (default): detects `nvcc` → resolves to `cuda12` (Windows) or
  `vulkan` (Linux); falls back to `cpu`
- Explicit values override auto-detection

### Phase 1b: Variant-Specific Directories & Image Tags (done)

**Download directories** are tagged by variant so multiple variants coexist
without re-downloading:

```
build/
├── llama-binaries-cpu/       # make prepare GPU_VARIANT=cpu
├── llama-binaries-vulkan/    # make prepare GPU_VARIANT=vulkan
└── llama-binaries-rocm/      # make prepare GPU_VARIANT=rocm
```

A stable symlink/junction `build/llama-binaries` always points to the active
variant — this is what the CGO directives in `internal/bindings/llamacpp.go`
reference. Running `make build GPU_VARIANT=X` automatically re-points the
symlink and rebuilds (~5s re-link). No `make clean` or re-download needed.

**Docker image tags** always include the variant as a suffix:

| GPU_VARIANT | Docker tag |
|-------------|------------|
| `cpu` | `llamacpp-server:latest-cpu` |
| `vulkan` | `llamacpp-server:latest-vulkan` |
| `rocm` | `llamacpp-server:latest-rocm` |
| `cuda12` | `llamacpp-server:latest-cuda12` |

**Clean targets** for managing downloaded binaries:

```bash
make clean-prepare                     # remove current variant's binaries
make clean-prepare GPU_VARIANT=vulkan  # remove vulkan binaries
make clean-prepare-all                 # remove ALL variant binaries
make clean                             # remove everything (all variants + Go binaries)
make docker-clean                      # remove all Docker images (all variant tags)
```

### Phase 2: GPU-Specific Runtime Images (future)

Multi-stage Dockerfile with variant-specific runtime base images:

```dockerfile
FROM debian:bookworm-slim AS runtime-cpu
RUN apt-get update && apt-get install -y libgomp1 ca-certificates ...

FROM debian:bookworm-slim AS runtime-vulkan
RUN apt-get update && apt-get install -y libgomp1 ca-certificates libvulkan1 mesa-vulkan-drivers ...

FROM rocm/dev-ubuntu-22.04:6.3 AS runtime-rocm
RUN ...

FROM runtime-${GPU_VARIANT} AS runtime
COPY --from=builder /app/cmd/llamacppserver/llamacppserver /app/
COPY --from=builder /app/build/llama-binaries-${GPU_VARIANT}/lib/ /app/
```

### Phase 3: Docker Compose GPU Profiles (future)

```yaml
services:
  server-cpu:
    build:
      args:
        GPU_VARIANT: cpu

  server-vulkan:
    build:
      args:
        GPU_VARIANT: vulkan
    devices:
      - /dev/dri:/dev/dri

  server-rocm:
    build:
      args:
        GPU_VARIANT: rocm
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
```

### Phase 4: CI Workflow (future)

```yaml
strategy:
  matrix:
    include:
      - os: ubuntu-latest
        gpu: cpu
        test: true
      - os: ubuntu-latest
        gpu: vulkan
        test: false        # build-only, no GPU on standard runners
      - os: ubuntu-latest
        gpu: rocm
        test: false        # build-only
      - os: windows-latest
        gpu: cpu
        test: true
      - os: macos-latest
        gpu: cpu
        test: true
```

GPU testing can be added later with self-hosted runners or GitHub GPU runners.

## Usage Examples

```bash
# Download multiple variants (done once, kept side-by-side)
make prepare GPU_VARIANT=cpu
make prepare GPU_VARIANT=vulkan

# Switch freely — no re-download, just re-link (~5s)
make build GPU_VARIANT=cpu
make build GPU_VARIANT=vulkan

# Docker images with variant tags
make docker-build-server                       # → llamacpp-server:latest-cpu
make docker-build-server GPU_VARIANT=vulkan    # → llamacpp-server:latest-vulkan
make docker-build-server GPU_VARIANT=rocm      # → llamacpp-server:latest-rocm

# Clean specific variant or all
make clean-prepare GPU_VARIANT=vulkan          # remove only vulkan binaries
make clean-prepare-all                         # remove all downloaded binaries
make docker-clean                              # remove all Docker images
```

## Summary: Why It's Not 20 Builds

| Concern | Reality |
|---------|---------|
| "20 llama.cpp release variants" | We use 3 for Docker (cpu, vulkan, rocm) + 1 per native dev machine |
| "OS × GPU combinatorial explosion" | Docker is Linux-only → 3 variants. Native auto-detects → 1 build |
| "Need CGO rebuild per GPU?" | Same Go code, same API. Only `make prepare` (binary download) differs |
| "Expensive to test on all platforms" | Build-only CI for GPU variants; GPU tests only on GPU runners |
| "Multiple Dockerfiles?" | Single parameterized Dockerfile with multi-stage runtime selection |
| "Switching variants is slow" | Variant dirs coexist — switching is just `make build` (~5s re-link) |

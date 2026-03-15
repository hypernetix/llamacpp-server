# llama.cpp Go Server

A Go server for running Large Language Models locally using **llama.cpp** with
dual **gRPC** and **HTTP+SSE** interfaces.

## Features

- **Dual API**: gRPC (Protobuf streaming) and HTTP+SSE interfaces — use either or both simultaneously
- **Streaming Inference**: Real-time token-by-token generation via gRPC server streaming or Server-Sent Events
- **Continuous Batching**: Shared-context inference engine that processes multiple concurrent requests in a single batched forward pass
- **Parallel Inference Slots**: Configurable `--n-parallel` to serve multiple requests concurrently with efficient KV cache sharing
- **Model Management**: Automatic loading and caching of GGUF models with progress reporting
- **GPU Acceleration**: CUDA (Windows/Linux), Metal (macOS), and Vulkan support
- **Multi-GPU**: Pipeline parallelism (`--split-mode layer`) and tensor parallelism (`--split-mode row`)
- **Cross-Platform**: Windows, Linux, and macOS — native builds and Docker images
- **CI/CD**: GitHub Actions workflow with automated multi-platform builds and integration tests

## Quick Start

```bash
# Full build: download llama.cpp binaries + build all Go executables
make all

# Run a baseline inference test (spawns server, loads model, runs inference)
make run-baselinetest MODEL_PATH=/path/to/your/model.gguf
```

### Quick Start with Docker

```bash
# Run integration test with your model
make docker-integration-test MODEL_PATH=/path/to/your/model.gguf

# Or run CI-style test (downloads a small test model automatically)
make docker-integration-test-ci
```

## Prerequisites

### Native Build

- **Go** 1.22 or later
- **Make** (GNU Make)
- **GCC/MinGW** — C compiler for CGO
  - Windows: MinGW-w64 via MSYS2 (includes `gendef`, `dlltool` for import libraries)
  - Linux: `build-essential` package
  - macOS: Xcode command line tools

### Docker Build

- **Docker** with Docker Compose v2
- No other dependencies required

### Windows-specific Setup

Install MSYS2 and required tools:

```powershell
# Install MSYS2 from https://www.msys2.org/
# Then in MSYS2 terminal:
pacman -S mingw-w64-x86_64-toolchain mingw-w64-x86_64-tools-git

# Add to PATH: C:\msys64\mingw64\bin
```

### macOS-specific Setup

```bash
# Install libomp (required for OpenMP support in llama.cpp)
brew install libomp

# Install Go (if not already installed)
brew install go
```

## Build System

### Main Targets

| Target | Description |
|--------|-------------|
| `make all` | Full build: download binaries + build all Go executables |
| `make prepare` | Download llama.cpp binaries + generate import libraries (Windows) |
| `make build` | Build all Go executables (assumes `prepare` was run) |
| `make clean` | Remove all build artifacts |
| `make help` | Show all available targets |

### Individual Build Targets

```bash
make build-llamacppserver      # Build the server
make build-llamacppclienttest  # Build the client test tool
make build-inferencetest1      # Build low-level inference test 1
make build-inferencetest2      # Build low-level inference test 2
```

### Run Targets

```bash
# Start server with default ports (gRPC 50052, HTTP 8082)
make run-llamacppserver

# Start server on custom ports
make run-llamacppserver GRPC_PORT=50053
make run-llamacppserver HTTP_PORT=8083
make run-llamacppserver GRPC_PORT=50053 HTTP_PORT=8083

# Run tests (MODEL_PATH required)
make run-baselinetest MODEL_PATH=/path/to/model.gguf                  # spawn server, test via gRPC
make run-baselinetest MODEL_PATH=/path/to/model.gguf TRANSPORT=http   # spawn server, test via HTTP
make run-paralleltest MODEL_PATH=/path/to/model.gguf                  # 4-slot concurrent inference test
make run-backpressuretest MODEL_PATH=/path/to/model.gguf              # oversubscription test (2N requests for N slots)

# Attach to an already running server
make run-baselinetest SERVER_PATH='' ATTACH_PORT=50052 MODEL_PATH=/path/to/model.gguf
make run-baselinetest SERVER_PATH='' ATTACH_PORT=8082 TRANSPORT=http MODEL_PATH=/path/to/model.gguf

# Low-level inference tests (no server, direct llama.cpp bindings)
make run-inferencetest1 MODEL_PATH=/path/to/model.gguf
make run-inferencetest2 MODEL_PATH=/path/to/model.gguf
```

### Docker Targets

| Target | Description |
|--------|-------------|
| `make docker-build` | Build all Docker images (server + client) |
| `make docker-build-server` | Build server Docker image |
| `make docker-build-client` | Build client test Docker image |
| `make docker-integration-test MODEL_PATH=<path>` | Run integration test with local model |
| `make docker-integration-test-ci` | Run integration test (downloads test model) |
| `make docker-clean` | Remove Docker images and volumes |

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_VERSION` | *(see Makefile)* | llama.cpp release version to download |
| `GPU_VARIANT` | `auto` | GPU variant: `auto`, `cpu`, `vulkan`, `rocm`, `cuda12`, `cuda13` |
| `GRPC_PORT` | `50052` | gRPC server port |
| `HTTP_PORT` | `8082` | HTTP+SSE server port |
| `MODEL_PATH` | *(none)* | Path to GGUF model file (required for tests) |
| `ATTACH_PORT` | `0` | Port of already running server (0 = spawn server) |
| `TRANSPORT` | `grpc` | Transport protocol: `grpc` or `http` |
| `SERVER_PATH` | *(auto)* | Path to server executable |
| `IMAGE_TAG` | `latest` | Docker image tag |

### Using Different llama.cpp Versions

```bash
# Download specific version
make LLAMA_VERSION=b6800 prepare

# Full build with specific version
make LLAMA_VERSION=b6800 all

# Docker build with specific version
make docker-build LLAMA_VERSION=b6800
```

## Project Structure

```
.
├── Makefile                    # Unified build system
├── api/
│   ├── proto/                  # gRPC / Protobuf API definition
│   │   ├── llmserver.proto
│   │   ├── llmserver.pb.go     # Generated Go code
│   │   └── llmserver_grpc.pb.go
│   └── http/                   # HTTP+SSE API definition
│       └── openapi.yaml        # OpenAPI 3.1 specification
├── cmd/
│   ├── llamacppserver/         # Server application (gRPC + HTTP)
│   ├── llamacppclienttest/     # Client test tool (gRPC + HTTP)
│   │   └── llmservice/         # Server lifecycle & transport clients
│   ├── inferencetest1/         # Low-level inference test 1
│   └── inferencetest2/         # Low-level inference test 2
├── internal/
│   ├── bindings/               # CGO bindings to llama.cpp C API
│   ├── inferenceengine/        # Continuous batching scheduler, slots, sampler
│   ├── llmservice/             # Transport-agnostic service layer
│   ├── grpcserver/             # gRPC server implementation
│   ├── httpserver/             # HTTP+SSE server implementation
│   ├── modelmanagement/        # Model loading and caching
│   └── logging/                # Structured logging
├── docker/
│   ├── Dockerfile.server       # Server Docker image
│   ├── Dockerfile.client       # Client test Docker image
│   ├── docker-compose.yml      # Local integration testing
│   └── docker-compose.ci.yml   # CI integration testing
├── scripts/
│   ├── integration-test.sh     # Integration test runner (Linux/macOS)
│   └── integration-test.ps1   # Integration test runner (Windows)
├── docs/
│   ├── PARALLELISM.md          # Parallelism modes and comparison with other solutions
│   └── CONTINUOUS_BATCHING.md  # Continuous batching architecture and benchmarks
└── .github/
    └── workflows/
        └── ci.yml              # GitHub Actions CI workflow
```

## Usage

### Server

```bash
# Start with both gRPC and HTTP interfaces (native defaults: 50052 / 8082)
./cmd/llamacppserver/llamacppserver --grpc-port 50052 --http-port 8082

# Start with 4 parallel inference slots and flash attention
./cmd/llamacppserver/llamacppserver --grpc-port 50052 --http-port 8082 --n-parallel 4 --flash-attn

# Bind to all interfaces (for Docker / remote access)
./cmd/llamacppserver/llamacppserver --host 0.0.0.0 --grpc-port 50052 --http-port 8082

# Or via make
make run-llamacppserver
```

#### Server Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host address to bind (use `0.0.0.0` for Docker/remote) |
| `--grpc-port` | `50052` | gRPC server port (disabled if empty) |
| `--http-port` | `8082` | HTTP+SSE server port (disabled if empty) |
| `--ngpu` | `99` | Number of GPU layers to offload |
| `--mmap` | `false` | Use memory-mapped I/O for model loading |
| `--flash-attn` | `false` | Enable flash attention for faster inference |
| `--n-parallel` | `1` | Number of concurrent inference slots |
| `--ctx-size` | `4096` | Total KV cache size (per-slot budget = ctx-size / n-parallel) |
| `--batch-size` | `2048` | Batch size for prompt processing |
| `--threads` | `0` | Threads for token generation (0 = auto) |
| `--threads-batch` | `0` | Threads for batch/prompt processing (0 = auto) |
| `--split-mode` | `layer` | Multi-GPU split: `none`, `layer` (pipeline), `row` (tensor parallelism) |
| `--main-gpu` | `0` | Main GPU index when `split-mode=none` |
| `--tensor-split` | *(empty)* | GPU split proportions, comma-separated (e.g. `0.5,0.5`) |

### Client Test

```bash
# Test against running gRPC server
./cmd/llamacppclienttest/llamacppclienttest --host 127.0.0.1 --port 50052 --transport grpc --model /path/to/model.gguf

# Test against running HTTP server
./cmd/llamacppclienttest/llamacppclienttest --host 127.0.0.1 --port 8082 --transport http --model /path/to/model.gguf

# Or via make
make run-baselinetest MODEL_PATH=/path/to/model.gguf
```

#### Client Test Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Server host address |
| `--port` | `0` | Server port (0 = spawn a new server) |
| `--transport` | `grpc` | Transport protocol: `grpc` or `http` |
| `--server` | *(auto)* | Path to server executable |
| `--model` | *(none)* | Path to GGUF model file (required) |
| `--test-mode` | `baseline` | Test mode (see below) |
| `--max-tokens` | `100` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `1.0` | Top-p (nucleus) sampling |
| `--top-k` | `0` | Top-k sampling (0 = disabled) |
| `--min-p` | `0.05` | Min-p sampling threshold |
| `--repeat-penalty` | `1.0` | Repetition penalty (1.0 = disabled) |
| `--seed` | `-1` | Random seed (-1 = random) |
| `--parallel-n` | `4` | Concurrent requests for parallel/backpressure modes |

#### Test Modes

| Mode | Description |
|------|-------------|
| `baseline` | Single inference request with default sampling |
| `greedy` | Deterministic inference (temperature=0) |
| `seeded` | Two runs with the same seed — verifies identical output |
| `stress` | Sequential multi-prompt stress test |
| `parallel` | Concurrent multi-slot inference test |
| `backpressure` | Sends 2N requests to N slots — verifies all complete under oversubscription |

### Model Requirements

- **Format**: GGUF models (e.g., `model.gguf`)
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, and other GGUF quantizations supported
- **Sources**: [Hugging Face](https://huggingface.co/models?search=gguf)

## API Documentation

The server exposes two equivalent interfaces. Both share the same inference engine and model state.

### gRPC API

Defined in [`api/proto/llmserver.proto`](api/proto/llmserver.proto).

| RPC | Description |
|-----|-------------|
| `Ping` | Health check |
| `LoadModel` | Load a GGUF model with streaming progress |
| `Predict` | Generate text with streaming token output |

### HTTP+SSE API

Defined in [`api/http/openapi.yaml`](api/http/openapi.yaml) (OpenAPI 3.1).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | `GET` | Health check |
| `/models/load` | `POST` | Load a GGUF model — returns SSE progress stream |
| `/completions` | `POST` | Generate text — streaming (SSE) or non-streaming JSON |

## Docker

### Building Docker Images

```bash
# Build both server and client images
make docker-build

# Or build individually
docker build -f docker/Dockerfile.server -t llamacpp-server .
docker build -f docker/Dockerfile.client -t llamacpp-client .
```

### Running Integration Tests

```bash
# With a local model file
make docker-integration-test MODEL_PATH=/path/to/model.gguf

# CI mode (downloads SmolLM2-135M test model automatically)
make docker-integration-test-ci

# Using scripts directly
./scripts/integration-test.sh --model /path/to/model.gguf
./scripts/integration-test.sh --ci

# Windows PowerShell
.\scripts\integration-test.ps1 -Model C:\path\to\model.gguf
.\scripts\integration-test.ps1 -CI
```

#### Integration Test Options

| Option | Description |
|--------|-------------|
| `--model PATH` | Path to local GGUF model file |
| `--model-url URL` | URL to download the model from |
| `--ci` | Use CI defaults (downloads SmolLM2-135M) |
| `--test-mode MODE` | Test mode: baseline, greedy, seeded, stress |
| `--no-cleanup` | Don't remove containers after test |
| `--build` | Force rebuild Docker images |
| `--verbose` | Show verbose output |

### Running Server Standalone

```bash
# Run server container (exposes gRPC on 50051, HTTP on 8080)
docker run -p 50051:50051 -p 8080:8080 \
  -v /path/to/model.gguf:/models/model.gguf:ro \
  llamacpp-server

# Or using docker-compose
MODEL_PATH=/path/to/model.gguf docker compose -f docker/docker-compose.yml up server-only
```

### Using in Other Projects

```yaml
# In your project's docker-compose.yml
services:
  llm-server:
    image: llamacpp-server:latest
    ports:
      - "50051:50051"   # gRPC
      - "8080:8080"     # HTTP+SSE
    volumes:
      - ${MODEL_PATH}:/models/model.gguf:ro

  your-service:
    build: .
    depends_on:
      llm-server:
        condition: service_healthy
    environment:
      - LLM_SERVER_HOST=llm-server
      - LLM_SERVER_GRPC_PORT=50051
      - LLM_SERVER_HTTP_PORT=8080
```

> **Note:** Docker images use ports 50051 (gRPC) and 8080 (HTTP) by default,
> while native builds default to 50052 and 8082.

## Build Details

### What `make prepare` Does

1. **Downloads binaries**: Official pre-built llama.cpp release for your platform
2. **Downloads headers**: Matching source code to extract header files
3. **Organizes files**: Creates `bin/`, `lib/`, `include/` structure
4. **Generates import libraries** (Windows only): Creates `.dll.a` files from DLLs using `gendef`/`dlltool`

### What Docker Build Does

The Dockerfiles use the same Makefile for consistency:

```dockerfile
# Uses Makefile to prepare llama.cpp binaries
RUN make prepare LLAMA_VERSION=${LLAMA_VERSION}

# Uses Makefile to build the server
RUN make build-llamacppserver
```

This ensures Docker builds are identical to local builds.

### Platform-Specific Notes

#### Windows

- **CUDA Support**: Automatically detected if `nvcc` is in PATH
- **Import Libraries**: Required for linking — generated automatically by `make prepare`
- **Runtime DLLs**: Automatically copied to executable directories by run targets

#### macOS

- **Apple Silicon (M1/M2/M3/M4)**: Metal backend enabled automatically
- **Intel Macs**: CPU-only by default
- **libomp**: Required for OpenMP support — install with `brew install libomp`

#### Linux

- **CUDA**: Detected automatically if `nvcc` is available
- **Vulkan**: Binary download uses Vulkan build for cross-vendor GPU support

#### Docker

- **Base Image**: `debian:bookworm-slim` for minimal size
- **Architecture**: Linux x64 only (for now)
- **Dependencies**: Only `libgomp1` and `ca-certificates` at runtime

## Further Reading

- [Parallelism Modes](docs/PARALLELISM.md) — thread, pipeline, and tensor parallelism; comparison with vLLM, SGLang, TGI
- [Continuous Batching](docs/CONTINUOUS_BATCHING.md) — shared-context architecture, benchmarks, and future work
- [Roadmap](docs/ROADMAP.md) — planned features and development directions
- [GPU Build Strategy](docs/GPU_BUILD_STRATEGY.md) — GPU variant builds, Docker images, and CI matrix

## CI/CD

The project includes GitHub Actions CI that runs on every push:

1. **Build & Test**: Builds on Ubuntu, Windows, and macOS
2. **Docker Integration Test**: Runs gRPC, HTTP, and parallel inference tests
3. **Artifacts**: Uploads built binaries for each platform

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details.

## License

This project is licensed under the MIT License.

### Component Licenses

- **llama.cpp**: MIT License
- **gRPC-Go**: Apache 2.0 License
- **Protocol Buffers**: BSD 3-Clause License

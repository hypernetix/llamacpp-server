# llama.cpp Go gRPC Server

A Go project for serving Large Language Models locally using **LLaMA.cpp** with gRPC interface.

## Features

- **gRPC Interface**: Clean API for model loading and text generation
- **Streaming Support**: Real-time text generation with streaming responses
- **Model Management**: Automatic loading and caching of GGUF models
- **Cross-platform**: Windows, Linux, and macOS support
- **GPU Acceleration**: CUDA (Windows/Linux) and Metal (macOS) support
- **Docker Support**: Ready-to-use Docker images for containerized deployment
- **CI/CD**: GitHub Actions workflow with automated testing

## Quick Start

```bash
# Full build: download binaries + build all Go executables
make all

# Run client test, it will run, connect and send test request to gRPC server with a specified model
make run-grpcclienttest MODEL_PATH=/path/to/your/model.gguf
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
- **GCC/MinGW** - C compiler for CGO
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

Install Homebrew and required dependencies:
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

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
make build-grpcserver      # Build gRPC server
make build-grpcclienttest  # Build gRPC client test
make build-inferencetest1  # Build inference test 1
make build-inferencetest2  # Build inference test 2
```

### Run Targets

```bash
# Start gRPC server (default port 50052)
make run-grpcserver

# Start server on custom port
make run-grpcserver GRPC_PORT=50053

# Run tests (MODEL_PATH required)
make run-inferencetest1 MODEL_PATH=/path/to/model.gguf
make run-inferencetest2 MODEL_PATH=/path/to/model.gguf
make run-grpcclienttest SERVER_PATH='' ATTACH_GRPC_PORT=50053 MODEL_PATH=/path/to/model.gguf # send request to an already running gRPC server
make run-grpcclienttest MODEL_PATH=/path/to/model.gguf # run the gRPC server at dynamic port, then send request to it
```

### Docker Targets

| Target | Description |
|--------|-------------|
| `make docker-build` | Build all Docker images (server + client) |
| `make docker-build-server` | Build gRPC server Docker image |
| `make docker-build-client` | Build client test Docker image |
| `make docker-integration-test MODEL_PATH=<path>` | Run integration test with local model |
| `make docker-integration-test-ci` | Run integration test (downloads test model) |
| `make docker-clean` | Remove Docker images and volumes |

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_VERSION` | `b6770` | llama.cpp release version to download |
| `GRPC_PORT` | `50052` | gRPC server port |
| `MODEL_PATH` | (none) | Path to GGUF model file (required for tests) |
| `ATTACH_GRPC_PORT` | (none) | Port of already running gRPC server, for grpcclienttest |
| `GRPC_SERVER_PATH` | (auto) | Path of gRPC server executable to run, for grpcclienttest |
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
├── Makefile                 # Unified build system
├── build/
│   └── llama-binaries/      # Downloaded binaries (auto-created)
│       ├── bin/             # llama.cpp executables
│       ├── lib/             # DLLs/shared libraries + import libs
│       └── include/         # Header files
├── api/
│   └── proto/               # Protocol buffer definitions
├── cmd/
│   ├── grpcserver/          # gRPC server application
│   ├── grpcclienttest/      # gRPC client test
│   ├── inferencetest1/      # Direct inference test 1
│   └── inferencetest2/      # Direct inference test 2
├── docker/
│   ├── Dockerfile.server    # gRPC server Docker image
│   ├── Dockerfile.client    # Client test Docker image
│   ├── docker-compose.yml   # Local integration testing
│   └── docker-compose.ci.yml # CI integration testing
├── scripts/
│   ├── integration-test.sh  # Integration test runner (Linux/macOS)
│   └── integration-test.ps1 # Integration test runner (Windows)
├── internal/
│   ├── bindings/            # CGO bindings to llama.cpp
│   ├── grpcserver/          # gRPC server implementation
│   ├── logging/             # Logging utilities
│   └── modelmanagement/     # Model loading and caching
└── .github/
    └── workflows/
        └── ci.yml           # GitHub Actions CI workflow
```

## Usage

### gRPC Server

```bash
# Start server (binds to 127.0.0.1 by default)
./cmd/grpcserver/grpcserver --port 50052

# Start server binding to all interfaces (for Docker/remote access)
./cmd/grpcserver/grpcserver --host 0.0.0.0 --port 50052

# Or via make
make run-grpcserver GRPC_PORT=50052
```

#### Server Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Host address to bind (use `0.0.0.0` for Docker) |
| `--port` | `50051` | Port to listen for gRPC connections |
| `--ngpu` | `99` | Number of GPU layers to offload |
| `--mmap` | `false` | Use memory-mapped I/O for model loading |

### gRPC Client Test

```bash
# Run client test against running server
./cmd/grpcclienttest/grpcclienttest --host 127.0.0.1 --port 50052 --model /path/to/model.gguf

# Or via make
make run-grpcclienttest MODEL_PATH=/path/to/model.gguf
```

#### Client Test Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Server host address to connect to |
| `--port` | (none) | Server port to connect to |
| `--server` | (none) | Path to server executable (starts server automatically) |
| `--model` | (none) | Path to GGUF model file (required) |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `1.0` | Top-p (nucleus) sampling |
| `--top-k` | `0` | Top-k sampling (0 = disabled) |
| `--max-tokens` | `100` | Maximum tokens to generate |
| `--test-mode` | `baseline` | Test mode: `baseline`, `greedy`, `seeded`, `stress` |
| `--seed` | `-1` | Random seed (-1 = random) |

### Model Requirements

- **Format**: GGUF models (e.g., `model.gguf`)
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 supported
- **Sources**: [Hugging Face](https://huggingface.co/models?search=gguf)

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
./scripts/integration-test.sh --model-url https://example.com/model.gguf

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
# Run server container with mounted model
docker run -p 50051:50051 \
  -v /path/to/model.gguf:/models/model.gguf:ro \
  llamacpp-server

# Or using docker-compose
MODEL_PATH=/path/to/model.gguf docker compose -f docker/docker-compose.yml up server-only
```

### Using in Other Projects

The gRPC server Docker image can be used as a dependency in other projects:

```yaml
# In your project's docker-compose.yml
services:
  llm-server:
    image: llamacpp-server:latest
    ports:
      - "50051:50051"
    volumes:
      - ${MODEL_PATH}:/models/model.gguf:ro

  your-service:
    build: .
    depends_on:
      llm-server:
        condition: service_healthy
    environment:
      - LLM_SERVER_HOST=llm-server
      - LLM_SERVER_PORT=50051
```

## API Documentation

The gRPC server implements the interface defined in `api/proto/llmserver.proto`:

### LoadModel

Loads a model from the specified path with progress updates.

```protobuf
rpc LoadModel(LoadModelRequest) returns (stream LoadModelResponse);
```

### Predict

Generates text based on input prompt with streaming support.

```protobuf
rpc Predict(PredictRequest) returns (stream PredictResponse);
```

### Ping

Health check endpoint.

```protobuf
rpc Ping(PingRequest) returns (PingResponse);
```

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
RUN make build-grpcserver
```

This ensures Docker builds are identical to local builds.

### Platform-Specific Notes

#### Windows

- **CUDA Support**: Automatically detected if `nvcc` is in PATH
- **Import Libraries**: Required for linking - generated automatically by `make prepare`
- **Runtime DLLs**: Automatically copied to executable directories by run targets

#### macOS

- **Apple Silicon (M1/M2/M3/M4)**: Metal backend enabled automatically
- **Intel Macs**: CPU-only by default
- **libomp**: Required for OpenMP support - install with `brew install libomp`

#### Linux

- **CUDA**: Detected automatically if `nvcc` is available
- **Vulkan**: Binary download uses Vulkan build for cross-vendor GPU support

#### Docker

- **Base Image**: `debian:bookworm-slim` for minimal size
- **Architecture**: Linux x64 only (for now)
- **Dependencies**: Only `libgomp1` and `ca-certificates` at runtime

## CI/CD

The project includes GitHub Actions CI that runs on every push:

1. **Build & Test**: Builds on Ubuntu, Windows, and macOS
2. **Docker Integration Test**: Runs containerized integration tests
3. **Artifacts**: Uploads built binaries for each platform

See `.github/workflows/ci.yml` for details.

## License

This project is licensed under the MIT License.

### Component Licenses
- **LLaMA.cpp**: MIT License
- **gRPC-Go**: Apache 2.0 License
- **Protocol Buffers**: BSD 3-Clause License

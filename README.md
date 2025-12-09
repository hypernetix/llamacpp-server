# llama.cpp Go gRPC Server

A Go project for serving Large Language Models locally using **LLaMA.cpp** with gRPC interface.

## Features

- **gRPC Interface**: Clean API for model loading and text generation
- **Streaming Support**: Real-time text generation with streaming responses
- **Model Management**: Automatic loading and caching of GGUF models
- **Cross-platform**: Windows, Linux, and macOS support
- **GPU Acceleration**: CUDA (Windows/Linux) and Metal (macOS) support

## Quick Start

```bash
# Full build: download binaries + build all Go executables
make all

# Run client test, it will run, connect and send test request to gRPC server with a specified model
make run-grpcclienttest MODEL_PATH=/path/to/your/model.gguf
```

## Prerequisites

- **Go** 1.21 or later
- **Make** (GNU Make)
- **GCC/MinGW** - C compiler for CGO
  - Windows: MinGW-w64 via MSYS2 (includes `gendef`, `dlltool` for import libraries)
  - Linux: `build-essential` package
  - macOS: Xcode command line tools

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

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_VERSION` | `b6770` | llama.cpp release version to download |
| `GRPC_PORT` | `50052` | gRPC server port |
| `MODEL_PATH` | (none) | Path to GGUF model file (required for tests) |
| `ATTACH_GRPC_PORT` | (none) | port of already running gRPC server, for grpcclienttest |
| `SERVER_PATH` | (none) | path of gRPC server executable to run, for grpcclienttest |

### Using Different llama.cpp Versions

```bash
# Download specific version
make LLAMA_VERSION=b6800 prepare

# Full build with specific version
make LLAMA_VERSION=b6800 all
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
├── cmd/
│   ├── grpcserver/          # gRPC server application
│   ├── grpcclienttest/      # gRPC client test
│   ├── inferencetest1/      # Direct inference test 1
│   └── inferencetest2/      # Direct inference test 2
└── internal/
    ├── bindings/            # CGO bindings to llama.cpp
    ├── grpcserver/          # gRPC server implementation
    ├── logging/             # Logging utilities
    ├── modelmanagement/     # Model loading and caching
    └── proto/               # Protocol buffer definitions
```

## Usage

### gRPC Server

```bash
# Start server
./cmd/grpcserver/grpcserver --port 50052

# Or via make
make run-grpcserver GRPC_PORT=50052
```

### gRPC Client Test

```bash
# Run client test against running server
./cmd/grpcclienttest/grpcclienttest --port 50052 --model /path/to/model.gguf

# Or via make
make run-grpcclienttest MODEL_PATH=/path/to/model.gguf
```

### Model Requirements

- **Format**: GGUF models (e.g., `model.gguf`)
- **Quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 supported
- **Sources**: [Hugging Face](https://huggingface.co/models?search=gguf)

## API Documentation

The gRPC server implements the interface defined in `internal/proto/llmserver.proto`:

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

## Troubleshooting

### Windows: "gendef not found"

Install MinGW-w64 tools via MSYS2:
```bash
pacman -S mingw-w64-x86_64-tools-git
```

### Windows: DLL not found at runtime

The run targets automatically copy DLLs. For manual execution:
```powershell
# Option 1: Copy DLLs to executable directory
copy build\llama-binaries\lib\*.dll cmd\grpcserver\

# Option 2: Add lib directory to PATH
$env:PATH += ";$PWD\build\llama-binaries\lib"
```

### Build errors: undefined reference to llama_*

Ensure `make prepare` completed successfully and import libraries were created:
```bash
ls build/llama-binaries/lib/*.a  # Should show .dll.a files on Windows
```

### macOS: "library not found for -lomp"

Install OpenMP via Homebrew:
```bash
brew install libomp
```

### macOS: Library loading errors at runtime

Ensure the library path includes the llama.cpp libraries:
```bash
# The Makefile handles this automatically, but for manual runs:
export DYLD_LIBRARY_PATH=$PWD/build/llama-binaries/lib:$DYLD_LIBRARY_PATH
./cmd/grpcserver/grpcserver --port 50052
```

### Updating llama.cpp Version

```bash
# Clean and rebuild with new version
make clean
make LLAMA_VERSION=b6800 all
```

## Performance Tips

1. **Use GPU layers**: When running, configure `-ngl` to offload layers to GPU
2. **Optimize threads**: Don't over-provision CPU threads
3. **Use quantized models**: Q4 or Q5 models are faster with minimal quality loss
4. **Match backend**: Use CUDA for NVIDIA GPUs, Metal for Apple Silicon

## License

This project is licensed under the MIT License.

### Component Licenses
- **LLaMA.cpp**: MIT License
- **gRPC-Go**: Apache 2.0 License
- **Protocol Buffers**: BSD 3-Clause License

# Unified Makefile for llama.cpp Go gRPC Server
# Handles: binary download, import library generation (Windows), and Go build
#
# Compatible with: MSYS make, MinGW make, GNU make on Windows/Linux/macOS
# On Windows, uses PowerShell for file operations (works with any make)
#
# Quick Start:
#   make prepare  - Download llama.cpp binaries + generate import libs (Windows)
#   make all      - Full build: prepare + build all Go binaries
#   make build    - Build Go binaries only (assumes 'prepare' was done)
#
# Run targets:
#   make run-grpcserver      - Run the gRPC server
#   make run-grpcclienttest  - Run the gRPC client test
#   make run-inferencetest1  - Run inference test 1
#   make run-inferencetest2  - Run inference test 2

# =============================================================================
# Configuration
# =============================================================================

# llama.cpp version to download (can be overridden: make LLAMA_VERSION=b6770)
LLAMA_VERSION ?= b6770

# Build directories (use forward slashes - works everywhere)
BUILD_DIR := build
LLAMA_DIR := $(BUILD_DIR)/llama-binaries

# Go commands
GO_BUILD_FLAGS := -v

# Default port for gRPC server
GRPC_PORT ?= 50052

# Default port for gRPC client test
ATTACH_GRPC_PORT ?= 0

# Default path for gRPC server
ifeq ($(OS),Windows_NT)
	GRPC_SERVER_PATH ?= "./cmd/grpcserver/grpcserver.exe"
else
	GRPC_SERVER_PATH ?= "./cmd/grpcserver/grpcserver"
endif

# Default model path (override with: make run-grpcclienttest MODEL_PATH=/path/to/model.gguf)
MODEL_PATH ?=

# =============================================================================
# Platform Detection
# =============================================================================

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
        ARCH := x64
    else
        ARCH := x86
    endif
    LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-win-cpu-$(ARCH).zip
    LLAMA_URL := https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/$(LLAMA_ARCHIVE)
    EXE := .exe
    # DLL list for running
    DLLS := ggml.dll ggml-base.dll ggml-cpu-x64.dll llama.dll
    # PowerShell commands (work with any make on Windows)
    PS := powershell -NoProfile -Command
    MKDIR_P = $(PS) "New-Item -ItemType Directory -Force -Path '$(1)' | Out-Null"
    RM_RF = $(PS) "if (Test-Path '$(1)') { Remove-Item -Recurse -Force '$(1)' }"
    RM_F = $(PS) "if (Test-Path '$(1)') { Remove-Item -Force '$(1)' }"
    CP_F = $(PS) "Copy-Item -Force '$(1)' '$(2)'"
    MV_F = $(PS) "if (Test-Path '$(1)') { Move-Item -Force '$(1)' '$(2)' }"
else
    DETECTED_OS := $(shell uname -s)
    ARCH := $(shell uname -m)
    ifeq ($(DETECTED_OS),Darwin)
        ifeq ($(ARCH),arm64)
            LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-macos-arm64.zip
        else
            LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-macos-x64.zip
        endif
    else
        LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-ubuntu-x64.zip
    endif
    LLAMA_URL := https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/$(LLAMA_ARCHIVE)
    EXE :=
    MKDIR_P = mkdir -p '$(1)'
    RM_RF = rm -rf '$(1)'
    RM_F = rm -f '$(1)'
    CP_F = cp -f '$(1)' '$(2)'
    MV_F = mv -f '$(1)' '$(2)' 2>/dev/null || true
endif

# =============================================================================
# CUDA Detection
# =============================================================================

ifeq ($(OS),Windows_NT)
    CUDA_AVAILABLE := $(shell $(PS) "if (Get-Command nvcc -ErrorAction SilentlyContinue) { 'yes' }")
else
    CUDA_AVAILABLE := $(shell command -v nvcc 2>/dev/null)
endif

ifdef CUDA_AVAILABLE
    ifeq ($(DETECTED_OS),Windows)
        CUDA_VERSION := 12.4
        LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-win-cuda-cu$(CUDA_VERSION)-$(ARCH).zip
        CUDA_ARCHIVE := cudart-llama-bin-win-cuda-cu$(CUDA_VERSION)-$(ARCH).zip
        LLAMA_URL := https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/$(LLAMA_ARCHIVE)
        CUDA_URL := https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/$(CUDA_ARCHIVE)
        DLLS += cudart64_12.dll cublas64_12.dll cublasLt64_12.dll
    else ifeq ($(DETECTED_OS),Linux)
        LLAMA_ARCHIVE := llama-$(LLAMA_VERSION)-bin-ubuntu-vulkan-x64.zip
        LLAMA_URL := https://github.com/ggml-org/llama.cpp/releases/download/$(LLAMA_VERSION)/$(LLAMA_ARCHIVE)
    endif
endif

# =============================================================================
# Main Targets
# =============================================================================

.PHONY: all prepare build clean help check-deps
.PHONY: download-binaries import-libs
.PHONY: build-grpcserver build-grpcclienttest build-inferencetest1 build-inferencetest2
.PHONY: run-grpcserver run-grpcclienttest run-inferencetest1 run-inferencetest2
.PHONY: copy-dlls-grpcserver copy-dlls-grpcclienttest copy-dlls-inferencetest1 copy-dlls-inferencetest2

all: prepare build
	@echo ""
	@echo "=== Build Complete ==="
	@echo "Executables built in cmd/*/."
	@echo "Run with: make run-grpcserver, make run-grpcclienttest, etc."

prepare: download-binaries import-libs
	@echo ""
	@echo "=== Preparation Complete ==="
	@echo "You can now build with: make build"

build: build-grpcserver build-grpcclienttest build-inferencetest1 build-inferencetest2
	@echo ""
	@echo "=== All Go binaries built ==="

# =============================================================================
# Check Dependencies
# =============================================================================

check-deps:
ifeq ($(OS),Windows_NT)
	@$(PS) "if (-not (Get-Command gendef -ErrorAction SilentlyContinue)) { Write-Warning 'gendef not found. Install: pacman -S mingw-w64-x86_64-tools-git' }"
else ifeq ($(DETECTED_OS),Darwin)
	@command -v brew >/dev/null 2>&1 || { echo "Error: Homebrew not found. Install from https://brew.sh"; exit 1; }
	@command -v go >/dev/null 2>&1 || { echo "Error: go not found. Install with: brew install go"; exit 1; }
	@[ -d "$$(brew --prefix libomp 2>/dev/null)" ] || { echo "Error: libomp not found. Install with: brew install libomp"; exit 1; }
else
	@command -v wget >/dev/null 2>&1 || { echo "Error: wget not found. Install with: sudo apt install wget"; exit 1; }
	@command -v unzip >/dev/null 2>&1 || { echo "Error: unzip not found. Install with: sudo apt install unzip"; exit 1; }
	@command -v go >/dev/null 2>&1 || { echo "Error: go not found. Install Go from https://go.dev/dl/"; exit 1; }
endif

# =============================================================================
# Download Binaries
# =============================================================================

download-binaries: check-deps
	@echo "=== Downloading llama.cpp $(LLAMA_VERSION) binaries for $(DETECTED_OS) $(ARCH) ==="
	@echo "Binary URL: $(LLAMA_URL)"
	@$(call MKDIR_P,$(LLAMA_DIR))
ifeq ($(OS),Windows_NT)
	$(PS) "Invoke-WebRequest -Uri '$(LLAMA_URL)' -OutFile '$(LLAMA_DIR)/$(LLAMA_ARCHIVE)'"
	$(PS) "Expand-Archive -Path '$(LLAMA_DIR)/$(LLAMA_ARCHIVE)' -DestinationPath '$(LLAMA_DIR)' -Force"
ifdef CUDA_AVAILABLE
	@echo "Downloading CUDA runtime: $(CUDA_URL)"
	$(PS) "Invoke-WebRequest -Uri '$(CUDA_URL)' -OutFile '$(LLAMA_DIR)/$(CUDA_ARCHIVE)'"
	$(PS) "Expand-Archive -Path '$(LLAMA_DIR)/$(CUDA_ARCHIVE)' -DestinationPath '$(LLAMA_DIR)' -Force"
endif
else
	wget -q -P $(LLAMA_DIR) $(LLAMA_URL)
	cd $(LLAMA_DIR) && unzip -q $(LLAMA_ARCHIVE)
endif
	@echo ""
	@echo "=== Downloading matching header files from $(LLAMA_VERSION) source ==="
ifeq ($(OS),Windows_NT)
	$(PS) "Invoke-WebRequest -Uri 'https://github.com/ggml-org/llama.cpp/archive/$(LLAMA_VERSION).zip' -OutFile '$(LLAMA_DIR)/llama-src.zip'"
	$(PS) "Expand-Archive -Path '$(LLAMA_DIR)/llama-src.zip' -DestinationPath '$(LLAMA_DIR)' -Force"
else
	wget -q -O $(LLAMA_DIR)/llama-src.zip https://github.com/ggml-org/llama.cpp/archive/$(LLAMA_VERSION).zip
	cd $(LLAMA_DIR) && unzip -q llama-src.zip
endif
	@echo ""
	@echo "=== Organizing directory structure ==="
	@$(call MKDIR_P,$(LLAMA_DIR)/include)
	@$(call MKDIR_P,$(LLAMA_DIR)/lib)
	@$(call MKDIR_P,$(LLAMA_DIR)/bin)
ifeq ($(OS),Windows_NT)
	-$(PS) "Move-Item -Path '$(LLAMA_DIR)/*.exe' -Destination '$(LLAMA_DIR)/bin/' -Force -ErrorAction SilentlyContinue"
	-$(PS) "Move-Item -Path '$(LLAMA_DIR)/*.dll' -Destination '$(LLAMA_DIR)/lib/' -Force -ErrorAction SilentlyContinue"
	-$(PS) "Move-Item -Path '$(LLAMA_DIR)/*.lib' -Destination '$(LLAMA_DIR)/lib/' -Force -ErrorAction SilentlyContinue"
else
	@# Linux/macOS archives may extract to build/bin/ subfolder - handle both cases
	@if [ -d "$(LLAMA_DIR)/build/bin" ]; then \
		echo "Moving files from nested build/bin/ structure..."; \
		mv $(LLAMA_DIR)/build/bin/llama-* $(LLAMA_DIR)/bin/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/build/bin/rpc-server $(LLAMA_DIR)/bin/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/build/bin/*.so* $(LLAMA_DIR)/lib/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/build/bin/*.dylib $(LLAMA_DIR)/lib/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/build/bin/LICENSE* $(LLAMA_DIR)/ 2>/dev/null || true; \
		rm -rf $(LLAMA_DIR)/build; \
	else \
		mv $(LLAMA_DIR)/llama-*[!.zip] $(LLAMA_DIR)/bin/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/*.so* $(LLAMA_DIR)/lib/ 2>/dev/null || true; \
		mv $(LLAMA_DIR)/*.dylib $(LLAMA_DIR)/lib/ 2>/dev/null || true; \
	fi
	-chmod +x $(LLAMA_DIR)/bin/* 2>/dev/null || true
	@# Create symlinks for library compatibility
	@echo "Creating library symlinks for compatibility..."
	@# Linux: ggml-cpu-x64 -> ggml-cpu
	@if [ -f "$(LLAMA_DIR)/lib/libggml-cpu-x64.so" ] && [ ! -f "$(LLAMA_DIR)/lib/libggml-cpu.so" ]; then \
		ln -sf libggml-cpu-x64.so $(LLAMA_DIR)/lib/libggml-cpu.so; \
		echo "  Created symlink: libggml-cpu.so -> libggml-cpu-x64.so"; \
	fi
	@# macOS arm64: ggml-cpu-arm64 -> ggml-cpu
	@if [ -f "$(LLAMA_DIR)/lib/libggml-cpu-arm64.dylib" ] && [ ! -f "$(LLAMA_DIR)/lib/libggml-cpu.dylib" ]; then \
		ln -sf libggml-cpu-arm64.dylib $(LLAMA_DIR)/lib/libggml-cpu.dylib; \
		echo "  Created symlink: libggml-cpu.dylib -> libggml-cpu-arm64.dylib"; \
	fi
	@# macOS x64: ggml-cpu-x64 -> ggml-cpu (Intel Macs)
	@if [ -f "$(LLAMA_DIR)/lib/libggml-cpu-x64.dylib" ] && [ ! -f "$(LLAMA_DIR)/lib/libggml-cpu.dylib" ]; then \
		ln -sf libggml-cpu-x64.dylib $(LLAMA_DIR)/lib/libggml-cpu.dylib; \
		echo "  Created symlink: libggml-cpu.dylib -> libggml-cpu-x64.dylib"; \
	fi
endif
	@echo "Copying headers from source archive..."
ifeq ($(OS),Windows_NT)
	-$(PS) "Copy-Item -Recurse -Force '$(LLAMA_DIR)/llama.cpp-$(LLAMA_VERSION)/include/*' '$(LLAMA_DIR)/include/'"
	@$(call MKDIR_P,$(LLAMA_DIR)/include/ggml)
	-$(PS) "Copy-Item -Recurse -Force '$(LLAMA_DIR)/llama.cpp-$(LLAMA_VERSION)/ggml/include/*' '$(LLAMA_DIR)/include/ggml/'"
else
	-cp -r $(LLAMA_DIR)/llama.cpp-$(LLAMA_VERSION)/include/* $(LLAMA_DIR)/include/ 2>/dev/null || true
	-mkdir -p $(LLAMA_DIR)/include/ggml
	-cp -r $(LLAMA_DIR)/llama.cpp-$(LLAMA_VERSION)/ggml/include/* $(LLAMA_DIR)/include/ggml/ 2>/dev/null || true
endif
	@echo "Cleaning up temporary files..."
	@$(call RM_F,$(LLAMA_DIR)/llama-src.zip)
	@$(call RM_RF,$(LLAMA_DIR)/llama.cpp-$(LLAMA_VERSION))
	@$(call RM_F,$(LLAMA_DIR)/$(LLAMA_ARCHIVE))
	@echo ""
	@echo "=== Download complete! ==="
	@echo "Binaries: $(LLAMA_DIR)/bin/"
	@echo "Libraries: $(LLAMA_DIR)/lib/"
	@echo "Headers: $(LLAMA_DIR)/include/"

# =============================================================================
# Import Libraries (Windows only)
# =============================================================================

import-libs:
ifeq ($(OS),Windows_NT)
	@echo ""
	@echo "=== Generating import libraries from DLLs (Windows) ==="
	@$(PS) "if (-not (Get-Command gendef -ErrorAction SilentlyContinue)) { Write-Error 'gendef not found. Install: pacman -S mingw-w64-x86_64-tools-git'; exit 1 }"
	@$(PS) "if (-not (Get-Command dlltool -ErrorAction SilentlyContinue)) { Write-Error 'dlltool not found. Install MinGW-w64/MSYS2'; exit 1 }"
	cd $(LLAMA_DIR)/lib && gendef llama.dll && dlltool -d llama.def -D llama.dll -l libllama.dll.a
	@$(call RM_F,$(LLAMA_DIR)/lib/llama.def)
	cd $(LLAMA_DIR)/lib && gendef ggml.dll && dlltool -d ggml.def -D ggml.dll -l libggml.dll.a
	@$(call RM_F,$(LLAMA_DIR)/lib/ggml.def)
	cd $(LLAMA_DIR)/lib && gendef ggml-base.dll && dlltool -d ggml-base.def -D ggml-base.dll -l libggml-base.dll.a
	@$(call RM_F,$(LLAMA_DIR)/lib/ggml-base.def)
	cd $(LLAMA_DIR)/lib && gendef ggml-cpu-x64.dll && dlltool -d ggml-cpu-x64.def -D ggml-cpu-x64.dll -l libggml-cpu.dll.a
	@$(call RM_F,$(LLAMA_DIR)/lib/ggml-cpu-x64.def)
	@echo "Import libraries created successfully!"
else
	@echo "Import libraries not needed on $(DETECTED_OS)"
endif

# =============================================================================
# Go Build Targets
# =============================================================================

build-grpcserver:
	@echo "Building grpcserver..."
	cd cmd/grpcserver && go build $(GO_BUILD_FLAGS) -o grpcserver$(EXE) .

build-grpcclienttest:
	@echo "Building grpcclienttest..."
	cd cmd/grpcclienttest && go build $(GO_BUILD_FLAGS) -o grpcclienttest$(EXE) .

build-inferencetest1:
	@echo "Building inferencetest1..."
	cd cmd/inferencetest1 && go build $(GO_BUILD_FLAGS) -o inferencetest1$(EXE) .

build-inferencetest2:
	@echo "Building inferencetest2..."
	cd cmd/inferencetest2 && go build $(GO_BUILD_FLAGS) -o inferencetest2$(EXE) .

# =============================================================================
# Copy shared libraries - needed for ggml_backend_load_all() to find backends
# =============================================================================

copy-dlls-grpcserver:
ifeq ($(OS),Windows_NT)
	@echo "Copying DLLs to cmd/grpcserver/..."
	@$(PS) "Copy-Item -Path '$(LLAMA_DIR)/lib/*.dll' -Destination 'cmd/grpcserver/' -Force"
else
	@echo "Copying shared libraries to cmd/grpcserver/..."
	@cp -f $(LLAMA_DIR)/lib/*.so* cmd/grpcserver/ 2>/dev/null || true
	@cp -f $(LLAMA_DIR)/lib/*.dylib cmd/grpcserver/ 2>/dev/null || true
endif

copy-dlls-grpcclienttest:
ifeq ($(OS),Windows_NT)
	@echo "Copying DLLs to cmd/grpcclienttest/..."
	@$(PS) "Copy-Item -Path '$(LLAMA_DIR)/lib/*.dll' -Destination 'cmd/grpcclienttest/' -Force"
else
	@echo "Copying shared libraries to cmd/grpcclienttest/..."
	@cp -f $(LLAMA_DIR)/lib/*.so* cmd/grpcclienttest/ 2>/dev/null || true
	@cp -f $(LLAMA_DIR)/lib/*.dylib cmd/grpcclienttest/ 2>/dev/null || true
endif

copy-dlls-inferencetest1:
ifeq ($(OS),Windows_NT)
	@echo "Copying DLLs to cmd/inferencetest1/..."
	@$(PS) "Copy-Item -Path '$(LLAMA_DIR)/lib/*.dll' -Destination 'cmd/inferencetest1/' -Force"
else
	@echo "Copying shared libraries to cmd/inferencetest1/..."
	@cp -f $(LLAMA_DIR)/lib/*.so* cmd/inferencetest1/ 2>/dev/null || true
	@cp -f $(LLAMA_DIR)/lib/*.dylib cmd/inferencetest1/ 2>/dev/null || true
endif

copy-dlls-inferencetest2:
ifeq ($(OS),Windows_NT)
	@echo "Copying DLLs to cmd/inferencetest2/..."
	@$(PS) "Copy-Item -Path '$(LLAMA_DIR)/lib/*.dll' -Destination 'cmd/inferencetest2/' -Force"
else
	@echo "Copying shared libraries to cmd/inferencetest2/..."
	@cp -f $(LLAMA_DIR)/lib/*.so* cmd/inferencetest2/ 2>/dev/null || true
	@cp -f $(LLAMA_DIR)/lib/*.dylib cmd/inferencetest2/ 2>/dev/null || true
endif

# =============================================================================
# Run Targets
# =============================================================================

# Library path for runtime (Linux/macOS need LD_LIBRARY_PATH / DYLD_LIBRARY_PATH)
# Include both the lib directory AND the executable directory (where .so files are copied)
ifeq ($(OS),Windows_NT)
    RUN_ENV :=
else ifeq ($(DETECTED_OS),Darwin)
    RUN_ENV := DYLD_LIBRARY_PATH=$(CURDIR)/$(LLAMA_DIR)/lib:$$DYLD_LIBRARY_PATH
else
    RUN_ENV := LD_LIBRARY_PATH=$(CURDIR)/$(LLAMA_DIR)/lib:$$LD_LIBRARY_PATH
endif

# Per-target RUN_ENV that includes the executable's directory for backend discovery
RUN_ENV_GRPCSERVER := $(RUN_ENV)
RUN_ENV_GRPCCLIENTTEST := $(RUN_ENV)
RUN_ENV_INFERENCETEST1 := $(RUN_ENV)
RUN_ENV_INFERENCETEST2 := $(RUN_ENV)
ifeq ($(DETECTED_OS),Darwin)
    RUN_ENV_GRPCSERVER := DYLD_LIBRARY_PATH=$(CURDIR)/cmd/grpcserver:$(CURDIR)/$(LLAMA_DIR)/lib:$$DYLD_LIBRARY_PATH
    RUN_ENV_GRPCCLIENTTEST := DYLD_LIBRARY_PATH=$(CURDIR)/cmd/grpcclienttest:$(CURDIR)/$(LLAMA_DIR)/lib:$$DYLD_LIBRARY_PATH
    RUN_ENV_INFERENCETEST1 := DYLD_LIBRARY_PATH=$(CURDIR)/cmd/inferencetest1:$(CURDIR)/$(LLAMA_DIR)/lib:$$DYLD_LIBRARY_PATH
    RUN_ENV_INFERENCETEST2 := DYLD_LIBRARY_PATH=$(CURDIR)/cmd/inferencetest2:$(CURDIR)/$(LLAMA_DIR)/lib:$$DYLD_LIBRARY_PATH
else ifneq ($(OS),Windows_NT)
    RUN_ENV_GRPCSERVER := LD_LIBRARY_PATH=$(CURDIR)/cmd/grpcserver:$(CURDIR)/$(LLAMA_DIR)/lib:$$LD_LIBRARY_PATH
    RUN_ENV_GRPCCLIENTTEST := LD_LIBRARY_PATH=$(CURDIR)/cmd/grpcclienttest:$(CURDIR)/$(LLAMA_DIR)/lib:$$LD_LIBRARY_PATH
    RUN_ENV_INFERENCETEST1 := LD_LIBRARY_PATH=$(CURDIR)/cmd/inferencetest1:$(CURDIR)/$(LLAMA_DIR)/lib:$$LD_LIBRARY_PATH
    RUN_ENV_INFERENCETEST2 := LD_LIBRARY_PATH=$(CURDIR)/cmd/inferencetest2:$(CURDIR)/$(LLAMA_DIR)/lib:$$LD_LIBRARY_PATH
endif

run-grpcserver: build-grpcserver copy-dlls-grpcserver
	@echo ""
	@echo "=== Running gRPC Server on port $(GRPC_PORT) ==="
	$(RUN_ENV_GRPCSERVER) ./cmd/grpcserver/grpcserver$(EXE) --port $(GRPC_PORT)

run-grpcclienttest: build-grpcclienttest copy-dlls-grpcclienttest copy-dlls-grpcserver
ifeq ($(MODEL_PATH),)
	@echo "Error: MODEL_PATH is required"
	@echo "Usage: make run-grpcclienttest MODEL_PATH=/path/to/model.gguf"
	@exit 1
endif
	@echo ""
	@echo "=== Running gRPC Client Test ==="
	$(RUN_ENV_GRPCCLIENTTEST) ./cmd/grpcclienttest/grpcclienttest$(EXE) --port $(ATTACH_GRPC_PORT) --server "$(GRPC_SERVER_PATH)" --model "$(MODEL_PATH)"

run-inferencetest1: build-inferencetest1 copy-dlls-inferencetest1
ifeq ($(MODEL_PATH),)
	@echo "Error: MODEL_PATH is required"
	@echo "Usage: make run-inferencetest1 MODEL_PATH=/path/to/model.gguf"
	@exit 1
endif
	@echo ""
	@echo "=== Running Inference Test 1 ==="
	$(RUN_ENV_INFERENCETEST1) ./cmd/inferencetest1/inferencetest1$(EXE) "$(MODEL_PATH)"

run-inferencetest2: build-inferencetest2 copy-dlls-inferencetest2
ifeq ($(MODEL_PATH),)
	@echo "Error: MODEL_PATH is required"
	@echo "Usage: make run-inferencetest2 MODEL_PATH=/path/to/model.gguf"
	@exit 1
endif
	@echo ""
	@echo "=== Running Inference Test 2 ==="
	$(RUN_ENV_INFERENCETEST2) ./cmd/inferencetest2/inferencetest2$(EXE) "$(MODEL_PATH)"

# =============================================================================
# Clean
# =============================================================================

clean:
	@echo "Cleaning build artifacts..."
	@$(call RM_RF,$(BUILD_DIR))
	@$(call RM_F,cmd/grpcserver/grpcserver$(EXE))
	@$(call RM_F,cmd/grpcclienttest/grpcclienttest$(EXE))
	@$(call RM_F,cmd/inferencetest1/inferencetest1$(EXE))
	@$(call RM_F,cmd/inferencetest2/inferencetest2$(EXE))
ifeq ($(OS),Windows_NT)
	-$(PS) "Remove-Item -Path 'cmd/grpcserver/*.dll' -Force -ErrorAction SilentlyContinue"
	-$(PS) "Remove-Item -Path 'cmd/grpcclienttest/*.dll' -Force -ErrorAction SilentlyContinue"
	-$(PS) "Remove-Item -Path 'cmd/inferencetest1/*.dll' -Force -ErrorAction SilentlyContinue"
	-$(PS) "Remove-Item -Path 'cmd/inferencetest2/*.dll' -Force -ErrorAction SilentlyContinue"
else
	-rm -f cmd/grpcserver/*.so* 2>/dev/null || true
	-rm -f cmd/grpcclienttest/*.so* 2>/dev/null || true
	-rm -f cmd/inferencetest1/*.so* 2>/dev/null || true
	-rm -f cmd/inferencetest2/*.so* 2>/dev/null || true
	-rm -f cmd/grpcserver/*.dylib 2>/dev/null || true
	-rm -f cmd/grpcclienttest/*.dylib 2>/dev/null || true
	-rm -f cmd/inferencetest1/*.dylib 2>/dev/null || true
	-rm -f cmd/inferencetest2/*.dylib 2>/dev/null || true
endif
	@echo "Clean complete."

# =============================================================================
# Help
# =============================================================================

help:
	@echo "=============================================="
	@echo "  llama.cpp Go gRPC Server - Build System"
	@echo "=============================================="
	@echo ""
	@echo "Main targets:"
	@echo "  make all              - Full build (prepare + build all Go binaries)"
	@echo "  make prepare          - Download binaries + generate import libs"
	@echo "  make build            - Build all Go binaries"
	@echo "  make clean            - Remove all build artifacts"
	@echo ""
	@echo "Individual build targets:"
	@echo "  make build-grpcserver      - Build gRPC server"
	@echo "  make build-grpcclienttest  - Build gRPC client test"
	@echo "  make build-inferencetest1  - Build inference test 1"
	@echo "  make build-inferencetest2  - Build inference test 2"
	@echo ""
	@echo "Run targets:"
	@echo "  make run-grpcserver                          - Run gRPC server"
	@echo "  make run-grpcclienttest MODEL_PATH=<path>    - Run client test"
	@echo "  make run-inferencetest1 MODEL_PATH=<path>    - Run inference test 1"
	@echo "  make run-inferencetest2 MODEL_PATH=<path>    - Run inference test 2"
	@echo ""
	@echo "Configuration variables:"
	@echo "  LLAMA_VERSION  	- llama.cpp version (default: $(LLAMA_VERSION))"
	@echo "  GRPC_PORT      	- gRPC server port (default: $(GRPC_PORT))"
	@echo "  ATTACH_GRPC_PORT 	- gRPC client test port (default: $(ATTACH_GRPC_PORT))"
	@echo "  GRPC_SERVER_PATH 	- Path to gRPC server executable (default: $(GRPC_SERVER_PATH))"
	@echo "  MODEL_PATH     	- Path to GGUF model file"
	@echo ""
	@echo "Detected environment:"
	@echo "  OS: $(DETECTED_OS)"
	@echo "  Architecture: $(ARCH)"
	@echo "  CUDA: $(if $(CUDA_AVAILABLE),Available,Not available)"
	@echo ""
	@echo "Examples:"
	@echo "  make all"
	@echo "  make run-grpcserver GRPC_PORT=50053"
	@echo "  make run-grpcclienttest MODEL_PATH=/path/to/model.gguf"
	@echo "  make LLAMA_VERSION=b6800 prepare"

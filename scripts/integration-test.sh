#!/bin/bash
#
# Integration test script for llamacpp server
# Runs Docker-based integration tests with configurable parameters
# Tests both gRPC and HTTP transports sequentially
#
# Usage:
#   ./scripts/integration-test.sh --model /path/to/model.gguf
#   ./scripts/integration-test.sh --model-url https://example.com/model.gguf
#   ./scripts/integration-test.sh --ci  # Use default CI model
#
# Options:
#   --model PATH       Path to local GGUF model file
#   --model-url URL    URL to download the model from
#   --ci               Use CI defaults (download SmolLM2-135M)
#   --test-mode MODE   Test mode: baseline, greedy, seeded, stress (default: greedy)
#   --no-cleanup       Don't remove containers after test
#   --build            Force rebuild Docker images
#   --verbose          Show verbose output
#   --help             Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
MODEL_URL=""
TEST_MODE="greedy"
CI_MODE=false
CLEANUP=true
FORCE_BUILD=false
VERBOSE=false
COMPOSE_FILE="docker/docker-compose.yml"

# CI defaults (SmolLM2-135M from bartowski)
CI_MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
CI_MODEL_NAME="SmolLM2-135M-Instruct-Q4_K_M.gguf"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    grep '^#' "$0" | head -20 | tail -17 | sed 's/^# \?//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-url)
            MODEL_URL="$2"
            shift 2
            ;;
        --ci)
            CI_MODE=true
            shift
            ;;
        --test-mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --no-cleanup)
            CLEANUP=false
            shift
            ;;
        --build)
            FORCE_BUILD=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Change to project root
cd "$PROJECT_ROOT"

# Validate inputs
if [[ "$CI_MODE" == true ]]; then
    log_info "Running in CI mode with default test model"
    COMPOSE_FILE="docker/docker-compose.ci.yml"
elif [[ -n "$MODEL_URL" ]]; then
    log_info "Using model URL: $MODEL_URL"
    # Download model to temp location
    TEMP_MODEL_DIR=$(mktemp -d)
    MODEL_NAME=$(basename "$MODEL_URL")
    MODEL_PATH="$TEMP_MODEL_DIR/$MODEL_NAME"
    log_info "Downloading model to $MODEL_PATH..."
    curl -L --retry 3 --retry-delay 5 -o "$MODEL_PATH" "$MODEL_URL"
elif [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -f "$MODEL_PATH" ]]; then
        log_error "Model file not found: $MODEL_PATH"
        exit 1
    fi
    log_info "Using local model: $MODEL_PATH"
else
    log_error "Model is required. Use --model, --model-url, or --ci"
    show_help
fi

# Build docker images
BUILD_ARGS=""
if [[ "$FORCE_BUILD" == true ]]; then
    BUILD_ARGS="--build"
fi

log_info "Starting integration test..."
log_info "  Compose file: $COMPOSE_FILE"
log_info "  Test mode: $TEST_MODE"
log_info "  LLAMA_VERSION: ${LLAMA_VERSION:-<not set>}"
log_info "  GPU_VARIANT: ${GPU_VARIANT:-<not set, compose default: cpu>}"

# Export environment variables for docker-compose
export TEST_MODE="$TEST_MODE"
if [[ -n "$MODEL_PATH" ]]; then
    export MODEL_PATH="$MODEL_PATH"
fi

# Container name suffixes
if [[ "$CI_MODE" == true ]]; then
    SERVER_CONTAINER="llamacpp-server-ci"
    CLIENT_GRPC_CONTAINER="llamacpp-client-grpc-ci"
    CLIENT_HTTP_CONTAINER="llamacpp-client-http-ci"
else
    SERVER_CONTAINER="llamacpp-server"
    CLIENT_GRPC_CONTAINER="llamacpp-client-grpc"
    CLIENT_HTTP_CONTAINER="llamacpp-client-http"
fi

# Cleanup function
cleanup() {
    local exit_code=$?
    if [[ "$CLEANUP" == true ]]; then
        log_info "Cleaning up containers..."
        docker compose -f "$COMPOSE_FILE" down -v --remove-orphans 2>/dev/null || true
    fi
    
    # Clean up temp model if downloaded
    if [[ -n "$TEMP_MODEL_DIR" ]] && [[ -d "$TEMP_MODEL_DIR" ]]; then
        rm -rf "$TEMP_MODEL_DIR"
    fi
    
    if [[ $exit_code -eq 0 ]]; then
        log_info "Integration test PASSED (gRPC, HTTP, Parallel)"
    else
        log_error "Integration test FAILED (exit code: $exit_code)"
    fi
    
    exit $exit_code
}
trap cleanup EXIT

# Run a single transport test
# Uses --exit-code-from to get the client's exit code (not the server's SIGTERM code)
# Usage: run_transport_test <transport_name> <server_service> <client_service> <client_container>
run_transport_test() {
    local transport=$1
    local server_service=$2
    local client_service=$3
    local client_container=$4

    log_info ""
    log_info "========================================="
    log_info "  Running $transport integration test"
    log_info "========================================="

    # --exit-code-from ensures compose returns the client's exit code, not the
    # server's SIGTERM exit code (137) from --abort-on-container-exit
    local compose_exit=0
    docker compose -f "$COMPOSE_FILE" up $BUILD_ARGS \
        --abort-on-container-exit --exit-code-from "$client_service" \
        --no-deps "$server_service" "$client_service" || compose_exit=$?

    log_info "Docker compose exit code: $compose_exit"
    log_info "Containers used:"
    docker ps -a --filter "name=llamacpp" --format "  {{.Names}}\t{{.Image}}\t{{.Status}}" 2>/dev/null || true

    if [[ $compose_exit -ne 0 ]]; then
        # Show actual client container exit code for debugging
        local client_exit
        client_exit=$(docker inspect "$client_container" --format='{{.State.ExitCode}}' 2>/dev/null || echo "unknown")
        log_error "$transport test FAILED (compose exit: $compose_exit, client exit: $client_exit)"
        return 1
    fi

    log_info "$transport test PASSED"
    return 0
}

# Build and prepare
log_info "Building and starting containers..."

if [[ "$CI_MODE" == true ]]; then
    # In CI mode, first run the model downloader to completion
    log_info "Downloading test model..."
    docker compose -f "$COMPOSE_FILE" up --build model-downloader
    if [[ $? -ne 0 ]]; then
        log_error "Failed to download model"
        exit 1
    fi
fi

# Only pass --build on the first run; images are reused for the second
FIRST_BUILD_ARGS="$BUILD_ARGS"

# Run gRPC test
run_transport_test "gRPC" "server" "client-grpc" "$CLIENT_GRPC_CONTAINER"
GRPC_EXIT=$?

# Clear --build for second run (images already built)
BUILD_ARGS=""

if [[ $GRPC_EXIT -ne 0 ]]; then
    log_error "gRPC test failed, skipping HTTP test"
    exit $GRPC_EXIT
fi

# Run HTTP test (server restarts, model reloads — acceptable for test)
run_transport_test "HTTP" "server" "client-http" "$CLIENT_HTTP_CONTAINER"
HTTP_EXIT=$?

if [[ $HTTP_EXIT -ne 0 ]]; then
    exit $HTTP_EXIT
fi

# Run parallel inference test (multi-slot server)
if [[ "$CI_MODE" == true ]]; then
    PARALLEL_CLIENT_CONTAINER="llamacpp-client-grpc-cb-ci"
else
    PARALLEL_CLIENT_CONTAINER="llamacpp-client-grpc-cb"
fi

run_transport_test "Parallel" "server-cb" "client-grpc-cb" "$PARALLEL_CLIENT_CONTAINER"
PARALLEL_EXIT=$?

if [[ $PARALLEL_EXIT -ne 0 ]]; then
    exit $PARALLEL_EXIT
fi

log_info ""
log_info "========================================="
log_info "  All tests PASSED (gRPC, HTTP, Parallel)"
log_info "========================================="

exit 0

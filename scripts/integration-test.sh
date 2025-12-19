#!/bin/bash
#
# Integration test script for llamacpp gRPC server
# Runs Docker-based integration tests with configurable parameters
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

# CI defaults
CI_MODEL_URL="https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf"
CI_MODEL_NAME="smollm2-135m-instruct-q4_k_m.gguf"

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

# Export environment variables for docker-compose
export TEST_MODE="$TEST_MODE"
if [[ -n "$MODEL_PATH" ]]; then
    export MODEL_PATH="$MODEL_PATH"
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
        log_info "Integration test PASSED"
    else
        log_error "Integration test FAILED (exit code: $exit_code)"
    fi
    
    exit $exit_code
}
trap cleanup EXIT

# Run the integration test
log_info "Building and starting containers..."

if [[ "$CI_MODE" == true ]]; then
    # In CI mode, first run the model downloader to completion
    log_info "Downloading test model..."
    docker compose -f "$COMPOSE_FILE" up --build model-downloader
    if [[ $? -ne 0 ]]; then
        log_error "Failed to download model"
        exit 1
    fi
    
    # Then run server and client (without model-downloader triggering abort)
    log_info "Running server and client..."
    COMPOSE_CMD="docker compose -f $COMPOSE_FILE up $BUILD_ARGS --abort-on-container-exit --no-deps server client"
else
    # Local mode: run all together
    COMPOSE_CMD="docker compose -f $COMPOSE_FILE up $BUILD_ARGS --abort-on-container-exit"
fi

if [[ "$VERBOSE" == true ]]; then
    $COMPOSE_CMD
else
    $COMPOSE_CMD 2>&1 | grep -v "^\s*$" | while read -r line; do
        echo "$line"
    done
fi

# Get container names
CLIENT_CONTAINER="llamacpp-client"
SERVER_CONTAINER="llamacpp-server"
if [[ "$CI_MODE" == true ]]; then
    CLIENT_CONTAINER="llamacpp-client-ci"
    SERVER_CONTAINER="llamacpp-server-ci"
fi

# Check server exit code first (if server failed, test failed)
SERVER_EXIT_CODE=$(docker inspect "$SERVER_CONTAINER" --format='{{.State.ExitCode}}' 2>/dev/null || echo "0")
if [[ "$SERVER_EXIT_CODE" != "0" ]]; then
    log_error "Server failed with exit code: $SERVER_EXIT_CODE"
    exit "$SERVER_EXIT_CODE"
fi

# Get the exit code of the client container
EXIT_CODE=$(docker inspect "$CLIENT_CONTAINER" --format='{{.State.ExitCode}}' 2>/dev/null || echo "1")

exit "$EXIT_CODE"


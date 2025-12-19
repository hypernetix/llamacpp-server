<#
.SYNOPSIS
    Integration test script for llamacpp gRPC server (Windows PowerShell version)

.DESCRIPTION
    Runs Docker-based integration tests with configurable parameters

.PARAMETER Model
    Path to local GGUF model file

.PARAMETER ModelUrl
    URL to download the model from

.PARAMETER CI
    Use CI defaults (download SmolLM2-135M)

.PARAMETER TestMode
    Test mode: baseline, greedy, seeded, stress (default: greedy)

.PARAMETER NoCleanup
    Don't remove containers after test

.PARAMETER Build
    Force rebuild Docker images

.PARAMETER Verbose
    Show verbose output

.EXAMPLE
    .\scripts\integration-test.ps1 -Model C:\models\model.gguf
    .\scripts\integration-test.ps1 -CI
    .\scripts\integration-test.ps1 -ModelUrl https://example.com/model.gguf
#>

param(
    [string]$Model,
    [string]$ModelUrl,
    [switch]$CI,
    [string]$TestMode = "greedy",
    [switch]$NoCleanup,
    [switch]$Build,
    [switch]$VerboseOutput
)

$ErrorActionPreference = "Stop"

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Get project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# CI defaults
$CIModelUrl = "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q4_k_m.gguf"
$CIModelName = "smollm2-135m-instruct-q4_k_m.gguf"

# Change to project root
Push-Location $ProjectRoot

$TempModelDir = $null
$ComposeFile = "docker/docker-compose.yml"

try {
    # Validate inputs
    if ($CI) {
        Write-Info "Running in CI mode with default test model"
        $ComposeFile = "docker/docker-compose.ci.yml"
    }
    elseif ($ModelUrl) {
        Write-Info "Using model URL: $ModelUrl"
        $TempModelDir = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), [System.Guid]::NewGuid().ToString()))
        $ModelName = [System.IO.Path]::GetFileName($ModelUrl)
        $Model = Join-Path $TempModelDir.FullName $ModelName
        Write-Info "Downloading model to $Model..."
        Invoke-WebRequest -Uri $ModelUrl -OutFile $Model
    }
    elseif ($Model) {
        if (-not (Test-Path $Model)) {
            Write-Err "Model file not found: $Model"
            exit 1
        }
        Write-Info "Using local model: $Model"
    }
    else {
        Write-Err "Model is required. Use -Model, -ModelUrl, or -CI"
        exit 1
    }

    # Set environment variables
    $env:TEST_MODE = $TestMode
    if ($Model) {
        $env:MODEL_PATH = $Model
    }

    Write-Info "Starting integration test..."
    Write-Info "  Compose file: $ComposeFile"
    Write-Info "  Test mode: $TestMode"

    # Run docker compose
    Write-Info "Building and starting containers..."
    
    if ($CI) {
        # In CI mode, first run the model downloader to completion
        Write-Info "Downloading test model..."
        $DownloadArgs = @("-f", $ComposeFile, "up", "--build", "model-downloader")
        & docker compose @DownloadArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Err "Failed to download model"
            exit 1
        }
        
        # Then run server and client (without model-downloader)
        Write-Info "Running server and client..."
        $ComposeArgs = @("-f", $ComposeFile, "up", "--abort-on-container-exit", "--no-deps", "server", "client")
        if ($Build) {
            $ComposeArgs += "--build"
        }
    }
    else {
        # Local mode: run all together
        $ComposeArgs = @("-f", $ComposeFile, "up", "--abort-on-container-exit")
        if ($Build) {
            $ComposeArgs += "--build"
        }
    }

    & docker compose @ComposeArgs
    $DockerExitCode = $LASTEXITCODE

    # Get client container exit code
    $ClientContainer = if ($CI) { "llamacpp-client-ci" } else { "llamacpp-client" }
    $ServerContainer = if ($CI) { "llamacpp-server-ci" } else { "llamacpp-server" }
    
    # Check server exit code first (if server failed, test failed)
    $ServerExitCode = docker inspect $ServerContainer --format='{{.State.ExitCode}}' 2>$null
    if ($ServerExitCode -and [int]$ServerExitCode -ne 0) {
        Write-Err "Integration test FAILED (server exit code: $ServerExitCode)"
        exit [int]$ServerExitCode
    }
    
    # Check client exit code
    $ClientExitCode = docker inspect $ClientContainer --format='{{.State.ExitCode}}' 2>$null
    if (-not $ClientExitCode) { $ClientExitCode = 1 }
    
    # Also fail if docker compose itself failed
    if ($DockerExitCode -ne 0 -and [int]$ClientExitCode -eq 0) {
        $ClientExitCode = $DockerExitCode
    }

    if ([int]$ClientExitCode -eq 0) {
        Write-Info "Integration test PASSED"
    }
    else {
        Write-Err "Integration test FAILED (exit code: $ClientExitCode)"
    }

    exit [int]$ClientExitCode
}
finally {
    # Temporarily allow errors for cleanup (docker compose outputs to stderr)
    $ErrorActionPreference = "Continue"
    
    # Cleanup
    if (-not $NoCleanup) {
        Write-Info "Cleaning up containers..."
        # Run cleanup, ignore any errors
        try {
            docker compose -f $ComposeFile down -v --remove-orphans 2>&1 | Out-Null
        } catch {
            # Ignore cleanup errors
        }
    }

    # Clean up temp model
    if ($TempModelDir -and (Test-Path $TempModelDir)) {
        Remove-Item -Recurse -Force $TempModelDir -ErrorAction SilentlyContinue
    }

    Pop-Location
}


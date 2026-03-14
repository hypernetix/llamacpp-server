<#
.SYNOPSIS
    Integration test script for llamacpp server (Windows PowerShell version)

.DESCRIPTION
    Runs Docker-based integration tests with configurable parameters.
    Tests both gRPC and HTTP transports sequentially.

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

# CI defaults (SmolLM2-135M from bartowski)
$CIModelUrl = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
$CIModelName = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

# Change to project root
Push-Location $ProjectRoot

$TempModelDir = $null
$ComposeFile = "docker/docker-compose.yml"

# Container names
$ServerContainer = "llamacpp-server"
$ClientGrpcContainer = "llamacpp-client-grpc"
$ClientHttpContainer = "llamacpp-client-http"

function Run-TransportTest {
    param(
        [string]$Transport,
        [string]$ClientService,
        [string]$ClientContainer,
        [bool]$DoBuild
    )

    Write-Info ""
    Write-Info "========================================="
    Write-Info "  Running $Transport integration test"
    Write-Info "========================================="

    $ComposeArgs = @("-f", $ComposeFile, "up", "--abort-on-container-exit", "--exit-code-from", $ClientService, "--no-deps", "server", $ClientService)
    if ($DoBuild) {
        $ComposeArgs += "--build"
    }

    Write-Info "Running: docker compose $($ComposeArgs -join ' ')"

    # Temporarily allow stderr from native commands — docker compose writes
    # status messages (e.g. "Container ... Creating") to stderr which PowerShell
    # treats as terminating errors under $ErrorActionPreference = "Stop" + 2>&1.
    $savedErrorPref = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"

    # Capture all output to prevent pipeline leaking (which would corrupt the return value)
    $composeOutput = & docker compose @ComposeArgs 2>&1
    $composeExitCode = $LASTEXITCODE

    $ErrorActionPreference = $savedErrorPref

    # Display captured output via Write-Host (bypasses pipeline)
    foreach ($line in $composeOutput) {
        Write-Host $line
    }

    Write-Info "Docker compose exit code: $composeExitCode"

    if ($composeExitCode -ne 0) {
        # Double-check actual client container exit code
        $savedErrorPref2 = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        $clientExit = & docker inspect $ClientContainer --format='{{.State.ExitCode}}' 2>&1
        $ErrorActionPreference = $savedErrorPref2
        Write-Err "$Transport test FAILED (compose exit: $composeExitCode, client exit: $clientExit)"
        return $composeExitCode
    }

    Write-Info "$Transport test PASSED"
    return 0
}

try {
    # Validate inputs
    if ($CI) {
        Write-Info "Running in CI mode with default test model"
        $ComposeFile = "docker/docker-compose.ci.yml"
        $ServerContainer = "llamacpp-server-ci"
        $ClientGrpcContainer = "llamacpp-client-grpc-ci"
        $ClientHttpContainer = "llamacpp-client-http-ci"
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

    # Build and prepare
    Write-Info "Building and starting containers..."

    if ($CI) {
        Write-Info "Downloading test model..."
        $DownloadArgs = @("-f", $ComposeFile, "up", "--build", "model-downloader")
        $savedErrorPref = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        & docker compose @DownloadArgs
        $dlExitCode = $LASTEXITCODE
        $ErrorActionPreference = $savedErrorPref
        if ($dlExitCode -ne 0) {
            Write-Err "Failed to download model"
            exit 1
        }
    }

    # Run gRPC test (with --build if requested)
    $GrpcExit = Run-TransportTest -Transport "gRPC" -ClientService "client-grpc" -ClientContainer $ClientGrpcContainer -DoBuild $Build.IsPresent

    if ($GrpcExit -ne 0) {
        Write-Err "gRPC test failed (exit code: $GrpcExit), skipping HTTP test"
        exit $GrpcExit
    }

    # Run HTTP test (no rebuild needed, images already built)
    $HttpExit = Run-TransportTest -Transport "HTTP" -ClientService "client-http" -ClientContainer $ClientHttpContainer -DoBuild $false

    if ($HttpExit -ne 0) {
        exit $HttpExit
    }

    # Run parallel inference test (multi-slot server)
    if ($CI) {
        $ClientParallelContainer = "llamacpp-client-grpc-cb-ci"
    } else {
        $ClientParallelContainer = "llamacpp-client-grpc-cb"
    }

    $ParallelExit = Run-TransportTest -Transport "Parallel" -ClientService "client-grpc-cb" -ClientContainer $ClientParallelContainer -DoBuild $false

    if ($ParallelExit -ne 0) {
        exit $ParallelExit
    }

    Write-Info ""
    Write-Info "========================================="
    Write-Info "  All tests PASSED (gRPC, HTTP, Parallel)"
    Write-Info "========================================="

    exit 0
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

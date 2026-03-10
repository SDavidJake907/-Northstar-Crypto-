param(
    [switch]$SkipGpu,
    [switch]$SkipNpu,
    [switch]$SkipBook,
    [switch]$SkipRust,
    [ValidateSet("Debug", "Release")]
    [string]$Config = "Release"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$GpuDir = Join-Path $ProjectRoot "cpp\gpu_infer"
$NpuDir = Join-Path $ProjectRoot "cpp\npu_infer"
$BookDir = Join-Path $ProjectRoot "cpp\book_engine"
$RustDir = Join-Path $ProjectRoot "rust_core"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
    Write-Host "OK: $Name" -ForegroundColor Green
}

function Build-CmakeProject {
    param(
        [string]$SourceDir,
        [string]$BuildDir,
        [string]$ConfigName
    )
    cmake -S $SourceDir -B $BuildDir | Out-Host
    cmake --build $BuildDir --config $ConfigName | Out-Host
}

Push-Location $ProjectRoot
try {
    if (-not $SkipGpu) {
        Invoke-Step "Build GPU DLL (hk_gpu_infer)" {
            Push-Location $GpuDir
            try {
                cmd /c build_gpu.bat | Out-Host
            }
            finally {
                Pop-Location
            }
        }
    }
    else {
        Write-Host "SKIP: GPU DLL build" -ForegroundColor Yellow
    }

    if (-not $SkipNpu) {
        $npuBuild = Join-Path $NpuDir "build"
        Invoke-Step "Build NPU DLL (hk_npu_infer)" {
            Build-CmakeProject -SourceDir $NpuDir -BuildDir $npuBuild -ConfigName $Config
        }
    }
    else {
        Write-Host "SKIP: NPU DLL build" -ForegroundColor Yellow
    }

    if (-not $SkipBook) {
        $bookBuild = Join-Path $BookDir "build"
        Invoke-Step "Build BookEngine DLL (book_engine)" {
            Build-CmakeProject -SourceDir $BookDir -BuildDir $bookBuild -ConfigName $Config
        }
    }
    else {
        Write-Host "SKIP: BookEngine DLL build" -ForegroundColor Yellow
    }

    if (-not $SkipRust) {
        Invoke-Step "Build Rust core (hybrid_kraken_core)" {
            Push-Location $RustDir
            try {
                cargo build --release | Out-Host
            }
            finally {
                Pop-Location
            }
        }
    }
    else {
        Write-Host "SKIP: Rust build" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "Build pipeline complete." -ForegroundColor Green
}
finally {
    Pop-Location
}

<#
.SYNOPSIS
  Clones and builds the llama.cpp programs DualLLM Portal needs, on Windows (CUDA + RPC).

.DESCRIPTION
  Builds three programs into <repo>\llama.cpp\build\bin\Release:
    llama-server        serves the model and the chat web UI
    llama-fit-params    lets the portal work out which context sizes fit
    ggml-rpc-server     the worker for devices served from this PC to another machine (optional)

  Every machine that takes part in a split must run the SAME llama.cpp commit - build the workers
  with scripts/build-llamacpp.sh at the same -Ref (and the same GGML_MAX_NAME if you change it; see
  -ExtraCMakeArgs). Mismatched builds fail in confusing ways.

  Prerequisites: Git, CMake 3.24+, Visual Studio 2022 (or Build Tools) with the "Desktop development
  with C++" workload, and the NVIDIA CUDA Toolkit.

.PARAMETER Ref
  Git branch, tag or commit to build. Defaults to the repository's default branch. Pin a commit to
  reproduce a build on your other machines.

.PARAMETER CudaArchitectures
  Value for CMAKE_CUDA_ARCHITECTURES. "native" builds for the GPUs installed in this PC; use a list
  such as "86;75" to also cover cards you only have elsewhere.

.PARAMETER ExtraCMakeArgs
  Anything else to pass to the cmake configure step, e.g. @('-DCMAKE_C_FLAGS=...').

.EXAMPLE
  .\scripts\build-llamacpp.ps1
  .\scripts\build-llamacpp.ps1 -Ref 1d2869c -CudaArchitectures '120;86;75'
#>
[CmdletBinding()]
param(
    [string]$Ref,
    [string]$CudaArchitectures = 'native',
    [string[]]$ExtraCMakeArgs = @(),
    [string]$RepoUrl = 'https://github.com/ggml-org/llama.cpp.git'
)

$ErrorActionPreference = 'Stop'

foreach ($tool in 'git', 'cmake') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool was not found on PATH." }
}
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue) -and -not $env:CUDA_PATH) {
    throw 'The CUDA Toolkit was not found (no nvcc on PATH and CUDA_PATH is not set). Install it from https://developer.nvidia.com/cuda-downloads.'
}

$repo = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repo 'llama.cpp'

if (-not (Test-Path (Join-Path $source '.git'))) {
    git clone $RepoUrl $source
    if ($LASTEXITCODE -ne 0) { throw 'git clone failed.' }
}
if ($Ref) {
    git -C $source fetch --tags origin
    git -C $source checkout $Ref
    if ($LASTEXITCODE -ne 0) { throw "Could not check out $Ref." }
}
Write-Host ('Building llama.cpp at ' + (git -C $source rev-parse --short HEAD))

$build = Join-Path $source 'build'
$configure = @('-S', $source, '-B', $build, '-DGGML_CUDA=ON', '-DGGML_RPC=ON', "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures") + $ExtraCMakeArgs
cmake @configure
if ($LASTEXITCODE -ne 0) { throw 'cmake configure failed.' }

cmake --build $build --config Release --target llama-server llama-fit-params ggml-rpc-server --parallel
if ($LASTEXITCODE -ne 0) { throw 'cmake build failed.' }

$bin = Join-Path $build 'bin\Release'
Write-Host "Done. Programs are in $bin"
Write-Host 'The default duallm.config.json paths already point there.'

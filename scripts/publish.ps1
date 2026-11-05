<#
.SYNOPSIS
  Builds DualLlmPortal.exe and installs it, with its native DLLs, next to your config.

.DESCRIPTION
  Publishes to a staging folder (.build\publish), starts the staged exe once to prove it comes up and
  answers its control API, and only then copies it over the installed one. Do not run a hand-typed
  `dotnet publish -o <repo root>` instead: publishing straight into the project's own folder tree can
  produce an exe with no XAML in it, which builds and links fine and then dies on startup.

  A running DualLlmPortal.exe is renamed to DualLlmPortal.exe.old rather than overwritten, so the
  install works even while the portal is open; restart it to pick up the new build.

.PARAMETER OutDir
  Where to install. Defaults to the repository root, which is the portal's "home" folder: it looks for
  duallm.config.json, llm_models and llama.cpp next to the exe.

.PARAMETER SkipVerify
  Skip the start-up check.

.EXAMPLE
  .\scripts\publish.ps1
#>
[CmdletBinding()]
param(
    [string]$OutDir,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
if (-not $OutDir) { $OutDir = $repo }
$project = Join-Path $repo 'DualLlmPortal\DualLlmPortal.csproj'
$staging = Join-Path $repo '.build\publish'

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw 'The .NET SDK was not found. Install the .NET 10 SDK from https://dotnet.microsoft.com/download.'
}

if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
dotnet publish $project -c Release -o $staging
if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed (exit code $LASTEXITCODE)." }

function Test-StagedPortal([string]$exe) {
    # Isolated home and config, so the check never touches a real config or a running portal's port.
    $home_ = Join-Path $repo ('.build\verify-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory $home_ | Out-Null
    $port = Get-Random -Minimum 20000 -Maximum 30000
    $config = Join-Path $home_ 'duallm.config.json'
    $json = '{ "controlApi": { "port": ' + $port + ' }, "devices": [ { "id": "gpu0", "type": "gpu", "label": "GPU 0" } ] }'
    Set-Content -Path $config -Value $json
    $env:DUALLM_HOME = $home_
    $env:DUALLM_CONFIG = $config
    $process = $null
    try {
        $process = Start-Process -FilePath $exe -ArgumentList '--headless' -PassThru -WindowStyle Hidden
        $deadline = (Get-Date).AddSeconds(30)
        while ((Get-Date) -lt $deadline) {
            if ($process.HasExited) { throw "the portal exited straight away (exit code $($process.ExitCode))." }
            try {
                $reply = Invoke-RestMethod "http://127.0.0.1:$port/control/health" -TimeoutSec 2
                if ($reply.ok) { return }
            } catch { }
            Start-Sleep -Milliseconds 500
        }
        throw 'the portal started but its control API did not answer within 30 s.'
    }
    finally {
        if ($process -and -not $process.HasExited) { & taskkill /PID $process.Id /T /F | Out-Null; $process.WaitForExit(5000) | Out-Null }
        Remove-Item Env:DUALLM_HOME, Env:DUALLM_CONFIG -ErrorAction SilentlyContinue
        Remove-Item $home_ -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $SkipVerify) {
    Write-Host 'Checking that the new build starts...'
    try { Test-StagedPortal (Join-Path $staging 'DualLlmPortal.exe') }
    catch { throw "The staged build is broken: $_ Nothing was installed." }
    Write-Host 'Build starts and answers its control API.'
}

$target = Join-Path $OutDir 'DualLlmPortal.exe'
if (Test-Path $target) {
    # Windows lets a running exe be renamed but not overwritten.
    try { Move-Item $target "$target.old" -Force }
    catch { throw "Cannot replace $target. Close the portal and run this again. ($_)" }
}
# Copy only what changed: the native sidecar DLLs are identical between builds and are locked while the
# portal is running, so overwriting them unconditionally would fail for no reason.
Get-ChildItem $staging -Recurse -File | ForEach-Object {
    $relative = $_.FullName.Substring($staging.Length).TrimStart('')
    $destination = Join-Path $OutDir $relative
    if ((Test-Path $destination) -and ((Get-Item $destination).Length -eq $_.Length) -and
        ((Get-FileHash $destination).Hash -eq (Get-FileHash $_.FullName).Hash)) { return }
    New-Item -ItemType Directory -Force (Split-Path $destination) | Out-Null
    Copy-Item $_.FullName $destination -Force
}

Write-Host "Installed to $OutDir"
if (Test-Path "$target.old") {
    Write-Host 'The previous exe was kept as DualLlmPortal.exe.old; restart the portal to use the new build.'
}
if (-not (Test-Path (Join-Path $OutDir 'duallm.config.json'))) {
    Write-Host 'No duallm.config.json yet: the first run writes one for your local GPUs. For other machines, start from duallm.example.json.'
}

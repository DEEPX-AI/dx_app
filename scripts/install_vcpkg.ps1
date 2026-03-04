<#
.SYNOPSIS
  Automatically install vcpkg in the project root (optionally install dependencies)
  and register the path to the user environment variable VCPKG_ROOT.

.DESCRIPTION
  - Default behavior: treat the parent of the scripts folder as the project root and use a vcpkg folder beneath it.
  - Reuse an existing installation if present.
  - Clone vcpkg from git and bootstrap it if necessary.
  - Optionally run dependency installation via manifest or standard install.
  - Optionally register the VCPKG_ROOT environment variable for the current user (requires appropriate permissions).

.PARAMETER TargetPath
  Path to install vcpkg. Default: <repoRoot>\vcpkg

.PARAMETER Triplet
  vcpkg triplet (default: x64-windows)

.PARAMETER Install
  Run vcpkg install to install dependencies. If vcpkg.json exists in the project root, manifest mode will be used.

.PARAMETER SetUserEnv
  Register VCPKG_ROOT as a persistent user environment variable after installation. Default: $false

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\scripts\install_vcpkg.ps1 -Install
#>

param(
  [string]$TargetPath,
  [string]$Triplet = "x64-windows",
  [switch]$Install,
  [switch]$SetUserEnv = $false
)

function Abort([string]$msg) {
  Write-Error $msg
  exit 1
}

# Resolve repo root (parent of scripts folder)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = (Resolve-Path (Join-Path $scriptDir ".." )).Path

if (-not $TargetPath) {
  $TargetPath = Join-Path $repoRoot "vcpkg"
}

Write-Host "Repository root: $repoRoot"
Write-Host "Requested target vcpkg path: $TargetPath"

# Prefer an already-installed vcpkg referenced by environment variable
$envVcpkgRoot = $env:VCPKG_ROOT
$vcpkgRootUsed = $null
$vcpkgExeRel = "vcpkg.exe"

if ($envVcpkgRoot) {
  $envVcpkgExe = Join-Path $envVcpkgRoot $vcpkgExeRel
  if (Test-Path $envVcpkgExe) {
    Write-Host "Using existing vcpkg from environment VCPKG_ROOT: $envVcpkgRoot"
    $vcpkgRootUsed = (Get-Item $envVcpkgRoot).FullName
  } else {
    Write-Warning "VCPKG_ROOT is set but vcpkg.exe was not found at: $envVcpkgExe. Ignoring and continuing."
  }
}

# If no env vcpkg, check project-local vcpkg
if (-not $vcpkgRootUsed) {
  $projVcpkgExe = Join-Path $TargetPath $vcpkgExeRel
  if (Test-Path $projVcpkgExe) {
    Write-Host "Using existing project-local vcpkg at: $TargetPath"
    $vcpkgRootUsed = (Get-Item $TargetPath).FullName
  }
}

# If still not found, clone into project-local TargetPath and bootstrap
if (-not $vcpkgRootUsed) {
  Write-Host "No existing vcpkg found; attempting to install project-local vcpkg into: $TargetPath"

  # Prompt user to confirm installation
  try {
    $answer = Read-Host "Install vcpkg into '$TargetPath'? (y/n)"
  } catch {
    Write-Warning "Unable to read user input. Aborting installation."
    exit 1
  }
  if (-not $answer) {
    Write-Host "No response received. Exiting without installing vcpkg."
    exit 0
  }
  if ($answer.Trim().ToLower().StartsWith('y')) {
    Write-Host "User confirmed vcpkg installation. Proceeding..."
  } else {
    Write-Host "User declined vcpkg installation. Exiting."
    exit 0
  }

  # Check git availability only when we need to clone
  $gitCmd = Get-Command git -ErrorAction SilentlyContinue
  if (-not $gitCmd) {
    Abort "git is not available on PATH. Installing project-local vcpkg requires git. Either install git or set VCPKG_ROOT to an existing vcpkg installation."
  }

  $cloneUrl = "https://github.com/microsoft/vcpkg.git"
  & git clone $cloneUrl $TargetPath
  if ($LASTEXITCODE -ne 0) {
    Abort "git clone failed (exit $LASTEXITCODE)."
  }

  $vcpkgExe = Join-Path $TargetPath $vcpkgExeRel
  if (-not (Test-Path $vcpkgExe)) {
    Write-Host "Performing vcpkg bootstrap..."
    $bootstrap = Join-Path $TargetPath "bootstrap-vcpkg.bat"
    if (-not (Test-Path $bootstrap)) {
      Abort "bootstrap-vcpkg.bat not found: $bootstrap"
    }
    Push-Location $TargetPath
    try {
      & cmd /c "bootstrap-vcpkg.bat"
      if ($LASTEXITCODE -ne 0) {
        Abort "vcpkg bootstrap failed (exit $LASTEXITCODE)."
      }
    } finally {
      Pop-Location
    }
  }
  $vcpkgRootUsed = (Get-Item $TargetPath).FullName
}

# Ensure vcpkg executable exists at chosen root
$vcpkgExePath = Join-Path $vcpkgRootUsed $vcpkgExeRel
if (-not (Test-Path $vcpkgExePath)) {
  Abort "vcpkg.exe does not exist at the selected path: $vcpkgExePath"
}

# Export for current session
$env:VCPKG_ROOT = $vcpkgRootUsed
Write-Host "VCPKG_ROOT for current session set to: $vcpkgRootUsed"

# (Optional) install dependencies: manifest mode if vcpkg.json exists, otherwise default
if ($Install) {
  Push-Location $repoRoot
  try {
    if (Test-Path (Join-Path $repoRoot "vcpkg.json")) {
      Write-Host "vcpkg.json found: installing in manifest mode..."
      & $vcpkgExePath install --triplet $Triplet
    } else {
      Write-Host "vcpkg.json not found: proceeding with default 'vcpkg install --triplet'..."
      & $vcpkgExePath install --triplet $Triplet
    }

    if ($LASTEXITCODE -ne 0) {
      Abort "vcpkg install failed (exit $LASTEXITCODE)."
    }
  } finally {
    Pop-Location
  }
}

# Optionally register user environment variable
if ($SetUserEnv) {
  try {
    [Environment]::SetEnvironmentVariable("VCPKG_ROOT", $vcpkgRootUsed, "User")
    Write-Host "VCPKG_ROOT registered in user environment variables: $vcpkgRootUsed"
    Write-Host "Note: This will not affect already-running Visual Studio instances or open terminals. Restart or open a new terminal to see the change."
  } catch {
    Write-Warning "Warning: error occurred while setting user environment variable: $($_.Exception.Message)"
  }
}

Write-Host "vcpkg installation and configuration complete."
Write-Host "Recommended next steps:"
Write-Host " - Use CMAKE_TOOLCHAIN_FILE='$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake' when configuring CMake, or"
Write-Host " - Update the CMAKE_TOOLCHAIN_FILE entry in CMakeSettings.json to point to the project-relative path to the toolchain file."
Write-Host ""
Write-Host "Example: powershell -ExecutionPolicy Bypass -File .\scripts\install_vcpkg.ps1 -Install"

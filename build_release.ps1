param(
    [string]$Python = "python",
    [string]$DistPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildEnvironment = Join-Path $ProjectRoot ".build-venv"
$BuildPython = Join-Path $BuildEnvironment "Scripts\python.exe"
if (-not $DistPath) {
    $DistPath = Join-Path $ProjectRoot "dist"
}

# $ErrorActionPreference does not apply to native executables, so every
# external command must have its exit code checked explicitly.
function Invoke-Checked {
    param([string]$Description, [scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $BuildPython)) {
    Invoke-Checked "Creating the build environment" {
        & $Python -m venv $BuildEnvironment
    }
}

Invoke-Checked "Upgrading pip" {
    & $BuildPython -m pip install --upgrade pip
}
Invoke-Checked "Installing locked requirements" {
    & $BuildPython -m pip install --requirement (
        Join-Path $ProjectRoot "requirements-lock.txt"
    )
}
Invoke-Checked "Unit tests" {
    & $BuildPython -m unittest discover -s (
        Join-Path $ProjectRoot "tests"
    ) -v
}

$Executable = Join-Path $DistPath "Behavioral_Maze_Analyzer.exe"
$Running = Get-Process -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $Executable }
if ($Running) {
    throw (
        "The executable is in use (PID " + ($Running.Id -join ", ") +
        "). Close the application or pass -DistPath to build elsewhere."
    )
}

Invoke-Checked "PyInstaller build" {
    & $BuildPython -m PyInstaller --clean --noconfirm --distpath $DistPath (
        Join-Path $ProjectRoot "Behavioral_Maze_Analyzer.spec"
    )
}
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "PyInstaller reported success but $Executable was not created."
}

$StartupCheck = Start-Process `
    -FilePath $Executable `
    -ArgumentList "--startup-check" `
    -WindowStyle Hidden `
    -PassThru
if (-not $StartupCheck.WaitForExit(60000)) {
    Stop-Process -Id $StartupCheck.Id -Force -ErrorAction SilentlyContinue
    throw "Packaged startup check timed out; an error dialog may be open."
}
$StartupCheck.Refresh()
if ($StartupCheck.ExitCode -ne 0) {
    throw (
        "Packaged startup check failed with exit code " +
        $StartupCheck.ExitCode
    )
}

$Hash = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
Write-Host ("Release created at " + $Executable)
Write-Host ("SHA-256: " + $Hash)

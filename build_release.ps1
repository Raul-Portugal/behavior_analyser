param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildEnvironment = Join-Path $ProjectRoot ".build-venv"
$BuildPython = Join-Path $BuildEnvironment "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $BuildPython)) {
    & $Python -m venv $BuildEnvironment
}

& $BuildPython -m pip install --upgrade pip
& $BuildPython -m pip install --requirement (
    Join-Path $ProjectRoot "requirements-lock.txt"
)
& $BuildPython -m unittest discover -s (
    Join-Path $ProjectRoot "tests"
) -v
& $BuildPython -m PyInstaller --clean --noconfirm (
    Join-Path $ProjectRoot "Behavioral_Maze_Analyzer.spec"
)

Write-Host (
    "Release created at " +
    (Join-Path $ProjectRoot "dist\Behavioral_Maze_Analyzer.exe")
)

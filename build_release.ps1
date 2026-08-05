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

$Executable = Join-Path (
    Join-Path $ProjectRoot "dist"
) "Behavioral_Maze_Analyzer.exe"
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

Write-Host (
    "Release created at " +
    $Executable
)

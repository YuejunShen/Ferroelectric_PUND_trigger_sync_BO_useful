$python = "C:\Users\16502\.conda\envs\bto-xopt-latest\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "The bto-xopt-latest Conda environment was not found at $python"
}

Push-Location $PSScriptRoot
try {
    & $python (Join-Path $PSScriptRoot "run_experiment.py")
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

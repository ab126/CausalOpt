$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = $null
foreach ($version in @("3.10", "3.9", "3.8")) {
    try { $candidate = & py "-$version" -c "import sys; print(sys.executable)" 2>$null }
    catch { $candidate = $null }
    if ($LASTEXITCODE -eq 0 -and $candidate) { $python = $candidate.Trim(); break }
}
if (-not $python) { throw "Python 3.10, 3.9, or 3.8 is required for CausalNex 0.12.1." }
$venv = Join-Path $repo ".venv-dynotears"
& $python -m venv $venv
$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $repo "requirements-dynotears-lock.txt")
& $venvPython -c "import sys, causalnex, numpy, scipy; from causalnex.structure.dynotears import from_numpy_dynamic; print('python',sys.version); print('causalnex',causalnex.__version__); print('numpy',numpy.__version__); print('scipy',scipy.__version__)"

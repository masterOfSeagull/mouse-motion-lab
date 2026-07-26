[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command py -ErrorAction SilentlyContinue)) { throw 'Install Python 3.12, then rerun this script.' }
& py -3.12 -c "import sys; assert sys.version_info[:2] == (3, 12)" 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required; install it or make py -3.12 available.' }
if (-not (Test-Path "$ProjectRoot\.venv\Scripts\python.exe")) { & py -3.12 -m venv "$ProjectRoot\.venv" }
& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install --upgrade pip
& "$ProjectRoot\.venv\Scripts\python.exe" -m pip install -e "$ProjectRoot[dev,training]"

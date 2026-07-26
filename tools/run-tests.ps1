[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PytestArgs)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:QT_QPA_PLATFORM = 'offscreen'
if (-not (Test-Path -LiteralPath "$ProjectRoot\build\native\Release\mousegen_cli.exe")) {
    & "$ProjectRoot\tools\build-native.ps1"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& "$ProjectRoot\.venv\Scripts\python.exe" -m pytest @PytestArgs

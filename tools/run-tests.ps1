[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$PytestArgs)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:QT_QPA_PLATFORM = 'offscreen'
& "$ProjectRoot\.venv\Scripts\python.exe" -m pytest @PytestArgs

[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArgs)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
& "$ProjectRoot\.venv\Scripts\python.exe" -m apps.control_panel.main @AppArgs

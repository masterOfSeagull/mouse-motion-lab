[CmdletBinding()]
param([switch]$Clean)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $ProjectRoot 'build\native'
if ($Clean -and (Test-Path $BuildRoot)) { Remove-Item -Recurse -Force $BuildRoot }
$Cmake = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $Cmake) {
  $candidate = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter cmake.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
  if (-not $candidate) { throw 'CMake was not found on PATH or in Visual Studio.' }
  $CmakePath = $candidate
} else { $CmakePath = $Cmake.Source }
& $CmakePath -S "$ProjectRoot\native" -B $BuildRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $CmakePath --build $BuildRoot --config Release
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

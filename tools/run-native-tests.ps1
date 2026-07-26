[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CTest = Get-Command ctest -ErrorAction SilentlyContinue
if (-not $CTest) {
    $candidate = Get-ChildItem 'C:\Program Files\Microsoft Visual Studio' -Recurse -Filter ctest.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
    if (-not $candidate) { throw 'CTest was not found on PATH or in Visual Studio.' }
    $CTestPath = $candidate
} else { $CTestPath = $CTest.Source }
& $CTestPath --test-dir "$ProjectRoot\build\native" -C Release --output-on-failure
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

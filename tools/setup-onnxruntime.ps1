[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Version = '1.21.0'
$ExpectedSha256 = '5C07BB2805CD666DDA75FA9BFA60E75F2F90D478B952298DD9D55C00740D81BF'
$CacheRoot = Join-Path $ProjectRoot 'build\dependencies'
$Archive = Join-Path $CacheRoot "onnxruntime-win-x64-$Version.zip"
$SdkRoot = Join-Path $CacheRoot "onnxruntime-win-x64-$Version"
$Uri = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/onnxruntime-win-x64-$Version.zip"
New-Item -ItemType Directory -Force -Path $CacheRoot | Out-Null
if (Test-Path -LiteralPath $Archive) {
    $Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
    if ($Actual -ne $ExpectedSha256) { Remove-Item -LiteralPath $Archive -Force }
}
if (-not (Test-Path -LiteralPath $Archive)) {
    Invoke-WebRequest -Uri $Uri -OutFile $Archive
}
$Actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
if ($Actual -ne $ExpectedSha256) { throw "ONNX Runtime SDK checksum mismatch: $Actual" }
if (-not (Test-Path -LiteralPath (Join-Path $SdkRoot 'include\onnxruntime_cxx_api.h'))) {
    if (Test-Path -LiteralPath $SdkRoot) { Remove-Item -LiteralPath $SdkRoot -Recurse -Force }
    Expand-Archive -LiteralPath $Archive -DestinationPath $CacheRoot
}
Write-Output $SdkRoot

@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup-dev.ps1"
  if errorlevel 1 exit /b %errorlevel%
)

".venv\Scripts\python.exe" -c "import torch, onnx, onnxruntime" >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\setup-dev.ps1"
  if errorlevel 1 exit /b %errorlevel%
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\build-native.ps1"
if errorlevel 1 exit /b %errorlevel%

".venv\Scripts\pythonw.exe" -m apps.control_panel.main
exit /b %errorlevel%

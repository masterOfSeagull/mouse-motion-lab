# Portable generator runtime

`tools/setup-onnxruntime.ps1` downloads and SHA-256-verifies the official ONNX Runtime 1.21.0 Windows x64 SDK into `build/dependencies`. `tools/build-native.ps1` passes that ignored cache to CMake and builds:

- `mousemotionlab_mousegen.dll`: C ABI and C++ wrapper from `include/mousemotionlab/mousegen_runtime.h`.
- `mousegen_cli.exe`: data-only JSON generation; it never injects input.
- `mousemotionlab_mousegen_tests.exe`: native API guard tests.

The runtime loads `velocity.onnx`, `manifest.json`, and `normalization.bin` from a portable export directory. It verifies every declared artifact hash before creating an ONNX Runtime session.

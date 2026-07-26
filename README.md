# MouseMotionLab

MouseMotionLab is a Windows-first mouse-motion research tool. Milestone 1 is complete. Milestone 2 is in progress: it now includes versioned collection metadata, explicit trial phase boundaries and cursor-at-appearance fields, a Windows Raw Input native library, a fullscreen Qt Quick target game, bounded Parquet raw-event persistence, and reversible session/trial review with trajectory previews. The deterministic 500-trial acceptance run passes; the real-hardware collection smoke test remains outstanding. It does not train models, generate trajectories, or play input back.

## Setup and run

```powershell
.\tools\setup-dev.ps1
.\tools\run-tests.ps1
.\tools\build-native.ps1
.\tools\run-control-panel.ps1
```

`run.bat` is the Programs Manager launcher. It creates the development environment and native Raw Input DLL if either is missing, then starts the control panel from this project folder.

The default data root is `%LOCALAPPDATA%\MouseMotionLab`; use `MOUSE_MOTION_LAB_DATA_ROOT` for a developer/test override. The worker stores job state and audit events in `app.db` there.

## Architecture

- `apps/control_panel`: PySide6 composition and presentation-only QML.
- `apps/worker`: deterministic diagnostics JSONL subprocess.
- `mouselearn/domain`: strict configuration and worker contracts.
- `mouselearn/storage`: data root, logs, migrations, repositories.
- `mouselearn/collection`: native capture adapter, bounded Parquet persistence, and Qt collection coordinator.
- `native`: C++20 version stub plus a Windows Raw Input DLL; it remains independent from Qt and Python.

Schema changes must be added as new SQLite migrations.

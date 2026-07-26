# MouseMotionLab

MouseMotionLab is a Windows-first mouse-motion research tool. Milestones 1 and 2 are complete; Milestone 3 is in progress. Collection uses cursor-relative, stratified target conditions, frame-presented target activation, explicit timeouts/terminal phases, an allocation-free native ring buffer, buffered Parquet row groups, and reversible session/trial review with trajectory previews. Earlier recordings are retained and marked legacy: their realized geometry is recomputed where possible, but their reaction-time measurements are not high-confidence ground truth. It does not train models, generate trajectories, or play input back.

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
- `mouselearn/collection`: cursor-relative scheduling, native capture adapter, buffered Parquet persistence, and Qt collection coordinator.
- `mouselearn/datasets` and `mouselearn/representation`: immutable snapshots plus stable canonical/spline/timing foundations.
- `native`: C++20 version stub plus a Windows Raw Input DLL; it remains independent from Qt and Python.

Schema changes must be added as new SQLite migrations.

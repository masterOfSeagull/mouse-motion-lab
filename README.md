# MouseMotionLab

MouseMotionLab is a Windows-first mouse-motion research tool. Milestones 1–8 are complete. Collection protocol 3 samples target centers uniformly over the feasible canvas with 12–36 logical-pixel radii. On the current Windows/Qt fullscreen path, a target is activated synchronously after it is made visible and its reaction timestamp is explicitly marked render-unconfirmed; preprocessing does not use reaction time. It uses explicit timeouts/terminal phases, an allocation-free native ring buffer, buffered Parquet row groups, and reversible session/trial review with trajectory previews. Dataset preprocessing runs in a worker, re-verifies raw hashes, and writes 64 equal-time canonical positions plus total movement duration per trial. Earlier recordings are retained and marked legacy or protocol-2 audit data; they are not selected for training.

The baseline pipeline loads immutable processed artifacts with the full 21-value screen-context condition vector, fits only the training split, and publishes nearest-neighbor retrieval and PCA/condition-binned Gaussian-mixture artifacts only after held-out correctness and same-seed validation. The Generator page builds both models in a worker process and previews 64-point equal-time trajectories without injecting input. Playback remains disabled.

Conditional flow matching uses the separately pinned PyTorch environment. Its worker records experiment losses and periodic resumable checkpoints, publishes only validated deterministic models, and exposes small and standard presets on the Experiments page. Flow, PCA-mixture, and retrieval models share the same preview runtime.

Validation reports compare held-out correctness and trajectory distributions. The Models page enforces hash-verified promotion gates and a single active model; generator previews classify requests using held-out condition-distance percentiles.

The active conditional-flow model exports to a standalone hash-verified ONNX package. PyTorch, Python ONNX Runtime, and the independent C++20 runtime share deterministic source generation and trajectory decoding with automated parity. Native setup downloads and checksum-verifies the pinned ONNX Runtime SDK under the ignored build tree; no SDK binaries are committed. External desktop playback remains disabled.

Milestone 8's local-canvas playback harness is desktop-validated. It never injects operating-system input, starts disarmed, requires Arm followed by Start, and aborts on Escape, focus loss, or substantial physical mouse movement. The control panel applies bounded responsive scaling so its complete design surface remains available from its 720x450 minimum through larger window sizes. The trajectory viewer fits the complete virtual desktop with preserved aspect ratio by default and provides 100-400% zoom with explicit two-axis scrollbars for detailed inspection.

## Setup and run

```powershell
.\tools\setup-dev.ps1
.\tools\run-tests.ps1
.\tools\build-native.ps1
.\tools\run-native-tests.ps1
.\tools\run-control-panel.ps1
```

`run.bat` is the Programs Manager launcher. It creates the development environment and native Raw Input DLL if either is missing, then starts the control panel from this project folder.

The default data root is `%LOCALAPPDATA%\MouseMotionLab`; use `MOUSE_MOTION_LAB_DATA_ROOT` for a developer/test override. The worker stores job state and audit events in `app.db` there.

## Architecture

- `apps/control_panel`: PySide6 composition and presentation-only QML.
- `apps/worker`: JSONL subprocesses for diagnostics and preprocessing.
- `mouselearn/domain`: strict configuration and worker contracts.
- `mouselearn/storage`: data root, logs, migrations, repositories.
- `mouselearn/collection`: cursor-relative scheduling, native capture adapter, buffered Parquet persistence, and Qt collection coordinator.
- `mouselearn/datasets`, `mouselearn/preprocessing`, and `mouselearn/representation`: immutable snapshots, verified processed artifacts, and stable canonical/spline/timing foundations.
- `mouselearn/models`: shared request/result contracts, snapshot-bound loading, retrieval and PCA-mixture baselines, validity-preserving decoding, artifact publication, and held-out validation.
- `mouselearn/models/conditional_flow.py` and `flow_training.py`: tracked CFM optimization, checkpoints, Euler/Heun inference, and validated model publication.
- `native`: C++20 version stub plus a Windows Raw Input DLL; it remains independent from Qt and Python.

Schema changes must be added as new SQLite migrations.

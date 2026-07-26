\# MouseMotionLab — Complete Build Specification



Build a Windows-first desktop application named \*\*MouseMotionLab\*\* for collecting a user’s point-to-target mouse movements, training personalized trajectory generators, validating generated movements against held-out recordings, managing model history, exporting deployable models, and generating timed mouse trajectories toward arbitrary circular targets.



The application is an HCI research and local automation tool. It must not include stealth features, anti-cheat bypasses, hidden background automation, target detection from third-party applications, or mechanisms intended to conceal synthetic input. Playback must be explicitly armed and disabled by default.

## Project progression tracking

### Completed: Milestone 1 — Foundation (2026-07-26)

The first vertical slice is implemented and verified. It provides the Python 3.12 project setup, strict version-1 configuration, versioned SQLite initialization, structured UTC JSON logs, a PySide6/QML control-panel shell, diagnostics, and a real JSONL diagnostic worker managed through `QProcess`. The native CMake target is intentionally a Qt/Python-independent version-stub only.

Verification completed: the Python/QML test suite passed (19 tests), the native C++20 stub configured and built, and `mouselearn doctor` completed against an isolated data root. The data-root contract is `%LOCALAPPDATA%\\MouseMotionLab` (or `MOUSE_MOTION_LAB_DATA_ROOT`) with `app.db`, `logs`, `raw_sessions`, `datasets`, `experiments`, `models`, `exports`, `cache`, and `temp`.

This milestone deliberately contains no input capture, recording, training, model generation, or playback. Existing migration 1 is a released baseline: subsequent schema changes must be appended as new migrations, not edits to that migration.

### Next: Milestone 2 — Collection

**Current status (2026-07-26): in progress.** Slices 1 through 3 are implemented: version-2 collection persistence, version-3 review metadata, and version-4 explicit phase markers/start-position fields; the separately testable Windows Raw Input DLL; and a Qt Quick collection session that uses a bounded background Parquet writer. Starting collection opens a separate distraction-free fullscreen target game; Escape and its visible Stop button end and save the session. Each new trial persists the actual physical cursor position when the target becomes visible, plus `inter_trial`, `target_visible`, and `trial_completed` boundaries. Future preprocessing will select only the target-visible-to-valid-click interval; inter-trial movement remains audit data by default. The Sessions page supports reviewing completed trials and reversible logical discard/restore of trials or full sessions, plus a trajectory preview plotted directly from the selected trial's phase-bounded raw Parquet events. Discarded data is excluded from future dataset snapshots while raw files and audit events remain intact. The collection UI only enables Start when both `mousemotionlab_mouse_io.dll` and PyArrow are available; overflow or writer failure terminates the session visibly. Automated persistence, QML-load, Python, and native ABI tests pass, but Milestone 2 is not yet complete.

The deterministic 500-trial synthetic-session acceptance test now passes. The Diagnostics page now exposes Raw Input readiness plus live state, observed-event count, buffer depth, overflow count, and QPC frequency. The recorded 20-trial hardware smoke session has phase markers and phase-bounded trajectory previews; the review cards use bounded action controls with scrolling, and each trajectory is time-coloured from violet through red (early to late) so path direction is visible even before training. Final acceptance remains contingent on confirming no overflow/error in that session. Do not begin Milestone 3 until that check is recorded here.

Milestone 2 will be implemented as four testable slices, in this order:

1. **Persistence contracts and migration 2.** Define typed session, trial, target-condition, click, capture-health, and raw-file-reference records; add the migration and repositories before any collection UI. Raw event samples will be written in bounded batches to Parquet under `raw_sessions`, while SQLite retains metadata and file references. If the Parquet writer is unavailable or its queue overflows, collection must fail visibly and record the reason; it must never silently drop events.
2. **Native Raw Input library.** Replace the version-stub-only role with a separately testable Windows `WM_INPUT` capture library that registers only for an active collection session. It will timestamp with `QueryPerformanceCounter`, record raw deltas, button flags, screen coordinates, device identity, and monitor/coordinate metadata, and expose an explicit capture-failure/overflow signal.
3. **Collection vertical slice.** Add a Python collection coordinator and an `AppController`-backed Qt Quick target game. Implement the documented trial state machine, reproducible balanced target selection, high-resolution target/click timestamps, session start/pause/stop, and the required on-screen capture status. QML remains presentation-only.
4. **Review and verification.** Add the trial viewer and collection diagnostics, then test migration/repository contracts, native event serialization, target-state transitions, Parquet persistence, overflow/failure handling, and a deterministic 500-trial synthetic-session run. Complete a manual Windows Raw Input smoke test before calling the milestone done.

Milestone 2 is complete only when its acceptance criteria below are met: a 500-trial session persists without data loss, target appearances and clicks have high-resolution timestamps, and both raw deltas and screen positions are stored. No preprocessing, training, generation, or playback work begins until then.



\---



\# 1. Primary workflow



The complete user workflow must be:



```text

Launch control panel

&#x20;   ↓

Configure collection session

&#x20;   ↓

Play target-click training game

&#x20;   ↓

Inspect and clean recorded trials

&#x20;   ↓

Create immutable dataset snapshot

&#x20;   ↓

Train one or more model types

&#x20;   ↓

Track training metrics and checkpoints

&#x20;   ↓

Run held-out validation

&#x20;   ↓

Compare models

&#x20;   ↓

Promote a selected model

&#x20;   ↓

Export deployable model package

&#x20;   ↓

Load model in generator runtime

&#x20;   ↓

Generate a trajectory for:

&#x20;   current position → circular target

&#x20;   with or without click

&#x20;   ↓

Preview, return trajectory, or explicitly play it

```



Every trained model must remain tied to:



\* Exact dataset snapshot

\* Preprocessing configuration

\* Model configuration

\* Source-code revision

\* Random seed

\* Environment information

\* Validation report



Never train directly against a mutable live dataset.



\---



\# 2. Platform and technology stack



\## Main platform



Initial production target:



```text

Windows 10 and Windows 11

64-bit

Single or multiple monitors

Per-monitor DPI awareness

```



Design interfaces so capture and playback backends can later be replaced for Linux or macOS.



\## Languages



Use:



```text

Python 3.12+

&#x20;   GUI backend

&#x20;   data management

&#x20;   preprocessing

&#x20;   model training

&#x20;   Python inference

&#x20;   reports



QML / Qt Quick

&#x20;   central control-panel interface

&#x20;   collection game

&#x20;   trajectory visualization

&#x20;   experiment and model-management screens



C++20

&#x20;   Windows Raw Input capture

&#x20;   high-resolution timestamps

&#x20;   optional playback scheduler

&#x20;   portable ONNX generator runtime

```



\## Main libraries



Use:



```text

PySide6

PyTorch

Meta flow\_matching library or a small internal compatible implementation

NumPy

SciPy

pandas

PyArrow

scikit-learn

ONNX

ONNX Runtime

Pydantic

PyYAML

SQLite

CMake

pybind11

pytest

GoogleTest

```



Optional developer-only libraries:



```text

TensorBoard

matplotlib

ruff

mypy

pre-commit

```



Do not require a web server, Docker, Redis, or an external tracking service for normal desktop operation.



\---



\# 3. Process architecture



Separate the application into four primary processes or libraries.



```text

┌──────────────────────────────────────────────┐

│ Control Panel                               │

│ PySide6 + QML                               │

│                                              │

│ Dataset, jobs, experiments, validation,      │

│ model registry, generator preview            │

└───────────────────┬──────────────────────────┘

&#x20;                   │ QProcess + JSONL events

&#x20;                   ▼

┌──────────────────────────────────────────────┐

│ Worker Process                              │

│ Python                                      │

│                                              │

│ preprocessing, training, validation, export  │

└───────────────────┬──────────────────────────┘

&#x20;                   │ files + SQLite

&#x20;                   ▼

┌──────────────────────────────────────────────┐

│ Project Store                               │

│                                              │

│ SQLite metadata                             │

│ Parquet samples                             │

│ checkpoints                                 │

│ reports                                     │

│ model artifacts                             │

└──────────────────────────────────────────────┘



┌──────────────────────────────────────────────┐

│ Native Mouse I/O                            │

│ C++ DLL / pybind11 module                   │

│                                              │

│ Raw Input capture                           │

│ timestamps                                  │

│ cursor queries                              │

│ optional explicitly armed playback          │

└──────────────────────────────────────────────┘



┌──────────────────────────────────────────────┐

│ Generator Runtime                           │

│ Python and C++ implementations              │

│                                              │

│ ONNX inference                              │

│ flow integration                            │

│ spline decoding                             │

│ timed trajectory generation                 │

└──────────────────────────────────────────────┘

```



The GUI must never perform model training or large preprocessing operations on its UI thread.



\---



\# 4. Repository structure



Create this monorepo structure:



```text

mouse-motion-lab/

├─ README.md

├─ LICENSE

├─ pyproject.toml

├─ CMakeLists.txt

├─ requirements/

│  ├─ base.txt

│  ├─ training.txt

│  └─ development.txt

├─ config/

│  ├─ default.yaml

│  ├─ collector\_presets.yaml

│  ├─ preprocessing\_presets.yaml

│  ├─ model\_presets.yaml

│  └─ validation\_presets.yaml

├─ apps/

│  ├─ control\_panel/

│  │  ├─ main.py

│  │  ├─ bootstrap.py

│  │  ├─ controllers/

│  │  ├─ viewmodels/

│  │  ├─ models/

│  │  ├─ services/

│  │  └─ qml/

│  │     ├─ Main.qml

│  │     ├─ pages/

│  │     ├─ components/

│  │     ├─ dialogs/

│  │     └─ theme/

│  ├─ worker/

│  │  ├─ \_\_main\_\_.py

│  │  ├─ protocol.py

│  │  └─ jobs/

│  └─ generator\_cli/

│     └─ \_\_main\_\_.py

├─ mouselearn/

│  ├─ domain/

│  ├─ storage/

│  ├─ collection/

│  ├─ preprocessing/

│  ├─ representation/

│  ├─ models/

│  │  ├─ base.py

│  │  ├─ retrieval.py

│  │  ├─ pca\_gmm.py

│  │  └─ conditional\_flow.py

│  ├─ training/

│  ├─ validation/

│  ├─ registry/

│  ├─ export/

│  └─ runtime/

├─ native/

│  ├─ mouse\_io/

│  │  ├─ include/

│  │  ├─ src/

│  │  ├─ bindings/

│  │  └─ tests/

│  └─ mousegen\_runtime/

│     ├─ include/

│     ├─ src/

│     ├─ cli/

│     └─ tests/

├─ schemas/

│  ├─ trial.schema.json

│  ├─ dataset.schema.json

│  ├─ experiment.schema.json

│  ├─ model\_manifest.schema.json

│  └─ generation.schema.json

├─ tests/

│  ├─ unit/

│  ├─ integration/

│  ├─ parity/

│  └─ fixtures/

├─ tools/

│  ├─ doctor.py

│  ├─ migrate\_database.py

│  ├─ inspect\_dataset.py

│  └─ benchmark\_runtime.py

└─ packaging/

&#x20;  ├─ pyinstaller/

&#x20;  ├─ inno\_setup/

&#x20;  └─ release/

```



\---



\# 5. Application storage



Use the following default location:



```text

%LOCALAPPDATA%\\MouseMotionLab\\

```



Directory layout:



```text

MouseMotionLab/

├─ app.db

├─ settings.yaml

├─ logs/

├─ raw\_sessions/

├─ datasets/

├─ experiments/

├─ models/

├─ exports/

├─ cache/

└─ temp/

```



\## Storage responsibilities



Use SQLite for:



\* Session metadata

\* Trial metadata

\* Dataset versions

\* Job state

\* Experiment metadata

\* Metric summaries

\* Model registry

\* Export history

\* Audit log



Use Parquet for:



\* Raw event sequences

\* Processed sample tensors

\* Per-trial feature tables

\* Validation feature tables



Use ordinary files for:



\* YAML configurations

\* Model checkpoints

\* ONNX models

\* Normalization statistics

\* Reports

\* Plots

\* Environment manifests



Do not store every high-frequency mouse event as an individual SQLite row.



\---



\# 6. Core identifiers



All persistent objects must use UUIDs.



Required IDs:



```text

session\_id

trial\_id

dataset\_id

dataset\_version\_id

job\_id

experiment\_id

checkpoint\_id

model\_id

validation\_run\_id

export\_id

```



Use UTC timestamps internally and display local time in the GUI.



\---



\# 7. Database tables



Implement migrations from the beginning.



Minimum tables:



```text

sessions

trials

dataset\_versions

dataset\_members

jobs

experiments

experiment\_metrics

checkpoints

validation\_runs

validation\_metrics

models

exports

settings

audit\_events

```



\## Job lifecycle



```text

queued

running

cancelling

cancelled

completed

failed

```



\## Experiment lifecycle



```text

draft

queued

training

completed

validation\_pending

validated

promoted

exported

archived

failed

```



\## Model lifecycle



```text

candidate

validated

promoted

deprecated

archived

```



Only one model may be marked as the active default model at a time.



\---



\# 8. Training-game collector



Implement the collection game directly in Qt Quick.



\## Trial state machine



```text

Idle

&#x20;   ↓

InterTrialDelay

&#x20;   ↓

TargetVisible

&#x20;   ↓

Tracking

&#x20;   ↓

ValidClick

&#x20;   ↓

TrialFinalization

&#x20;   ↓

InterTrialDelay

```



Additional exits:



```text

Timeout

Cancelled

WindowFocusLost

CaptureFailure

UserPaused

```



\## Target behavior



Each trial displays one circular target.



Target properties:



```text

center\_x

center\_y

radius

appearance\_timestamp

target\_color

optional border

```



Randomization must be controlled and reproducible.



Sample conditions from balanced distributions rather than unrestricted random screen positions.



Condition axes:



```text

Movement distance

Target radius

Movement angle

Screen region

Fitts-style difficulty band

```



Suggested configurable default bins:



```text

Distance:

80–180 px

180–350 px

350–600 px

600–900 px

900+ px where screen permits



Radius:

8–14 px

14–24 px

24–40 px

40–70 px

70+ px



Direction:

16 angular sectors



Screen region:

center

left

right

top

bottom

corners

```



Prevent targets from extending beyond the selected monitor or collection canvas.



Use a random inter-trial delay, for example 400–1,200 ms, so the user does not learn a fixed rhythm.



Do not encourage maximum speed through aggressive scores. Show collection progress, accuracy, and coverage instead.



\## Click behavior



Record every click after target appearance.



A trial ends only after the first valid click inside the target.



If the user misclicks:



\* Keep the misclick event.

\* Keep subsequent corrections.

\* Continue recording until a valid click.

\* Mark the trial as containing one or more misclicks.

\* Allow preprocessing rules to include or exclude such trials.



\## Collection modes



Implement:



```text

Standard

Balanced coverage

Repeated-condition study

Manual target configuration

Validation-only session

```



Repeated-condition mode should show similar conditions multiple times to measure natural variation for the same input condition.



\---



\# 9. Native event capture



Implement a native Windows capture module.



\## Record both domains



For every raw input message, store:



```text

high-resolution timestamp

raw relative dx

raw relative dy

raw button flags

screen cursor x

screen cursor y

active monitor

foreground collection-window state

```



Use:



```text

RegisterRawInputDevices

WM\_INPUT

GetRawInputData

QueryPerformanceCounter

GetCursorPos

```



Register only while a collection session is active.



\## Event record



Use a packed native event structure similar to:



```cpp

struct MouseEvent {

&#x20;   std::uint64\_t timestamp\_ticks;

&#x20;   std::int32\_t raw\_dx;

&#x20;   std::int32\_t raw\_dy;

&#x20;   std::int32\_t screen\_x;

&#x20;   std::int32\_t screen\_y;

&#x20;   std::uint16\_t button\_flags;

&#x20;   std::uint16\_t event\_flags;

&#x20;   std::uint32\_t device\_id;

};

```



Convert native performance-counter ticks to nanoseconds during serialization while retaining original tick frequency in session metadata.



\## Session environment



Store:



```text

virtual desktop bounds

monitor bounds

monitor DPI

Qt device-pixel ratio

Windows pointer-speed setting when available

enhance-pointer-precision state when available

mouse device identifier

raw-input device information

application version

native module version

```



Never silently mix incompatible coordinate systems.



\## Buffering



Use a lock-free or low-contention ring buffer in the native module.



The Python/QML side should consume events in batches rather than crossing the Python boundary for each raw event.



Requirements:



```text

No event allocation in the hot message handler where practical

No disk writes in the input message callback

Batch draining from Python

Dropped-event counter

Overflow warning

```



\---



\# 10. Trial raw-data schema



Each trial must contain:



```text

trial\_id

session\_id

trial\_index



target\_appearance\_ns

valid\_click\_ns

trial\_end\_ns



start\_cursor\_x

start\_cursor\_y



target\_center\_x

target\_center\_y

target\_radius



screen\_bounds

monitor\_id

dpi\_scale



all timestamped movement events

all mouse-button events



valid\_click\_position

misclick\_count

focus\_lost

timed\_out

capture\_overflow

manually\_excluded

quality\_flags

```



Store the event sequence in a Parquet row group or referenced Parquet fragment.



\---



\# 11. Dataset management



Raw sessions are never modified after finalization.



Dataset creation must produce an immutable dataset snapshot.



\## Dataset builder



Allow the user to select:



\* Sessions

\* Date ranges

\* Collection modes

\* Valid or misclicked trials

\* Target-size ranges

\* Distance ranges

\* Quality filters

\* User-defined tags



A dataset version must store:



```text

dataset\_version\_id

name

description

creation timestamp

ordered trial IDs

raw-session hashes

preprocessing configuration

split configuration

feature schema version

code revision

```



\## Splits



Do not randomly split individual neighboring trials by default.



Support:



```text

Session-held-out split

Day-held-out split

Condition interpolation holdout

Condition extrapolation holdout

Repeated-condition holdout

Random split for diagnostics only

```



Default:



```text

70% sessions train

15% sessions validation

15% sessions test

```



Where insufficient sessions exist, warn the user rather than pretending the split is independent.



\---



\# 12. Preprocessing pipeline



Create deterministic preprocessing stages.



```text

Load trial

&#x20;   ↓

Validate timestamps and event order

&#x20;   ↓

Remove duplicate zero-time events

&#x20;   ↓

Build screen-space position sequence

&#x20;   ↓

Detect movement onset

&#x20;   ↓

Separate reaction and movement phases

&#x20;   ↓

Canonicalize geometry

&#x20;   ↓

Fit spatial B-spline

&#x20;   ↓

Fit monotonic timing representation

&#x20;   ↓

Extract conditions and statistics

&#x20;   ↓

Standardize model vectors

&#x20;   ↓

Write processed dataset

```



Each stage must be independently testable and cacheable.



\## Movement onset



Do not define onset as the first nonzero mouse event.



Implement a configurable detector based on:



\* Cumulative displacement threshold

\* Sustained velocity threshold

\* Minimum consecutive duration



Default conceptual behavior:



```text

Ignore tiny initial sensor noise.

Declare onset when displacement and sustained motion exceed thresholds.

```



Store:



```text

reaction\_time

movement\_onset\_timestamp

movement\_duration

target\_entry\_timestamp

entry\_to\_click\_duration

```



\---



\# 13. Canonical coordinate transform



For each trial:



```text

s = movement start

c = target center

D = distance from s to c

θ = angle from s to c

```



Transform a screen position `p` to canonical coordinates:



```text

p\_canonical = rotate(p - s, -θ) / D

```



After transformation:



```text

start = (0, 0)

target center = (1, 0)

target radius = original\_radius / D

```



Keep the original absolute distance, radius, angle, screen location, and monitor-edge information in the condition vector.



The transform must have exact forward and inverse implementations shared by preprocessing and runtime.



\---



\# 14. Spatial trajectory representation



Use a clamped cubic B-spline.



Initial global configuration:



```text

degree = 3

control\_point\_count = 16

fixed start control = true

separate endpoint representation = true

```



Do not vary the control-point count between individual samples in one model version.



For control index `i`, define a linear reference between start and endpoint:



```text

reference\_i = alpha\_i × endpoint

alpha\_i = i / (K - 1)

```



Store intermediate control points as residuals:



```text

control\_residual\_i = actual\_control\_i - reference\_i

```



The model generates only the intermediate residual controls.



The final endpoint is generated separately.



\## Fitting requirements



Fit the spline by regularized least squares.



Expose:



```text

control-point count

smoothing coefficient

endpoint weighting

velocity weighting

maximum reconstruction error

```



Measure reconstruction error in original pixels after inverse transformation.



The preprocessing report must show:



```text

median reconstruction error

95th percentile reconstruction error

maximum reconstruction error

error by movement duration

error by correction count

```



If the selected control-point count loses meaningful corrections, the GUI must recommend rebuilding the dataset with more controls.



Do not apply arbitrary post-generation Gaussian noise.



\---



\# 15. Timing representation



Spatial geometry and physical timing must be represented separately.



Use:



```text

x(t) = spatial\_spline(g(t / T))

```



Where:



```text

T = total movement duration

g = monotonic time-to-spline-progress mapping

```



Use a fixed number of timing intervals:



```text

timing\_interval\_count = 12

```



Represent each interval with an unconstrained logit.



Decode:



```text

positive\_i = softplus(logit\_i) + epsilon

normalized\_i = positive\_i / sum(positive)

progress\_knots = cumulative\_sum(normalized)

```



This guarantees monotonic forward traversal of the spline parameter while still allowing the spatial spline itself to contain overshoots and reversals.



Generate and store separately:



```text

log reaction time

log movement duration

log entry-to-click duration

```



Use configurable lower and upper clamps only as runtime safety bounds, not as replacements for learning.



\---



\# 16. Endpoint representation



Represent the final click or stopping position relative to the target:



```text

q = (endpoint - target\_center) / target\_radius

```



Therefore a valid endpoint satisfies:



```text

norm(q) < 1

```



Use an unconstrained two-dimensional latent endpoint representation `z`.



Decode it into the unit disk:



```text

radius = tanh(norm(z))

q = radius × z / max(norm(z), epsilon)

```



During preprocessing, perform the inverse transform with a clipped radius:



```text

z = atanh(norm(q)) × q / max(norm(q), epsilon)

```



This makes generated endpoints valid by construction.



For no-click movement, endpoint means the requested stopping point inside the target.



\---



\# 17. Model condition vector



The conditional generator receives:



```text

log movement distance

log target radius

log effective difficulty

target radius / movement distance



sin(direction)

cos(direction)



normalized starting screen x

normalized starting screen y

normalized target screen x

normalized target screen y



distance to left screen edge

distance to right screen edge

distance to top screen edge

distance to bottom screen edge



previous cursor velocity x

previous cursor velocity y



click requested flag

monitor DPI scale

optional session/style embedding

```



Derived effective difficulty can use:



```text

log2(distance / target\_width + 1)

target\_width = 2 × radius

```



All continuous features must be normalized using statistics stored with the dataset.



Preserve both canonicalized geometry and screen-context features. This allows general movement sharing while retaining real directional or edge-dependent behavior.



\---



\# 18. Generated model vector



The initial CFM model output vector should contain:



```text

14 intermediate 2D spline-control residuals = 28 values

12 timing logits                           = 12 values

endpoint latent                            = 2 values

log reaction time                          = 1 value

log movement duration                      = 1 value

log click dwell                            = 1 value

\------------------------------------------------------

Total                                      = 45 values

```



If click is disabled, the click-dwell field may be generated but ignored.



Every component must be standardized before flow training.



Store separate mean and scale values per semantic group.



\---



\# 19. Common generator interface



All models must implement the same interface.



```python

class MovementGenerator(Protocol):

&#x20;   def fit(

&#x20;       self,

&#x20;       dataset: ProcessedDataset,

&#x20;       config: ModelConfig,

&#x20;       callbacks: TrainingCallbacks,

&#x20;   ) -> TrainingResult:

&#x20;       ...



&#x20;   def generate(

&#x20;       self,

&#x20;       conditions: ConditionBatch,

&#x20;       seeds: SeedBatch,

&#x20;   ) -> GeneratedParameterBatch:

&#x20;       ...



&#x20;   def save(self, destination: Path) -> None:

&#x20;       ...



&#x20;   @classmethod

&#x20;   def load(cls, source: Path) -> "MovementGenerator":

&#x20;       ...

```



Implement three generators:



```text

Nearest-neighbor retrieval baseline

PCA + conditional Gaussian-mixture baseline

Conditional flow-matching model

```



The GUI must compare them without assuming that the neural model is automatically superior.



\---



\# 20. Retrieval baseline



Build this first.



Algorithm:



```text

Normalize requested conditions

Find nearest recorded conditions

Sample one of the nearest trials

Optionally blend only highly compatible neighbors

Transform its canonical spline parameters to the request

Return its timing and endpoint parameters

```



Configurable values:



```text

neighbor count

distance metric weights

temperature

maximum neighbor distance

whether blending is allowed

```



If no close training condition exists, return an explicit out-of-distribution warning.



This is the minimum-quality baseline for every later model.



\---



\# 21. PCA + conditional mixture baseline



Pipeline:



```text

Standardized trajectory vectors

&#x20;   ↓

PCA

&#x20;   ↓

Low-dimensional latent vectors

&#x20;   ↓

Conditional mixture-density network or condition-binned GMM

&#x20;   ↓

Sample latent

&#x20;   ↓

Inverse PCA

```



Configurable:



```text

PCA retained variance

latent dimension

mixture component count

condition encoder size

covariance type

```



This provides a data-efficient learned baseline.



\---



\# 22. Conditional flow-matching model



Learn:



```text

Gaussian source distribution

&#x20;   ↓

Condition-dependent velocity field

&#x20;   ↓

Distribution of valid trajectory parameter vectors

```



The network represents:



```text

vθ(xτ, τ, c)

```



Where:



```text

xτ = current state in trajectory-parameter space

τ = generative flow time

c = target and context conditions

```



\## Initial probability path



Use the basic linear path first:



```text

x0 \~ Normal(0, I)

x1 = real standardized trajectory vector

τ \~ Uniform(0, 1)



xτ = (1 - τ)x0 + τx1

target velocity = x1 - x0

```



Loss:



```text

MSE(predicted\_velocity, target\_velocity)

```



Do not add optimal-transport coupling in the first implementation.



\## Initial network



Use:



```text

Condition encoder:

&#x20;   3-layer MLP

&#x20;   hidden width 128

&#x20;   output width 128



Flow-time embedding:

&#x20;   Fourier or sinusoidal features

&#x20;   width 64



Velocity network:

&#x20;   residual MLP

&#x20;   6 residual blocks

&#x20;   hidden width 256

&#x20;   SiLU

&#x20;   RMSNorm or LayerNorm

&#x20;   FiLM condition injection in every block



Output:

&#x20;   same dimensionality as trajectory vector

```



Make all dimensions configurable.



\## Training defaults



Use configurable defaults approximately like:



```text

optimizer: AdamW

learning rate: 0.0003

weight decay: 0.00001

batch size: 256 or 512

gradient clipping: 1.0

EMA enabled: true

mixed precision: true when supported

validation interval: every epoch

checkpoint interval: configurable

early stopping: optional

```



Do not hard-code GPU requirements. Support CPU training for small tests and CUDA when available.



\## Numerical solver



Use Heun integration by default.



Initial inference step count:



```text

16

```



Expose:



```text

Euler

Midpoint

Heun

RK4

```



Benchmark 4, 8, 16, 32, and 64 steps during validation.



The flow solver generates one complete parameter vector. It is unrelated to physical mouse-event sampling frequency.



\---



\# 23. Training orchestration



Training must run in a separate worker process.



Launch with:



```text

python -m apps.worker train --experiment-id <uuid>

```



The worker emits line-delimited JSON events through stdout.



Example:



```json

{

&#x20; "type": "metric",

&#x20; "job\_id": "...",

&#x20; "step": 1400,

&#x20; "epoch": 12,

&#x20; "name": "validation\_loss",

&#x20; "value": 0.0412,

&#x20; "timestamp": "..."

}

```



Required worker messages:



```text

started

stage\_changed

progress

metric

checkpoint\_saved

warning

log

completed

cancelled

failed

```



The control panel must use `QProcess`, parse these messages, and update SQLite.



The worker must also write its authoritative state to disk so an application restart can recover completed or failed jobs.



\## Training artifacts



Each experiment directory contains:



```text

config.yaml

dataset\_manifest.json

environment.json

git\_revision.txt

stdout.log

stderr.log

metrics.parquet

checkpoints/

previews/

final/

```



\---



\# 24. Training dashboard



During training, display:



```text

Current stage

Epoch

Step

Elapsed duration

Estimated dataset progress

Training loss

Validation loss

Learning rate

Gradient norm

Samples per second

CPU usage

GPU usage when available

GPU memory

Latest checkpoint

Warnings

```



The user must be able to:



```text

Pause collection jobs where supported

Cancel training

Open logs

Open experiment directory

Compare current run with another run

Resume from checkpoint

```



Do not claim that a process has paused unless the worker has acknowledged it.



\---



\# 25. Validation framework



Validation must evaluate both correctness and distributional similarity.



\## Hard correctness checks



Require:



```text

Start point exactly matches requested start

Endpoint is inside target

Timestamps are monotonic

Movement duration is positive

Reaction time is nonnegative

No NaN or infinity

No point leaves configured virtual desktop unless explicitly permitted

Spline decoding succeeds

Timing mapping is monotonic

Click occurs only when requested

```



\## Distribution metrics



Compare real and generated held-out movements for:



```text

Reaction-time distribution

Movement-time distribution

Entry-to-click distribution

Endpoint distribution within target

Path length / direct distance

Maximum perpendicular deviation

Signed average lateral deviation

Peak velocity

Time of peak velocity

Peak acceleration

Velocity-profile shape

Number of velocity peaks

Overshoot frequency

Direction-reversal count

Target entry count

Target re-entry count

Misclick-related behavior where enabled

```



Use appropriate statistics:



```text

Wasserstein distance

Kolmogorov–Smirnov statistic

Energy distance

Maximum mean discrepancy

Mean and quantile differences

Calibration error across condition bins

```



A real-versus-generated classifier may be used as one diagnostic statistic, but must be labeled as a research distribution-similarity metric, not as a mechanism for bypassing external detection.



\## Condition response tests



For fixed random seeds, vary:



```text

Distance

Radius

Difficulty

Direction

Screen position

Previous velocity

```



Verify that the generator responds smoothly and plausibly.



\## Holdout classes



Validation UI must separate:



```text

Known-condition held-out sessions

Interpolation conditions

Sparse conditions

Extrapolation conditions

Out-of-distribution requests

```



Never combine all cases into one score without exposing the breakdown.



\---



\# 26. Model scorecard



Generate a model scorecard with:



```text

Hard validity rate

In-distribution quality score

Interpolation score

Extrapolation warning score

Timing similarity

Geometry similarity

Endpoint similarity

Diversity score

Mode-coverage score

Inference latency

Export parity

```



A promoted model must pass configurable gates.



Initial gates should include:



```text

100% finite output

100% valid monotonic timing

100% endpoint validity

No critical runtime errors

Export parity within tolerance

Validation report completed

```



Distribution thresholds should initially be informational until enough data exists to select meaningful limits.



\---



\# 27. Model registry



Each model registry entry contains:



```text

model\_id

display name

model type

experiment ID

dataset version ID

creation timestamp

status

validation run ID

validation summary

checkpoint location

ONNX export location

runtime compatibility version

notes

tags

active-default flag

```



Registry actions:



```text

Validate

Promote

Set active

Export

Clone configuration

Compare

Deprecate

Archive

Delete local artifacts with confirmation

```



Promotion must never overwrite the source experiment.



\---



\# 28. Model export package



Export a self-contained directory:



```text

MouseModel\_<name>\_<version>/

├─ model.onnx

├─ manifest.json

├─ condition\_normalization.json

├─ output\_normalization.json

├─ spline\_spec.json

├─ runtime\_config.json

├─ validation\_summary.json

├─ LICENSES.txt

└─ README.txt

```



\## Manifest contents



```text

format version

model ID

model type

dataset version

model input names and shapes

model output names and shapes

condition feature order

output feature order

normalization values

spline degree

spline knot vector

control count

timing interval count

default solver

default solver steps

supported runtime versions

model hash

artifact hashes

```



No runtime may infer feature ordering from source-code assumptions. It must read the manifest.



\---



\# 29. ONNX export strategy



Export only the learned velocity network to ONNX.



Keep these operations in the runtime:



```text

Gaussian sampling

ODE integration

output denormalization

endpoint disk transform

timing softplus and normalization

B-spline evaluation

canonical-to-screen transform

physical timestamp creation

```



This keeps solver choice and output rate configurable without re-exporting the network.



Verify PyTorch-versus-ONNX parity with:



```text

Fixed conditions

Fixed source-noise vectors

Fixed flow times

Velocity-output comparison

Full generated parameter comparison

Decoded-trajectory comparison

```



Store maximum and mean parity errors in the export report.



\---



\# 30. Generator request API



Define:



```cpp

struct GenerationRequest {

&#x20;   double start\_x;

&#x20;   double start\_y;



&#x20;   double target\_center\_x;

&#x20;   double target\_center\_y;

&#x20;   double target\_radius;



&#x20;   double previous\_velocity\_x;

&#x20;   double previous\_velocity\_y;



&#x20;   double virtual\_desktop\_left;

&#x20;   double virtual\_desktop\_top;

&#x20;   double virtual\_desktop\_width;

&#x20;   double virtual\_desktop\_height;



&#x20;   double dpi\_scale;



&#x20;   bool include\_reaction\_delay;

&#x20;   bool click\_requested;



&#x20;   std::uint64\_t random\_seed;

&#x20;   std::uint32\_t output\_rate\_hz;

&#x20;   std::uint32\_t solver\_steps;

};

```



Define output:



```cpp

struct TrajectorySample {

&#x20;   std::int64\_t relative\_time\_ns;

&#x20;   double x;

&#x20;   double y;

};



struct GenerationResult {

&#x20;   std::vector<TrajectorySample> samples;



&#x20;   std::int64\_t reaction\_delay\_ns;

&#x20;   std::int64\_t movement\_duration\_ns;

&#x20;   std::int64\_t click\_delay\_ns;



&#x20;   double endpoint\_x;

&#x20;   double endpoint\_y;



&#x20;   bool click\_requested;

&#x20;   bool out\_of\_distribution;

&#x20;   double condition\_distance\_score;



&#x20;   std::uint64\_t seed;

};

```



The default API returns a trajectory. It does not automatically inject mouse input.



\---



\# 31. Generation pipeline



Runtime generation:



```text

Validate request

&#x20;   ↓

Derive distance and direction

&#x20;   ↓

Build condition vector

&#x20;   ↓

Normalize condition

&#x20;   ↓

Generate Gaussian source vector from seed

&#x20;   ↓

Integrate conditional flow using ONNX velocity network

&#x20;   ↓

Denormalize generated parameter vector

&#x20;   ↓

Decode endpoint into target disk

&#x20;   ↓

Build spatial B-spline controls

&#x20;   ↓

Build monotonic timing mapping

&#x20;   ↓

Decode durations

&#x20;   ↓

Evaluate at requested physical output rate

&#x20;   ↓

Rotate, scale, and translate into screen coordinates

&#x20;   ↓

Apply final numeric corrections

&#x20;   ↓

Return timed samples and optional click timing

```



\## Exact boundary rules



Force:



```text

First trajectory sample = requested start position

Final trajectory sample = generated valid endpoint

Final endpoint remains inside circle with numeric safety margin

```



Do not modify intermediate points with an arbitrary endpoint correction after generation unless validation detects numeric drift.



\---



\# 32. Spline evaluation



The generator must evaluate:



```text

position(t) = spline(progress(t))

```



At:



```text

t = 0

t = 1 / output\_rate

t = 2 / output\_rate

...

t = movement duration

```



Default output rate:



```text

250 Hz

```



Allow:



```text

125 Hz

250 Hz

500 Hz

1000 Hz

custom

```



Use double precision for spline and timing evaluation in the C++ runtime.



The output is allowed to contain repeated integer pixel coordinates after rounding. Preserve floating-point coordinates in the generated result and round only at the playback boundary.



\---



\# 33. Optional playback harness



Playback is an optional testing layer.



It must be disabled by default.



Playback controls:



```text

Arm playback

Disarm

Preview only

Start

Abort

Emergency stop hotkey

```



Safety rules:



```text

Require visible armed state

Require explicit user action for each playback

Never start at application launch

Never run hidden

Never block physical user input

Abort when substantial physical mouse movement is detected

Abort when Escape is pressed

Abort when target application loses allowed focus, if configured

Do not capture generated playback as training data

```



Implement a fixed internal marker and active-playback state so generated samples are excluded from collection.



Use a dedicated high-priority scheduling thread but never a real-time kernel driver.



The first release should only enable playback inside the application’s own test canvas. External desktop playback can remain behind an advanced explicit setting.



\---



\# 34. Central-control-panel GUI



Use a persistent left sidebar.



Pages:



```text

Dashboard

Collect

Sessions

Datasets

Train

Experiments

Validate

Models

Generator

Diagnostics

Settings

```



\## Dashboard



Show:



```text

Total sessions

Total usable trials

Total recorded movement time

Condition-space coverage

Latest dataset version

Running jobs

Latest completed experiment

Active model

Latest validation status

Recent warnings

```



Primary actions:



```text

Start collection

Build dataset

Train model

Validate candidate

Open generator

```



\## Collect page



Sections:



```text

Session preset

Monitor/canvas selection

Target distribution

Trial count or duration

Coverage preview

Mouse-device status

Capture diagnostics

Start/pause/stop

Live session statistics

```



When collection begins, switch to a distraction-free full-window target game.



\## Sessions page



Provide:



```text

Session table

Filters

Trial count

Usable count

Collection date

Device

Monitor

Tags

Quality warnings

Open session

Exclude session

```



Session detail:



```text

Timeline

Condition distribution

Trajectory overlays

Reaction-time histogram

Movement-time summary

Misclick list

Capture gaps

Individual trial viewer

```



\## Datasets page



Provide:



```text

Dataset versions

Included sessions

Filter rules

Split summary

Condition-space heatmap

Spline reconstruction report

Preprocessing warnings

Create new immutable version

Clone settings

```



\## Train page



Provide presets:



```text

Retrieval baseline

PCA mixture

Conditional flow — small

Conditional flow — standard

Conditional flow — custom

```



Configuration sections:



```text

Dataset version

Model architecture

Training settings

Condition features

Spline representation

Timing representation

Hardware

Output directory

Random seed

```



Show estimated sample count and warnings before starting.



\## Experiments page



Table columns:



```text

Name

Model type

Dataset

Status

Started

Duration

Best validation loss

Validity

Inference latency

Tags

```



Experiment detail:



```text

Configuration

Live metrics

Checkpoints

Logs

Generated previews

Validation history

Artifacts

Environment

```



\## Validate page



Allow selection of:



```text

Model

Dataset test split

Validation preset

Generated samples per condition

Solver

Solver steps

Random seeds

```



Display:



```text

Validity summary

Metric scorecards

Real/generated overlays

Timing distributions

Endpoint distributions

Velocity profiles

Condition-bin breakdown

Out-of-distribution analysis

Failure examples

```



\## Models page



Implement the registry actions described earlier.



Clearly distinguish:



```text

Candidate

Validated

Active

Deprecated

```



\## Generator page



Provide an interactive canvas.



The user can:



```text

Set start point

Set target center

Adjust target radius

Choose click/no-click

Set seed or randomize

Select model

Set solver steps

Set output rate

Generate

Generate multiple variants

Preview movement

Arm local test playback

Export trajectory

```



Visualization must show:



```text

Start

Target circle

Generated endpoint

Spline control polygon toggle

Trajectory

Velocity encoded along timeline

Timing cursor during preview

```



Show generated metadata:



```text

Reaction delay

Movement duration

Click delay

Path length

Peak speed

Condition-distance warning

Random seed

Inference time

```



\## Diagnostics page



Show:



```text

Raw Input status

Capture event rate

Dropped events

Detected mouse devices

Monitor layout

DPI values

Python environment

PyTorch availability

CUDA availability

ONNX Runtime providers

Native DLL versions

Database status

Storage usage

Recent errors

```



Provide a “Run system check” action.



\---



\# 35. GUI architecture



Use MVVM-style separation.



QML must not directly access SQLite, PyTorch, ONNX, or native handles.



Expose backend objects such as:



```text

AppController

NavigationController

DashboardViewModel

CollectorViewModel

SessionsViewModel

DatasetViewModel

TrainingViewModel

ExperimentsViewModel

ValidationViewModel

ModelRegistryViewModel

GeneratorViewModel

DiagnosticsViewModel

SettingsViewModel

```



Use `QAbstractListModel` for tables and lists.



Use explicit signals for:



```text

job progress

session statistics

model changes

validation completion

generator preview updates

errors

```



QML JavaScript should remain limited to presentation logic.



\---



\# 36. Visualization components



Build reusable QML components:



```text

TrajectoryCanvas

TargetCircle

MetricCard

StatusBadge

ProgressPanel

ConditionCoverageView

DistributionPlot

VelocityProfilePlot

EndpointPlot

ExperimentMetricPlot

LogViewer

WarningBanner

ModelSelector

SeedControl

```



For initial implementation, custom trajectory plotting may use QML Canvas or a dedicated painted item.



The trajectory viewer must support:



```text

Zoom

Pan

Real/generated overlay

Normalized/screen coordinates

Velocity markers

Click markers

Target-entry markers

Spline control points

Playback animation

```



\---



\# 37. Configuration system



Use versioned Pydantic models loaded from YAML.



Top-level configuration:



```yaml

schema\_version: 1



application:

&#x20; data\_root: auto

&#x20; log\_level: INFO



collector:

&#x20; monitor: primary

&#x20; trials: 500

&#x20; timeout\_ms: 5000

&#x20; inter\_trial\_delay\_ms: \[400, 1200]



preprocessing:

&#x20; spatial\_control\_points: 16

&#x20; spline\_degree: 3

&#x20; timing\_intervals: 12

&#x20; onset\_detector: default



training:

&#x20; model\_type: conditional\_flow

&#x20; batch\_size: 256

&#x20; learning\_rate: 0.0003

&#x20; seed: 42



generation:

&#x20; solver: heun

&#x20; solver\_steps: 16

&#x20; output\_rate\_hz: 250



playback:

&#x20; enabled: false

&#x20; external\_desktop\_enabled: false

&#x20; emergency\_stop\_key: Escape

```



Store the fully resolved configuration with every experiment.



Unknown configuration fields must produce warnings or errors instead of being silently ignored.



\---



\# 38. Command-line interfaces



Implement:



```text

mouselearn doctor

mouselearn collect

mouselearn dataset build

mouselearn preprocess

mouselearn train

mouselearn validate

mouselearn export

mouselearn generate

mouselearn benchmark

```



Example generation command:



```text

mouselearn generate \\

&#x20; --model <export-directory> \\

&#x20; --start 400,300 \\

&#x20; --target 1200,700 \\

&#x20; --radius 30 \\

&#x20; --click \\

&#x20; --seed 1234 \\

&#x20; --output trajectory.json

```



Default CLI behavior returns data only and does not inject input.



\---



\# 39. Generated trajectory formats



Support JSON:



```json

{

&#x20; "schema\_version": 1,

&#x20; "model\_id": "...",

&#x20; "seed": 1234,

&#x20; "request": {},

&#x20; "reaction\_delay\_ns": 182000000,

&#x20; "movement\_duration\_ns": 476000000,

&#x20; "click\_delay\_ns": 43000000,

&#x20; "samples": \[

&#x20;   {

&#x20;     "relative\_time\_ns": 0,

&#x20;     "x": 400.0,

&#x20;     "y": 300.0

&#x20;   }

&#x20; ]

}

```



Also support CSV:



```text

relative\_time\_ns,x,y,event

```



Possible events:



```text

move

button\_down

button\_up

```



\---



\# 40. Out-of-distribution detection



Build a condition-distance estimator.



Initial implementation:



```text

Normalize requested condition

Find nearest training conditions

Compute weighted nearest-neighbor distance

Compare against validation-derived percentiles

```



Return:



```text

in\_distribution

sparse

out\_of\_distribution

```



Display the result in the Generator page.



Do not prevent generation automatically unless hard geometric limits are violated. Warn the user and log the request.



\---



\# 41. Logging and auditability



Use structured logs.



Every important action records:



```text

timestamp

severity

component

event type

relevant object ID

message

structured context

```



Audit:



```text

Session created

Dataset built

Training started

Training cancelled

Model validated

Model promoted

Model exported

Playback armed

Playback started

Playback aborted

Settings changed

```



Do not record unrelated user desktop activity.



\---



\# 42. Testing strategy



\## Unit tests



Test:



```text

Coordinate transforms

Endpoint disk transform and inverse

Spline basis evaluation

Spline fitting

Timing monotonicity

Condition normalization

Output denormalization

Flow training interpolation

Euler and Heun solvers

Dataset splitting

Manifest parsing

Configuration migrations

```



\## Property tests



Verify:



```text

Canonical transform followed by inverse recovers point

Generated endpoint always lies in target

Timing progress is always monotonic

First and last trajectory samples are exact

Same request and same seed are deterministic

Different seeds generally produce different samples

No generated value is nonfinite

```



\## Native tests



Test:



```text

Raw Input registration lifecycle

Ring-buffer overflow behavior

Timestamp conversion

Monitor coordinate conversion

Playback cancellation

Emergency stop

```



\## Integration tests



Test:



```text

Synthetic session → dataset

Dataset → retrieval model

Dataset → small flow model

Model → validation

PyTorch → ONNX export

ONNX → Python runtime

ONNX → C++ runtime

```



\## Parity tests



For fixed test fixtures, compare:



```text

Python PyTorch velocity network

Python ONNX Runtime velocity network

C++ ONNX Runtime velocity network

Python spline decoder

C++ spline decoder

```



Set explicit tolerances.



\## GUI tests



Test major view models without rendering QML.



Add a small set of Qt integration tests for:



```text

Navigation

Collection start/stop

Training job launch

Job cancellation

Model promotion

Generator preview

```



\---



\# 43. Synthetic test-data generator



Create a deterministic synthetic dataset generator for development.



It should create trajectories with selectable modes:



```text

Direct

Left curve

Right curve

Overshoot

One correction

Two corrections

Slow target settling

```



This permits complete pipeline testing before enough human data exists.



Synthetic data must always be labeled and must never be mixed into a real-person production dataset without an explicit setting.



\---



\# 44. Performance targets



Initial desktop targets:



```text

GUI remains responsive during collection and training

No normal capture-buffer overflow at 1000 Hz input

Trajectory generation below 30 ms on a typical desktop CPU

250 Hz trajectory evaluation below 2 ms after parameter generation

Model load below 500 ms after warm filesystem cache

Deterministic seeded generation

```



Treat these as engineering targets and report measured results instead of assuming success.



\---



\# 45. Packaging



\## Development build



Provide scripts:



```text

setup-dev.ps1

build-native.ps1

run-control-panel.ps1

run-tests.ps1

```



\## Release build



Use:



```text

CMake for native DLLs

PyInstaller onedir for the control panel

Inno Setup for Windows installer

```



Prefer `onedir` initially because PySide6, ONNX Runtime, and native DLL packaging are easier to diagnose than a single-file package.



Installer components:



```text

Control panel

Native capture module

Python ONNX runtime

Optional C++ runtime SDK

Example model package

Documentation

```



The installer must not automatically enable playback or create a startup task.



\---



\# 46. C++ runtime API



Expose a small C ABI around the C++ runtime for compatibility.



Example:



```cpp

typedef void\* MGHandle;



MGResult mg\_create(

&#x20;   const char\* model\_directory,

&#x20;   MGHandle\* out\_handle

);



MGResult mg\_generate(

&#x20;   MGHandle handle,

&#x20;   const MGGenerationRequest\* request,

&#x20;   MGTrajectory\* out\_trajectory

);



void mg\_free\_trajectory(

&#x20;   MGTrajectory\* trajectory

);



void mg\_destroy(

&#x20;   MGHandle handle

);



const char\* mg\_last\_error(

&#x20;   MGHandle handle

);

```



Also provide a modern C++ wrapper.



The runtime library must not depend on Qt.



\---



\# 47. Failure handling



No worker or native error should crash the entire control panel without a readable report.



Handle:



```text

Corrupt database

Missing model file

Manifest mismatch

Unsupported model format

ONNX load failure

CUDA unavailable

Capture device removal

Monitor layout change

Insufficient dataset

Training NaN

Disk full

Cancelled job

Invalid target radius

Start equals target center

Target outside virtual desktop

```



For `start == target center`, generate only endpoint settling or an optional click delay rather than dividing by zero.



\---



\# 48. Implementation milestones



Implement in this order.



\## Milestone 1 — Foundation



Deliver:



```text

Repository

Configuration system

SQLite migrations

Logging

Basic PySide6/QML shell

Navigation

Diagnostics page

Worker JSONL protocol

```



Acceptance:



```text

Application launches

Database initializes

Worker test job reports progress

GUI remains responsive

```



\## Milestone 2 — Collection



Deliver:



```text

Native Raw Input module

Target game

Session storage

Trial viewer

Collection diagnostics

```



Acceptance:



```text

500-trial session records without data loss

Target appearance and clicks have high-resolution timestamps

Raw deltas and screen positions are both stored

```



\## Milestone 3 — Dataset and representation



Deliver:



```text

Dataset snapshots

Session-held-out splits

Canonical transform

Spline fit

Timing fit

Reconstruction report

Dataset GUI

```



Acceptance:



```text

Dataset build is deterministic

Processed data can be reconstructed

Forward/inverse transforms pass parity tests

```



\## Milestone 4 — Baselines



Deliver:



```text

Retrieval model

PCA mixture model

Common generator interface

Generator preview page

Baseline validation

```



Acceptance:



```text

Model generates valid trajectories for unseen in-range conditions

Same seed reproduces same trajectory

```



\## Milestone 5 — Conditional flow



Deliver:



```text

CFM model

Training worker

Checkpointing

Experiment tracking

Training dashboard

Flow inference

```



Acceptance:



```text

Small synthetic dataset can be overfit

Multimodal synthetic modes remain represented

Real dataset training completes

```



\## Milestone 6 — Validation and registry



Deliver:



```text

Validation framework

Metric reports

Model comparisons

Registry

Promotion gates

Out-of-distribution warnings

```



Acceptance:



```text

Candidate can be validated and promoted

Reports remain tied to exact model and dataset

```



\## Milestone 7 — Export and portable runtime



Deliver:



```text

ONNX export

Python ONNX runtime

C++ ONNX runtime

Manifest

CLI

Parity tests

```



Acceptance:



```text

PyTorch, Python ONNX, and C++ ONNX outputs match within tolerance

Export directory loads independently of the training project

```



\## Milestone 8 — Playback test harness



Deliver:



```text

Local application-canvas playback

High-resolution scheduler

Arming state

Abort behavior

Emergency stop

Generated-input exclusion

```



Acceptance:



```text

Playback never starts unarmed

Escape stops playback

Physical movement aborts playback

Generated events are not collected as training data

```



\## Milestone 9 — Packaging



Deliver:



```text

PyInstaller build

Installer

Release verification

Clean-machine test

Documentation

```



\---



\# 49. Codex implementation rules



Follow these rules throughout development:



1\. Implement one milestone at a time.

2\. Do not create placeholder buttons that silently do nothing.

3\. Keep business logic outside QML.

4\. Keep training outside the GUI process.

5\. Keep the C++ generator independent from Qt and Python.

6\. Use typed models at all process and file boundaries.

7\. Version every persistent schema.

8\. Write migrations before changing database structures.

9\. Make every model reproducible from its stored manifest.

10\. Never use the latest mutable dataset directly for training.

11\. Never silently drop capture events.

12\. Never silently replace an unavailable CUDA device with another backend; report the fallback.

13\. Never promote a model without a completed validation report.

14\. Never inject mouse input unless playback is visibly armed.

15\. Never add stealth, evasion, anti-cheat, or hidden automation behavior.

16\. Add unit tests with every mathematical component.

17\. Add Python/C++ parity fixtures before implementing the release runtime.

18\. Prefer a small working vertical slice over broad unfinished scaffolding.

19\. Keep sample synthetic data clearly separated from real recordings.

20\. Update README architecture and setup instructions at the end of every milestone.



\---


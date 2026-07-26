# EDM diffusion vs. flow toy

This additive experiment compares the existing conditional flow design with an
EDM-style continuous diffusion model on the rank-one target
`c * [-2, -1, 0, 1, 2]`, where `c` spans `[-2, 2]`. It does not register EDM as
a production model.

The diffusion formulation follows [Elucidating the Design Space of
Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) and the
[official NVIDIA EDM implementation](https://github.com/NVlabs/edm).

Both arms use identical normalized train/validation splits, constant 21-value
zero conditions, 192-by-3 SiLU MLPs, AdamW settings, checkpoint rules, and
generated-sample seeds. EDM uses its standard preconditioning and weighted
log-normal-noise loss, a Karras schedule, and an 18-step deterministic Heun
sampler with 35 network evaluations. Flow uses 16 Heun steps and 32 evaluations.

Run the selected seven-size, three-seed CPU sweep from the repository root:

```powershell
.\.venv\Scripts\python.exe .\toys\diffusion_vs_flow\run.py --sizes 67,128,256,512,1000,2000,4000 --seeds 42,43,44 --samples 2048 --output-dir build/toy-diffusion-vs-flow
```

The output directory contains `runs.json`, `summary.csv`,
`generated-vectors.csv`, and `comparison.md`. Strict best-within-250 and
validation-driven converged checkpoints are reported independently; generated
quality metrics never influence training or checkpoint selection.

For a fixed-epoch follow-up on selected arms, use `--algorithms` and
`--fixed-epochs`. For example, `--sizes 128 --algorithms flow
--fixed-epochs 7500` retains the best first-250 checkpoint and also reports the
best checkpoint across all 7,500 epochs.

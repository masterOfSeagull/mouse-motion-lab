# Flow-correlation toy

This experiment isolates whether MouseMotionLab's conditional-flow objective can learn correlation among coordinates of one output vector.

Each training target is

```text
c * [-2, -1, 0, 1, 2]
```

with `c` covering `[-2, 2]`. The experiment compares a constant 21-value condition against 21 independent Gaussian garbage values that have no relationship to `c`. A correct joint model should generate a distribution along that one-dimensional line while ignoring the nuisance inputs. A marginal or mean-collapsed model would instead lose the sign relationships or converge near the all-zero vector.

Training, validation, and test data are separate. The best validation checkpoint is evaluated using unseen test conditions and an unseen test `c` distribution. Input-to-output correlations and linear nuisance-input R-squared are reported for the garbage-condition cases.

Run it from the repository root:

```powershell
.\.venv\Scripts\python.exe .\toys\flow_correlation\run.py
```

The summary report is written to `build/toy-flow-correlation/results.json`, and every generated vector, projected scalar, and test input is written to `build/toy-flow-correlation/generated-vectors-garbage-input.csv`. Key metrics include the fraction of variance explained by the target line, relative projection error, correlation-sign error, scalar quantiles, empirical Wasserstein and KS distances, tail mass, nuisance-input dependence, and the predicted scalar standard deviation relative to the test distribution.

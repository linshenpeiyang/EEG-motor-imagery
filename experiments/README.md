# Model comparison

The experiment asks whether learned spatial features improve cross-subject
imagery classification over the two spectral baselines used during development.
It preserves the original pilot comparison as part of the project's evidence.

```bash
python -m eegmem prepare
python -m eegmem evaluate
```

The shared evaluator in `eegmem/pipeline.py` runs all four methods on the same
450 trials. Each fold holds out one fixed subject pair: 1–2, 3–4, 5–6, 7–8 or
9–10. The other eight subjects provide training data. Every method uses a
training-fitted scaler and logistic regression.

| Method | Features |
| --- | --- |
| C3/C4 | Two run-standardized log mean spectral-power features |
| All channels | 64 run-standardized log mean spectral-power features |
| CSP | Six spatial log-variance features |
| FBCSP | Two CSP features from each of five bands; eight selected by mutual information |

CSP filters and FBCSP feature selection are fitted inside each training fold.
Held-out labels are used only for scoring. FBCSP uses fourth-order zero-phase
Butterworth filters at 8–12, 12–16, 16–20, 20–24 and 24–30 Hz. All methods receive
the same prepared epochs; the spatial methods do not use the spectral models'
run-wise feature normalization.

## Reproduction details

`spatial.py` preserves the original custom CSP calculation: average centered
trial cross-products, an absolute diagonal ridge of `1e-6`, and a `1e-8` offset
before taking log variance. These scale-dependent constants are retained to
make the historical comparison reproducible. They are not a tuned or exhaustive
CSP benchmark. FBCSP now fixes the mutual-information seed to zero, so the
feature selection can be repeated. The earlier script did not set this seed.

Exact fold scores and method settings are stored in
[results/metrics.json](../results/metrics.json). Evaluation creates no persistent
spatial model and leaves the assistant's model, reports and memory unchanged.

## Interpretation

The comparison supports a model choice within this ten-subject pilot. It does
not establish that CSP or FBCSP generally fails across subjects, nor isolate
which preprocessing or regularization choice caused a score difference.
Subject-specific calibration, relative covariance regularization and other
spatial implementations remain untested here. The five folds were used during
method selection, so an additional unseen cohort is needed for confirmation.

The deployed classifier remains all-channel spectral power with logistic
regression. The memory layer retains C3/C4 features regardless of the chosen
classifier. These two responsibilities allow model experiments without changing
the interpretation of stored reference features.

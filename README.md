# EEG State Memory Assistant

A memory-augmented assistant for EEG state recording and analysis. Every newly
arriving batch of signals goes through four stages: analyze, compare against a
persistent memory bank, write a report with meaning and suggestions, then grow
the memory by one batch.

## Why this project exists

Two problems shaped the design.

Language agents forget a conversation once the session ends. The standard fix
is an external log the agent reads when a new session starts. This project
applies the same idea to EEG analysis. The classifier gets an external,
persistent memory bank.

A single batch of data produces one isolated result. Statistically that is
n=1, which is not evidence. The memory bank accumulates every detection, so
the sample size grows with use and the conclusions get stronger.

## Architecture

```
new batch -> feature extraction -> state recognition -> memory comparison -> report -> memory +1
```

| Module | File | Job |
| --- | --- | --- |
| Memory bank | eegmem/memory.py | SQLite persistence, queries, per-class statistics |
| Batch stream | eegmem/batch.py | slice trials into arrival-ordered batches |
| Analysis | eegmem/analyze.py | bandpower features, frozen logistic regression prior |
| Comparison | eegmem/compare.py | z-score reference ranges, k-NN, anti-echo gate |
| Report | eegmem/report.py | Markdown report with meaning, suggestions, uncertainty |
| Assistant | eegmem/app.py | one-command loop with idempotent memory |

## Quick start

Python 3.12 recommended. Install dependencies with `pip install -r requirements.txt`.

```bash
# single-subject pipeline
python scripts/01_load.py
python scripts/03_epochs.py
python scripts/04_bandpower.py

# memory assistant
python -m eegmem.analyze
python -m eegmem.app --reset
python -m eegmem.app stats

# multi-subject upgrade, downloads about 74 MB from PhysioNet
python scripts/10_download_subjects.py
python scripts/11_subject_batches.py
python scripts/12_run_subjects.py
python scripts/13_run_subjects_csp.py
python scripts/14_compare_cross_subject.py
python scripts/15_memory_curve.py
```

## Core design

The classifier and the memory bank use different features. The classifier
needs discriminative features. The memory bank needs comparable features. CSP
shows why: its spatial patterns are the strongest within one subject, and they
do not transfer across subjects. The assistant uses all-channel bandpower for
recognition and keeps C3/C4 bandpower in memory for reference ranges and
neighbor comparison.

The prior model is frozen after calibration. The memory bank grows with use.
This is the textbook versus clinical experience split. Retraining the prior on
memory data would be rewriting the textbook at every patient visit.

An anti-echo gate disables neighbor voting when one class occupies 70% or more
of memory. Without the gate a biased classifier would be amplified into false
consensus.

Uncertainty is reported honestly. Confidence is not reliability. Low
confidence and insufficient samples are stated explicitly with re-check
advice. The report supports clinical judgment and does not replace it.

## Experimental findings

These results come from running the pipeline, not from tuning stories.

A hand-written rule that picks the side with lower energy was systematically
left-biased at 39 of 45 trials. It confused a subject-specific channel offset
with desynchronization.

A prior model calibrated on 20 trials reached 0.80 cross-validation accuracy
and 0.56 at deployment. Its bias flipped direction. Small-sample priors are
unstable.

Mixing six subjects without normalization polluted the memory reference
ranges. 27% of trials were flagged as anomalies. Session-wise z-scoring cut
this to 7% while classification accuracy stayed unchanged. The transform is
invisible to the classifier and critical for the memory.

CSP reached 64% to 78% within one subject and about 50% across subjects.
FBCSP scored 0.504 ± 0.080 in leave-two-out cross validation. Spatial
patterns are individual.

The deployed model is all-channel bandpower plus logistic regression. It
scored 0.627 ± 0.032 across five subject-wise folds and 64% on a held-out
deployment of six subjects.

![Memory growth](figures/memory_growth.png)

The left panel shows the cumulative accuracy estimate converging as the
memory grows. The right panel shows the anomaly rate staying near the
expected false-alarm level. The memory makes estimates stable. It does not
make the model magical.

## Limitations and next steps

The pilot covers 10 of 109 PhysioNet subjects. Cross-subject accuracy around
63% is near the plateau for subject-independent simple models. Reaching 70%
or more realistically requires per-subject calibration or transfer methods
such as Riemannian alignment. The reference-range gate can be upgraded to
confidence intervals and drift detection.

## Ethics

This is a research prototype. Its output is auxiliary reference material, not
medical diagnosis. Clinical decisions belong to qualified personnel using
complete clinical context. Data comes from the public PhysioNet EEG Motor
Movement/Imagery Dataset.

## Related research

Continual learning, catastrophic forgetting, memory-augmented agents, and
retrieval-augmented generation. This project is a memory layer for EEG
analysis that never forgets and grows with data.

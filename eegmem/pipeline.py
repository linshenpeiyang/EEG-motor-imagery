"""One reproducible data preparation and evaluation protocol."""
from pathlib import Path
import json

import numpy as np

from eegmem.analyze import batch_features, make_model
from eegmem.compare import compare_batch
from eegmem.memory import create_db, load_all, save_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECTS = range(1, 11)
RUNS = [4, 8, 12]
TRAIN_SUBJECTS = range(1, 5)
TEST_SUBJECTS = range(5, 11)


def subject_paths(root, subjects):
    paths = [Path(root) / 'data' / 'subjects' / f'S{s:03d}R{r:02d}.npz'
             for s in subjects for r in RUNS]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise ValueError(f"Missing batches: {', '.join(missing)}. Run prepare first.")
    return paths


def prepare(root):
    """Download cached EDF runs and save one batch per complete run."""
    import mne
    from mne.datasets import eegbci

    directory = Path(root) / 'data' / 'subjects'
    directory.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        for run in RUNS:
            source = eegbci.load_data(subject, runs=[run],
                                      path=Path(root) / 'data' / 'mne_data',
                                      update_path=False, verbose=False)[0]
            raw = mne.io.read_raw_edf(source, preload=True, verbose=False)
            eegbci.standardize(raw)
            raw.filter(7, 30, verbose=False)
            events, event_id = mne.events_from_annotations(
                raw, event_id={'T1': 2, 'T2': 3}, verbose=False)
            epochs = mne.Epochs(raw, events, event_id, -1, 4, baseline=(None, 0),
                                reject=None, preload=True, verbose=False)
            path = directory / f'S{subject:03d}R{run:02d}.npz'
            np.savez_compressed(path, data=epochs.get_data(),
                                labels=epochs.events[:, -1] - 2,
                                ch_names=np.array(epochs.ch_names),
                                sfreq=epochs.info['sfreq'], subject=subject, run=run)
            print(f'{path.name}: {len(epochs)} trials')


def evaluate(root):
    """Evaluate in temporary memory without changing the assistant's model or records."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cache = {}
    channels = sfreq = None
    for path in subject_paths(root, SUBJECTS):
        x, y, channels, sfreq = batch_features(path, channels, sfreq)
        if y is None:
            raise ValueError('Evaluation requires known labels')
        cache[path.stem] = (x, y)
    picks = [channels.index(c) for c in ['C3', 'C4']]

    def dataset(subjects):
        batches = [cache[p.stem] for p in subject_paths(root, subjects)]
        return np.vstack([x for x, _ in batches]), np.concatenate([y for _, y in batches])

    folds = {'C3/C4': [], 'All channels': []}
    for first in range(1, 11, 2):
        held = [first, first + 1]
        train = [s for s in SUBJECTS if s not in held]
        x_train, y_train = dataset(train)
        x_test, y_test = dataset(held)
        for name, indices in [('C3/C4', picks), ('All channels', slice(None))]:
            model = make_model().fit(x_train[:, indices], y_train)
            score = model.score(x_test[:, indices], y_test)
            folds[name].append(float(score))
        print(f'Evaluated subjects {held}', flush=True)

    x_train, y_train = dataset(TRAIN_SUBJECTS)
    model = make_model().fit(x_train, y_train)
    conn = create_db(':memory:')
    counts, accuracy, flags, per_subject = [], [], [], {}
    total = correct = flagged = 0
    try:
        for path in subject_paths(root, TEST_SUBJECTS):
            x, y = cache[path.stem]
            pred, prob = model.predict(x), model.predict_proba(x).max(axis=1)
            trials = [{'trial': i, 'c3': float(x[i, picks[0]]), 'c4': float(x[i, picks[1]]),
                       'pred': int(pred[i]), 'prob': float(prob[i]), 'label': int(y[i])}
                      for i in range(len(y))]
            results, evidence = compare_batch(load_all(conn), trials)
            save_batch(conn, path.stem, path.stem, results, evidence)
            hits = int((pred == y).sum())
            n_flags = sum(r['verdict'] == 'Outside reference' for r in results)
            total += len(y)
            correct += hits
            flagged += n_flags
            counts.append(total)
            accuracy.append(correct / total)
            flags.append(n_flags / len(y))
            subject = path.stem[:4]
            old = per_subject.setdefault(subject, {'correct': 0, 'trials': 0})
            old['correct'] += hits
            old['trials'] += len(y)
    finally:
        conn.close()
    metrics = {
        'protocol': 'Five fixed subject-pair folds; complete-run normalization without labels',
        'feature_window_seconds': [-1, 4],
        'fold_accuracy': folds,
        'deployment': {'training_subjects': list(TRAIN_SUBJECTS),
                       'test_subjects': list(TEST_SUBJECTS), 'correct': correct,
                       'trials': total, 'accuracy': correct / total,
                       'reference_flags': flagged, 'per_subject': per_subject},
        'curve': {'trials': counts, 'cumulative_accuracy': accuracy, 'batch_flag_rate': flags},
    }
    output = Path(root) / 'results'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(counts, accuracy, 'o-', color='#2a6f97')
    axes[0].axhline(0.5, color='gray', linestyle='--', label='Balanced chance reference')
    axes[0].set(xlabel='Recorded trials', ylabel='Cumulative accuracy', ylim=(0.3, 1),
                title='Frozen model: accumulating evaluation evidence')
    axes[0].legend(fontsize=8)
    axes[1].plot(range(1, len(flags) + 1), flags, 's-', color='#b6465f')
    axes[1].set(xlabel='Incoming run', ylabel='Fraction outside reference', ylim=(0, 0.6),
                title='Historical reference flags per batch')
    fig.tight_layout()
    images = Path(root) / 'figures'
    images.mkdir(parents=True, exist_ok=True)
    fig.savefig(images / 'memory_growth.png', dpi=150)
    plt.close(fig)
    for name, scores in folds.items():
        print(f'{name}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}')
    print(f'Deployment: {correct}/{total}; reference flags: {flagged}')
    return metrics

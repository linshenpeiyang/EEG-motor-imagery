"""Validated batch features and a frozen left/right imagery classifier."""
from pathlib import Path

import numpy as np
from mne.time_frequency import psd_array_multitaper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_VERSION = "run-zscore-log-mean-psd-8-30-v1"


def load_batch(path):
    """Read volts in trial/channel/time order; labels are optional."""
    with np.load(path, allow_pickle=False) as z:
        data = np.asarray(z["data"], dtype=float)
        channels = z["ch_names"].tolist()
        sfreq = float(z["sfreq"])
        labels = np.asarray(z["labels"]) if "labels" in z else None
    if data.ndim != 3 or data.shape[0] < 2 or data.shape[2] < 2:
        raise ValueError("A batch needs at least two trials in trial/channel/time order")
    if len(channels) != data.shape[1] or len(set(channels)) != len(channels):
        raise ValueError("Channel names must be unique and match the data")
    if not {"C3", "C4"}.issubset(channels):
        raise ValueError("C3 and C4 are required for memory comparison")
    if not np.isfinite(data).all() or not np.isfinite(sfreq) or sfreq <= 60:
        raise ValueError("Signals must be finite and sampling frequency must exceed 60 Hz")
    if labels is not None:
        if labels.shape != (len(data),) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Labels must contain one 0 or 1 per trial")
        labels = labels.astype(int)
    return data, channels, sfreq, labels


def session_zscore(features):
    """Normalize each channel across the complete incoming run, without labels."""
    std = features.std(axis=0)
    return (features - features.mean(axis=0)) / np.where(std > 1e-12, std, 1.0)


def batch_features(path, expected_channels=None, expected_sfreq=None):
    data, channels, sfreq, labels = load_batch(path)
    if expected_channels is not None and channels != expected_channels:
        raise ValueError(f"{Path(path).name}: channel order differs from the training data")
    if expected_sfreq is not None and sfreq != expected_sfreq:
        raise ValueError(f"{Path(path).name}: sampling frequency differs from training")
    psd, _ = psd_array_multitaper(data, sfreq=sfreq, fmin=8, fmax=30,
                                normalization="length", verbose=False)
    features = session_zscore(np.log(np.maximum(psd.mean(axis=-1), np.finfo(float).tiny)))
    return features, labels, channels, sfreq


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))


def train_model(paths):
    features, labels = [], []
    channels = sfreq = None
    for path in paths:
        x, y, channels, sfreq = batch_features(path, channels, sfreq)
        if y is None:
            raise ValueError("Training batches require known labels")
        features.append(x)
        labels.append(y)
    if not features:
        raise ValueError("No training batches found; run prepare first")
    x, y = np.vstack(features), np.concatenate(labels)
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both left and right imagery")
    return {"model": make_model().fit(x, y), "channels": channels, "sfreq": sfreq,
            "feature_version": FEATURE_VERSION, "training_batches": [p.stem for p in paths],
            "training_trials": len(y)}


def analyze_batch(path, artifact):
    if artifact["feature_version"] != FEATURE_VERSION:
        raise ValueError("Feature definition changed; retrain and use a new memory database")
    x, labels, channels, _ = batch_features(path, artifact["channels"], artifact["sfreq"])
    model = artifact["model"]
    predictions, probabilities = model.predict(x), model.predict_proba(x).max(axis=1)
    c3, c4 = channels.index("C3"), channels.index("C4")
    return [{"trial": i, "c3": float(x[i, c3]), "c4": float(x[i, c4]),
             "pred": int(predictions[i]), "prob": float(probabilities[i]),
             "label": None if labels is None else int(labels[i])}
            for i in range(len(x))]

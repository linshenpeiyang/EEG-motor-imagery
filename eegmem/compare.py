"""Compare a batch with earlier predicted records, before inserting any new trial."""
import numpy as np

from eegmem.memory import class_stats


def compare_batch(rows, trials):
    stats = [class_stats(rows, label) for label in [0, 1]]
    points = np.array([[r["c3"], r["c4"]] for r in rows])
    balanced = len(rows) >= 5 and max(s["n"] for s in stats) / len(rows) < 0.7
    results = []
    for trial in trials:
        s = stats[trial["pred"]]
        values = [trial["c3"], trial["c4"]]
        z = [(v - m) / d if d is not None and d > 1e-12 else None
             for v, m, d in zip(values, s["mean"], s["std"])]
        if s["n"] == 0:
            verdict = "Cold start"
        elif s["n"] < 5 or any(value is None for value in z):
            verdict = "Insufficient reference"
        elif max(abs(value) for value in z) > 2:
            verdict = "Outside reference"
        else:
            verdict = "Within reference"
        agreement = None
        if balanced:
            indices = np.argsort(((points - values) ** 2).sum(axis=1), kind="stable")[:3]
            agreement = sum(rows[int(i)]["pred"] == trial["pred"] for i in indices) >= 2
        results.append({**trial, "z": z, "verdict": verdict,
                        "low_confidence": trial["prob"] < 0.6,
                        "neighbor_agreement": bool(agreement) if agreement is not None else None})
    return results, stats

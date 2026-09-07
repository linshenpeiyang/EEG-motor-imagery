"""Compact, reproducible Markdown reports from stored results."""
from pathlib import Path

LABELS = ["Left-hand imagery", "Right-hand imagery"]


def render_report(name, results, evidence, context=None):
    lines = [f"# EEG batch {name}", "", "Research output only. No clinical diagnosis.", "",
             "Features are run-standardized C3/C4 log mean spectral power.",
             "Table z-scores compare features with historical predicted-class records. Model probabilities are uncalibrated.", "",
             "| Trial | Prediction | Probability | C3 z | C4 z | Reference | Notes |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in results:
        z = ["n/a" if v is None else f"{v:.2f}" for v in r["z"]]
        notes = []
        if r["low_confidence"]:
            notes.append("Low model confidence; review prediction")
        if r["verdict"] == "Outside reference":
            notes.append("Review signal quality and repeat recording if needed")
        if r["neighbor_agreement"] is False:
            notes.append("Historical predictions disagree")
        elif r["neighbor_agreement"] is None:
            notes.append("Neighbor vote unavailable: small or imbalanced memory")
        lines.append(f"| {r['trial']} | {LABELS[r['pred']]} | {r['prob']:.2f} | "
                     f"{z[0]} | {z[1]} | {r['verdict']} | {'; '.join(notes) or 'No additional flag'} |")
    labeled = [r for r in results if r["label"] is not None]
    lines.extend(["", "## Batch summary", "", f"Trials: {len(results)}."])
    if labeled:
        correct = sum(r["label"] == r["pred"] for r in labeled)
        lines.append(f"Known-label matches: {correct}/{len(labeled)}. This is a descriptive batch result.")
    else:
        lines.append("No known labels; accuracy cannot be calculated.")
    lines.extend(["", "## Memory before this batch", ""])
    for label, s in zip(LABELS, evidence):
        lines.append(f"- {label}: {s['n']} records.")
        if s["n"]:
            for channel, mean, std in zip(["C3", "C4"], s["mean"], s["std"]):
                spread = "unavailable" if std is None else f"{std:.3f}"
                lines.append(f"  {channel} mean {mean:.3f}, sample standard deviation {spread}.")
    lines.extend(["", "Historical agreement is not independent verification. A reference flag does not identify its cause.", ""])
    if context is not None:
        lines.extend(["## Reference provenance", "",
                      f"Policy: {context['policy']}. Review event cutoff: {context['review_event_cutoff']}.",
                      "Source IDs and review versions are stored with this batch in analysis_context.",
                      f"Reference trials: {len(context['sources'])}. Later reviews do not rewrite this report.", ""])
    return "\n".join(lines)


def write_report(directory, name, results, evidence, context=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(render_report(name, results, evidence, context), encoding="utf-8")
    temporary.replace(path)
    return path

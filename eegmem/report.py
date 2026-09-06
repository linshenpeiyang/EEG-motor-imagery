"""第 5 步：反馈报告（Report）——把分析变成"意义 + 建议 + 诚实的不确定性"。

助手不是自动诊断机：它把信号判成什么状态、和记忆库比较如何、
对应的建议，写成一份人可读的 MD 报告留档（和我们的对话记录同一思路）。
三条设计原则：
  1. 结论永远附证据（置信度、z 分数、记忆库统计），不只给一个标签；
  2. 拿不准就明说拿不准，并给复核建议——置信度不等于可靠性；
  3. 人在回路：报告支持临床判断，不代替临床判断。
"""
from datetime import datetime
from pathlib import Path

import joblib

from eegmem.analyze import CALIBRATION_BATCHES, MODEL_PATH
from eegmem.compare import BATCH_DIR, run_batch
from eegmem.memory import class_stats, clear_db, create_db

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"

STATE_LABELS = {0: "Left-hand imagery", 1: "Right-hand imagery"}

# 每种比较结论对应一类建议（规则模板，不是玄学）
SUGGESTIONS = {
    "Cold start": "No history for this state yet. This call rests on the prior "
                  "model alone. Hold off on decisions until similar samples accumulate.",
    "Insufficient samples": "Fewer than five records for this state. The reference "
                            "range is not stable yet. Treat this as a preliminary impression.",
    "Anomaly": "Outside the historical reference range. Check electrode impedance "
               "and subject state. Re-test if needed.",
    "Low confidence": "The C3/C4 gap is small. Treat this as uncertain and do not "
                      "act on it alone.",
    "Normal": "Consistent with historical records of the same state.",
}


def suggest(r):
    """根据比较结论拼出针对该试次的建议。"""
    parts = [SUGGESTIONS.get(r["verdict"], "")]
    if r["nbr_agree"] is False:
        parts.append("Neighbor vote disagrees. The call may be wrong. Re-check.")
    if r["note"] and "imbalance" in r["note"]:
        parts.append("Memory class imbalance. Treat the comparison as a hint only.")
    return " ".join(parts)


def render_report(batch_name, results):
    """Render one batch into a Markdown report."""
    lines = [
        f"# EEG State Analysis Report: {batch_name}",
        "",
        f"> Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"> Trials in this batch: {len(results)}",
        "> This report supports clinical judgment. It does not replace it.",
        "",
    ]
    for r in results:
        z3s = f"{r['z3']:.1f}" if r["z3"] is not None else "n/a"
        z4s = f"{r['z4']:.1f}" if r["z4"] is not None else "n/a"
        lines += [
            f"## Trial {r['trial']}: {STATE_LABELS[r['pred']]}"
            f" (confidence {r['prob']:.2f})",
            "",
            f"- Features: C3={r['c3']:.3f}, C4={r['c4']:.3f}",
            f"- Memory comparison: z3={z3s}, z4={z4s}; verdict: {r['verdict']}",
            f"- Suggestion: {suggest(r)}",
            "",
        ]
    lines.append("---")
    lines.append("")
    lines.append("### Batch summary")
    lines.append("")
    correct = sum(x["pred"] == x["label"] for x in results)
    lines.append(f"- Matches with known labels: {correct}/{len(results)}."
                 f" For teaching only; {len(results)} is too few trials for "
                 f"the ratio to be meaningful.")
    lines.append("- When the verdict is Anomaly or Low confidence, check signal "
                 "quality and subject state before trusting the call.")
    lines.append("")
    return "\n".join(lines)


def render_memory_evidence(conn):
    """List the memory bank's statistics so the operator can judge for themselves."""
    s0, s1 = class_stats(conn, 0), class_stats(conn, 1)

    def fmt_val(mean, std):
        if mean is None:
            return "no records"
        if std is None:
            return f"{mean:.3f} (too few samples for std)"
        return f"{mean:.3f}±{std:.3f}"

    lines = ["## Memory evidence (before this batch was written)", ""]
    for name, s in [("Left-hand imagery", s0), ("Right-hand imagery", s1)]:
        lines.append(f"- {name}: n={s['n']}, "
                     f"C3 mean={fmt_val(s['c3_mean'], s['c3_std'])}, "
                     f"C4 mean={fmt_val(s['c4_mean'], s['c4_std'])}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit("还没有先验模型：请先运行 analyze.py 完成标定")
    model = joblib.load(MODEL_PATH)
    conn = create_db()
    clear_db(conn)

    REPORT_DIR.mkdir(exist_ok=True)
    files = [p for p in sorted(BATCH_DIR.glob("batch_*.npz"))
             if p.stem not in CALIBRATION_BATCHES]
    for path in files[:-1]:          # 前几批只积累记忆
        run_batch(conn, path, model)

    current = files[-1]              # 最后一批：比较 + 出报告
    evidence = render_memory_evidence(conn)   # 注意：这是"写入前"的证据
    results = run_batch(conn, current, model)
    report = render_report(current.stem, results) + "\n" + evidence

    out_path = REPORT_DIR / f"{current.stem}_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"报告已写入 {out_path}\n")
    print("=" * 60)
    print(report)

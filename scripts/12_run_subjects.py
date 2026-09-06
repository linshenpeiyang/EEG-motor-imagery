"""升级包第三步：跨受试者跑同一个助手（含"会话内标准化"修复）。

标定：受试者 1~4（12 个 run）训练先验模型；
上岗：受试者 5~10（18 个 run）逐批喂给记忆助手。
助手本体（analyze/compare/report 的函数）原样复用，不改一行。

踩坑记录：跨受试者"基线偏移"（batch effect）——每位受试者 C3/C4 的
绝对能量水平不同，原始数据直接混合时，记忆库参考范围失真
（270 条里 74 条被误标"异常"）。修复：analyze.session_zscore 做
"会话内标准化"，去掉绝对水平差异、保留相对抑制信息。
注意：标准化不改变分类精度（线性模型管道本就含 StandardScaler），
它保护的是记忆库的参考范围——记忆层和分类器对数据的要求不同。
"""
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegmem.analyze import (MODEL_PATH, extract_features, session_zscore,
                            train_model)
from eegmem.compare import run_batch
from eegmem.memory import class_stats, clear_db, create_db
from eegmem.report import REPORT_DIR, render_memory_evidence, render_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_DIR = PROJECT_ROOT / "data" / "subjects"
CALIB_SUBJECTS = [1, 2, 3, 4]
DEPLOY_SUBJECTS = [5, 6, 7, 8, 9, 10]


def subject_paths(subjects):
    """按受试者编号挑出批次文件。"""
    return sorted(p for p in SUBJECT_DIR.glob("S*.npz")
                  if int(p.stem[1:4]) in subjects)


def load_feat_labels(paths):
    """读取若干批次，拼出 (特征, 标签)。"""
    X, y = [], []
    for p in paths:
        with np.load(p) as z:
            feat = session_zscore(
                extract_features(z["data"], z["ch_names"], float(z["sfreq"]),
                                 channels=None))
            X.append(feat)
            y.append(z["labels"])
    return np.vstack(X), np.concatenate(y)


if __name__ == "__main__":
    cal_paths = subject_paths(CALIB_SUBJECTS)
    dep_paths = subject_paths(DEPLOY_SUBJECTS)

    # 1) 标定先验模型（受试者 1~4，共 12 批 / 180 试次）
    X, y = load_feat_labels(cal_paths)
    model = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"先验模型：在受试者 {CALIB_SUBJECTS}（{len(X)} 试次）上标定并保存")

    # 2) 助手上岗（受试者 5~10，逐批进记忆）
    conn = create_db()
    clear_db(conn)
    REPORT_DIR.mkdir(exist_ok=True)
    total = correct = n_anomaly = 0
    for p in dep_paths:
        evidence = render_memory_evidence(conn)
        results = run_batch(conn, p, model)
        text = render_report(p.stem, results) + "\n" + evidence
        (REPORT_DIR / f"{p.stem}_report.md").write_text(text, encoding="utf-8")
        c = sum(r["pred"] == r["label"] for r in results)
        n_anomaly += sum(r["verdict"] == "Anomaly" for r in results)
        total += len(results)
        correct += c
        print(f"{p.stem}: {c}/{len(results)} 判对")

    print(f"\n上岗合计: {correct}/{total} = {correct/total:.0%}，"
          f"其中被标'Anomaly' {n_anomaly} 条")
    print("记忆库最终:")
    print("  左手想象:", class_stats(conn, 0))
    print("  右手想象:", class_stats(conn, 1))
    conn.close()

"""升级包第五步：CSP 判别 + C3/C4 记忆（分类与记忆的职责分离）。

正是上一课洞察的工程落地——同一批信号喂出两套特征，各干各的：
  分类器要判别性 -> 用 CSP 特征（空间滤波后的 log 方差）判断状态；
  记忆层要可比性 -> 仍存 C3/C4 会话标准化特征做参考范围/邻居比较。
compare / report 原样复用，只替换了"判定"这一小步。
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegmem.analyze import extract_features, session_zscore
from eegmem.compare import compare_trial
from eegmem.csp import CSP
from eegmem.memory import add_record, class_stats, clear_db, create_db
from eegmem.report import REPORT_DIR, render_memory_evidence, render_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_DIR = PROJECT_ROOT / "data" / "subjects"
CSP_MODEL_PATH = PROJECT_ROOT / "models" / "model_csp.joblib"
CALIB_SUBJECTS = [1, 2, 3, 4]
DEPLOY_SUBJECTS = [5, 6, 7, 8, 9, 10]
N_COMPONENTS = 6


def subject_paths(subjects):
    return sorted(p for p in SUBJECT_DIR.glob("S*.npz")
                  if int(p.stem[1:4]) in subjects)


def load_raw(paths):
    """读原始波形与标签（CSP 需要原始多通道数据，不能用 2 维特征）。"""
    X, y = [], []
    for p in paths:
        with np.load(p) as z:
            X.append(z["data"])
            y.append(z["labels"])
    return np.vstack(X), np.concatenate(y)


def make_csp_model():
    return make_pipeline(CSP(n_components=N_COMPONENTS),
                         StandardScaler(),
                         LogisticRegression(max_iter=1000))


if __name__ == "__main__":
    cal, dep = subject_paths(CALIB_SUBJECTS), subject_paths(DEPLOY_SUBJECTS)

    # 1) 标定 CSP 模型，并先看它自己的交叉验证水平
    Xc, yc = load_raw(cal)
    scores = cross_val_score(make_csp_model(), Xc, yc, cv=5)
    print(f"CSP 在标定受试者 {CALIB_SUBJECTS} 上的 5 折 CV: "
          f"{scores.mean():.3f} ± {scores.std():.3f}")
    model = make_csp_model().fit(Xc, yc)
    joblib.dump(model, CSP_MODEL_PATH)
    print(f"模型已保存到 {CSP_MODEL_PATH.name}\n")

    # 2) 上岗：CSP 判状态，C3/C4 管记忆
    conn = create_db()
    clear_db(conn)
    REPORT_DIR.mkdir(exist_ok=True)
    total = correct = n_anomaly = 0
    for p in dep:
        with np.load(p) as z:
            data = z["data"]
            labels = z["labels"]
            feat = session_zscore(extract_features(data, z["ch_names"],
                                                   float(z["sfreq"])))
        pred = model.predict(data)
        prob = model.predict_proba(data).max(axis=1)
        trials = [{"trial": i,
                   "c3": float(feat[i, 0]), "c4": float(feat[i, 1]),
                   "pred": int(pred[i]), "prob": float(prob[i]),
                   "label": int(labels[i])} for i in range(len(data))]

        evidence = render_memory_evidence(conn)     # 写入前的记忆证据
        results = [compare_trial(conn, t) for t in trials]
        for r in results:                           # 先比后写，防自己和自己比
            add_record(conn, r["c3"], r["c4"], r["pred"], r["prob"],
                       batch=p.stem, note=r["note"])
        text = render_report(p.stem, results) + "\n" + evidence
        (REPORT_DIR / f"{p.stem}_csp_report.md").write_text(text, encoding="utf-8")

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

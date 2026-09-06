"""第 4 步：与记忆库比较（Compare）——让历史给当前批次投票。

对当前批次每个试次做三件事：
  1. 找相似：在 (C3, C4) 平面上找记忆库里距离最近的 k 条历史记录；
  2. 算异常：把当前 C3/C4 与【预测状态】的历史分布比较得 z 分数，
     |z| > 2 视为"超出参考范围"（正态分布约 95% 的值落在 ±2 内）；
  3. 邻居投票：最近 k 条记录多数是什么状态，与本次判定是否一致。
然后把这批结果连同备注写回记忆库——比较完，记忆又长大一点。

关键顺序：先查历史、后写入。若先写入再查，新记录会"自己和自己比"。

升级（第 3.5 课）：分类不再用手写规则，改用 analyze.py 标定好的
逻辑回归（先验知识）；邻居投票前先检查记忆类别是否平衡，
防止把分类器的偏差固化成一呼百应的"回声"。
"""
from pathlib import Path

import joblib
import numpy as np

from eegmem.analyze import (
    CALIBRATION_BATCHES,
    MODEL_PATH,
    STATE_NAMES,
    analyze_batch,
)
from eegmem.memory import add_record, class_stats, clear_db, create_db, load_all

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = PROJECT_ROOT / "data" / "batches"
K_NEIGHBORS = 3
Z_THRESHOLD = 2.0


def nearest_neighbors(rows, c3, c4, k=K_NEIGHBORS):
    """返回最近的 k 条 [(距离, 状态)]；记忆为空时返回空列表。"""
    if not rows:
        return []
    pts = np.array([[r[3], r[4]] for r in rows])
    dist = np.sqrt(((pts - np.array([c3, c4])) ** 2).sum(axis=1))
    order = np.argsort(dist)[:k]
    return [(float(dist[i]), rows[i][5]) for i in order]


def z_score(value, mean, std):
    """z = (x - 均值) / 标准差；标准差未知（样本太少）时返回 None。"""
    if std is None or std == 0:
        return None
    return (value - mean) / std


def compare_trial(conn, trial):
    """先查历史分布与邻居，给出冷启动/样本少/异常/低置信/正常 之一。"""
    stats = class_stats(conn, trial["pred"])
    z3 = z_score(trial["c3"], stats["c3_mean"], stats["c3_std"])
    z4 = z_score(trial["c4"], stats["c4_mean"], stats["c4_std"])
    rows = load_all(conn)
    nbrs = nearest_neighbors(rows, trial["c3"], trial["c4"])

    if stats["n"] == 0:
        verdict, note = "Cold start", "No history for this state; record kept for reference"
    elif stats["n"] < 5:
        verdict, note = "Insufficient samples", f"Only {stats['n']} records for this state; reference range not stable yet"
    else:
        zmax = max(abs(z3), abs(z4))
        if zmax > Z_THRESHOLD:
            verdict = "Anomaly"
            note = f"Outside reference range z3={z3:.1f} z4={z4:.1f}"
        elif trial["prob"] < 0.6:
            verdict = "Low confidence"
            note = f"Small C3/C4 gap prob={trial['prob']:.2f}"
        else:
            verdict, note = "Normal", None

    nbr_agree = None
    if nbrs:
        n_left = sum(1 for r in rows if r[5] == 0)
        dominant = max(n_left, len(rows) - n_left) / len(rows)
        if len(rows) >= 4 and dominant >= 0.7:
            # anti-echo: voting would only repeat the dominant class
            note = ((note + "; memory class imbalance, vote suspended")
                    if note else "Memory class imbalance, vote suspended")
        else:
            agree_votes = sum(1 for _, p in nbrs if p == trial["pred"])
            nbr_agree = agree_votes > len(nbrs) / 2
            if not nbr_agree:
                note = (note + "; neighbor vote disagrees") if note else "Neighbor vote disagrees"

    return {**trial, "z3": z3, "z4": z4, "verdict": verdict,
            "note": note, "nbrs": [p for _, p in nbrs],
            "nbr_agree": nbr_agree}


def run_batch(conn, path, model):
    """分析 + 比较 + 写回记忆库，返回本批逐试次结果。"""
    results = [compare_trial(conn, t) for t in analyze_batch(path, model)]
    for r in results:
        add_record(conn, r["c3"], r["c4"], r["pred"], r["prob"],
                   batch=path.stem, note=r["note"])
    return results


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        raise SystemExit("还没有先验模型：请先运行 analyze.py 完成标定")
    model = joblib.load(MODEL_PATH)

    conn = create_db()
    clear_db(conn)   # 演示：清掉第 1 步手动插的演示数据，从冷启动开始
    files = [p for p in sorted(BATCH_DIR.glob("batch_*.npz"))
             if p.stem not in CALIBRATION_BATCHES]

    total = total_correct = 0
    for path in files:
        results = run_batch(conn, path, model)
        print(f"\n== {path.name}（写入前记忆 {len(load_all(conn)) - len(results)} 条）==")
        for r in results:
            z3s = f"z3={r['z3']:.1f}" if r["z3"] is not None else "z3=--"
            z4s = f"z4={r['z4']:.1f}" if r["z4"] is not None else "z4=--"
            agree = ""
            if r["nbr_agree"] is not None:
                agree = "，邻居一致" if r["nbr_agree"] else "，邻居不一致"
            ok = "对" if r["pred"] == r["label"] else "错"
            remark = f" | 备注: {r['note']}" if r["note"] else ""
            print(f"  试次{r['trial']}: 判{STATE_NAMES[r['pred']]} {r['prob']:.2f} | "
                  f"{r['verdict']} | {z3s} {z4s}{agree} | "
                  f"真实{STATE_NAMES[r['label']]}({ok}){remark}")
            total += 1
            total_correct += r["pred"] == r["label"]

    print(f"\n共 {total} 个试次，机制分类器判对 {total_correct} 个 "
          f"（{total_correct / total:.0%}）")
    print("记忆库最终状态:")
    print("  左手想象:", class_stats(conn, 0))
    print("  右手想象:", class_stats(conn, 1))
    conn.close()

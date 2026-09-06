"""升级包第八步：记忆增长可视化——证据随样本量收敛。

一张图回答"记忆模块到底有没有用"：
  左图：累计判对率随试次数的变化——n 越大，估计越稳，向模型的
        真实上岗水平收敛（注意：收敛到真实值，而不是被吹高）；
  右图：逐批异常率——早期参考范围不成熟、判得忽高忽低，
        n 攒够后回到 ~5% 的正常报警率。
"""
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Hiragino Sans GB", "PingFang SC", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eegmem.analyze import MODEL_PATH
from eegmem.compare import run_batch
from eegmem.memory import clear_db, create_db

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBJECT_DIR = PROJECT_ROOT / "data" / "subjects"
DEPLOY_SUBJECTS = [5, 6, 7, 8, 9, 10]
FIG_DIR = PROJECT_ROOT / "figures"


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)
    conn = create_db()
    clear_db(conn)
    paths = sorted(p for p in SUBJECT_DIR.glob("S*.npz")
                   if int(p.stem[1:4]) in DEPLOY_SUBJECTS)

    ns, cum_acc, batch_anom = [], [], []
    total = correct = 0
    for p in paths:
        results = run_batch(conn, p, model)
        correct += sum(r["pred"] == r["label"] for r in results)
        total += len(results)
        ns.append(total)
        cum_acc.append(correct / total)
        n_anom = sum(r["verdict"] == "Anomaly" for r in results)
        batch_anom.append(n_anom / len(results))
        print(f"{p.stem}: 累计 {total} 试次, 累计判对率 {correct/total:.1%}, "
              f"本批异常 {n_anom}/{len(results)}")
    conn.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ns, cum_acc, "o-", color="#2a6f97")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, label="随机水平 50%")
    ax1.set_xlabel("记忆库累计试次数 n")
    ax1.set_ylabel("累计判对率")
    ax1.set_ylim(0.3, 1.0)
    ax1.set_title("证据累积：估计随 n 增大而收敛")
    ax1.legend()

    ax2.plot(range(1, len(batch_anom) + 1), batch_anom, "s-",
             color="#d1495b")
    ax2.axhline(0.05, color="gray", ls="--", lw=1,
                label="正常报警率约 5%")
    ax2.set_xlabel("批次序号（每批 15 试次）")
    ax2.set_ylabel("本批异常率")
    ax2.set_ylim(0, 0.6)
    ax2.set_title("参考范围成熟：异常率向 ~5% 收敛")
    ax2.legend()

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "memory_growth.png"
    fig.savefig(out, dpi=150)
    print(f"图已保存: {out}")
    print(f"最终累计判对率: {correct/total:.1%}（{correct}/{total}）")

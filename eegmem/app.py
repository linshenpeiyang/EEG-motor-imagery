"""第 6 步：助手（App）——一条命令串起全流程。

用法：
  python -m eegmem.app           处理所有尚未入档的新批次
  python -m eegmem.app batch_05  只处理指定批次
  python -m eegmem.app --reset   先清空记忆（回到冷启动）再处理
  python -m eegmem.app stats     只看记忆库现状

关键设计：助手靠记忆库判断"哪些批次已经处理过"，
重复运行不会重复入档——它记得自己做过什么，重启也不会忘。
这正是这个项目立项的初衷：给脑电分析加一层跨会话、不会遗忘的记忆。
"""
import argparse
from pathlib import Path

import joblib

from eegmem.analyze import CALIBRATION_BATCHES, MODEL_PATH
from eegmem.compare import BATCH_DIR, run_batch
from eegmem.memory import class_stats, clear_db, create_db, load_all
from eegmem.report import REPORT_DIR, render_memory_evidence, render_report


def processed_batches(conn):
    """记忆库里已经入过档的批次名集合。"""
    rows = conn.execute("SELECT DISTINCT batch FROM states").fetchall()
    return {r[0] for r in rows}


def process_batch(conn, path, model):
    """处理一个新批次：分析 → 与记忆比较 → 写入记忆 → 出报告。

    已经入过档的批次直接跳过（记忆的幂等性）。
    """
    if path.stem in processed_batches(conn):
        return None
    evidence = render_memory_evidence(conn)      # 写入前的记忆证据
    results = run_batch(conn, path, model)       # 比较并写入
    text = render_report(path.stem, results) + "\n" + evidence
    out = REPORT_DIR / f"{path.stem}_report.md"
    out.write_text(text, encoding="utf-8")
    return out


def new_batches():
    """按顺序返回所有"新信号"批次（标定批次除外）。"""
    return [p for p in sorted(BATCH_DIR.glob("batch_*.npz"))
            if p.stem not in CALIBRATION_BATCHES]


def show_stats(conn):
    total = len(load_all(conn))
    done = sorted(processed_batches(conn))
    print(f"记忆库共 {total} 条记录；已入档批次: {done}")
    print("  左手想象:", class_stats(conn, 0))
    print("  右手想象:", class_stats(conn, 1))


def main():
    parser = argparse.ArgumentParser(description="EEG 状态记忆助手")
    parser.add_argument("batch", nargs="?", help="只处理指定批次，如 batch_05")
    parser.add_argument("--reset", action="store_true",
                        help="先清空记忆库（冷启动）")
    parser.add_argument("--stats", action="store_true", help="只看记忆库现状")
    args = parser.parse_args()

    conn = create_db()
    if args.reset:
        clear_db(conn)
        print("记忆库已清空，回到冷启动。\n")

    if args.stats or args.batch == "stats":
        show_stats(conn)
        return

    if not MODEL_PATH.exists():
        raise SystemExit("还没有先验模型：请先运行 analyze.py 完成标定")
    model = joblib.load(MODEL_PATH)
    REPORT_DIR.mkdir(exist_ok=True)

    if args.batch:
        path = BATCH_DIR / f"{args.batch}.npz"
        if not path.exists():
            raise SystemExit(f"批次不存在：{path}")
        out = process_batch(conn, path, model)
        if out:
            print(f"{args.batch}: 已处理，报告 -> {out}")
        else:
            print(f"{args.batch}: 已在记忆中，跳过（想重来可用 --reset）")
        return

    done, skipped = [], 0
    for path in new_batches():
        out = process_batch(conn, path, model)
        if out:
            done.append(out)
        else:
            skipped += 1
    for out in done:
        print("新报告:", out)
    if done:
        print(f"\n本轮处理 {len(done)} 个新批次；跳过已入档 {skipped} 个。")
    elif skipped:
        print("所有批次都已入档，没有新信号。\n"
              "——助手记得自己处理过什么；想重新演示请用 --reset。")
    else:
        print("没有可处理的批次。")
    show_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()

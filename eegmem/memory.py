"""记忆库（MemoryBank）——助手的外部记忆（教学第 1 步）。

作用：把每一次"新检测"的分析结果持久化保存，下次检测时翻出来比较。
类比科室的病历档案库：表头固定（时间/特征/状态/置信度），每来一次检测多一行。

技术选型：SQLite。它 Python 自带、单文件、无需服务器；
相比直接写文本文件，它支持 SQL 查询（按条件"调档案"），更接近真实系统。
"""
import sqlite3
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "memory.db"


def create_db(path=DB_PATH):
    """建表（IF NOT EXISTS：库已存在就复用，不会重复建）。返回连接。"""
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS states (
            id    INTEGER PRIMARY KEY AUTOINCREMENT,  -- 病历号，自动递增
            ts    TEXT NOT NULL,                      -- 检测时间
            batch TEXT NOT NULL,                      -- 批次编号（哪一批信号）
            c3    REAL NOT NULL,                      -- C3 通道 8-30Hz log 能量
            c4    REAL NOT NULL,                      -- C4 通道 8-30Hz log 能量
            pred  INTEGER NOT NULL,                   -- 状态: 0=左手想象 1=右手想象
            prob  REAL NOT NULL,                      -- 该状态的置信度 0~1
            note  TEXT                                -- 备注（异常原因等，可空）
        )
    """)
    conn.commit()
    return conn


def add_record(conn, c3, c4, pred, prob, batch="?", note=None):
    """写入一条记录（一次新检测 = 一行新病历）。返回新记录的病历号。"""
    ts = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO states (ts, batch, c3, c4, pred, prob, note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (ts, batch, c3, c4, pred, prob, note),
    )
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def clear_db(conn):
    """清空记忆库（教学演示用：回到冷启动）。"""
    conn.execute("DELETE FROM states")
    conn.commit()


def load_all(conn):
    """把全部档案翻出来看。"""
    return conn.execute(
        "SELECT id, ts, batch, c3, c4, pred, prob, note FROM states"
    ).fetchall()


def load_by_state(conn, pred):
    """按状态调档案：比如把所有"左手想象"的记录取出来。"""
    return conn.execute(
        "SELECT c3, c4 FROM states WHERE pred = ?", (pred,)
    ).fetchall()


def class_stats(conn, pred):
    """算某一状态的统计量：样本量、均值、样本标准差（除以 n-1）。

    记忆库的证据就藏在这里：n 越大，均值越可信、标准差越稳定。
    """
    rows = load_by_state(conn, pred)
    n = len(rows)
    if n == 0:
        return {"n": 0, "c3_mean": None, "c3_std": None,
                "c4_mean": None, "c4_std": None}
    c3 = [r[0] for r in rows]
    c4 = [r[1] for r in rows]

    def mean_std(xs):
        m = sum(xs) / n
        # n < 2 时无法估计标准差（除以 n-1 会除零）
        s = None if n < 2 else (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5
        return m, s

    c3_mean, c3_std = mean_std(c3)
    c4_mean, c4_std = mean_std(c4)
    return {"n": n, "c3_mean": c3_mean, "c3_std": c3_std,
            "c4_mean": c4_mean, "c4_std": c4_std}


if __name__ == "__main__":
    conn = create_db()
    # 演示：假设历史档案里已有 3 条"左手想象"、2 条"右手想象"
    # （数值沿用我们之前算出的量级；重复运行会再插一遍，属正常演示行为）
    add_record(conn, c3=-20.8, c4=-21.2, pred=0, prob=0.72, batch="演示")
    add_record(conn, c3=-20.6, c4=-21.0, pred=0, prob=0.65, batch="演示")
    add_record(conn, c3=-21.0, c4=-21.4, pred=0, prob=0.81, batch="演示")
    add_record(conn, c3=-21.1, c4=-20.7, pred=1, prob=0.74, batch="演示")
    add_record(conn, c3=-21.3, c4=-20.5, pred=1, prob=0.78, batch="演示")

    print("记忆库现有记录数:", len(load_all(conn)))
    print("左手想象(0)的统计:", class_stats(conn, 0))
    print("右手想象(1)的统计:", class_stats(conn, 1))
    conn.close()

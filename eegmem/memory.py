"""Transactional SQLite records, bound to one model artifact."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from eegmem.review import initialize_reviews


def create_db(path, model_id=None):
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS batches (
            name TEXT PRIMARY KEY, digest TEXT NOT NULL UNIQUE, created TEXT NOT NULL,
            results TEXT NOT NULL, evidence TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS trials (
            batch TEXT NOT NULL, trial INTEGER NOT NULL,
            c3 REAL NOT NULL, c4 REAL NOT NULL, pred INTEGER NOT NULL,
            prob REAL NOT NULL, label INTEGER, verdict TEXT NOT NULL,
            PRIMARY KEY (batch, trial)
        );
        CREATE INDEX IF NOT EXISTS trials_pred ON trials(pred);
    """)
    if model_id is not None:
        row = conn.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
        if row is not None and row[0] != model_id:
            conn.close()
            raise ValueError("Memory belongs to a different model; choose a new --database path")
        with conn:
            conn.execute("INSERT OR IGNORE INTO metadata VALUES ('model', ?)", (model_id,))
    initialize_reviews(conn)
    return conn


def load_all(conn):
    return conn.execute("SELECT * FROM trials ORDER BY batch, trial").fetchall()


def class_stats(rows, pred):
    values = np.array([[r["c3"], r["c4"]] for r in rows if r["pred"] == pred])
    n = len(values)
    mean = values.mean(axis=0).tolist() if n else [None, None]
    std = values.std(axis=0, ddof=1).tolist() if n > 1 else [None, None]
    return {"n": n, "mean": mean, "std": std}


def save_batch(conn, name, digest, results, evidence, context=None):
    """Commit every trial and its report evidence together, or roll back all of them."""
    with conn:
        conn.execute("INSERT INTO batches VALUES (?, ?, ?, ?, ?)",
                     (name, digest, datetime.now(timezone.utc).isoformat(),
                      json.dumps(results), json.dumps(evidence)))
        conn.executemany("INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [(name, r["trial"], r["c3"], r["c4"], r["pred"], r["prob"],
                           r["label"], r["verdict"]) for r in results])
        if context is not None:
            conn.execute("INSERT INTO analysis_context VALUES (?, ?)", (name, json.dumps(context)))

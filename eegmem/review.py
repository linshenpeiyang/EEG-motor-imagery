"""Searchable memory and append-only human review, separate from model predictions."""
import math
from datetime import datetime, timezone


def initialize_reviews(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch TEXT NOT NULL, trial INTEGER NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('label', 'exclude', 'reopen')),
            reviewed_label INTEGER, reviewer TEXT NOT NULL, reason TEXT NOT NULL,
            created TEXT NOT NULL,
            FOREIGN KEY(batch, trial) REFERENCES trials(batch, trial)
        );
        CREATE INDEX IF NOT EXISTS reviews_trial ON reviews(batch, trial, id);
        CREATE TABLE IF NOT EXISTS analysis_context (
            batch TEXT PRIMARY KEY, context TEXT NOT NULL
        );
    ''')


def records(conn):
    rows = conn.execute('''
        SELECT t.*, b.created, b.rowid AS batch_order,
               r.id AS review_id, r.action, r.reviewed_label, r.reviewer, r.reason
        FROM trials t JOIN batches b ON b.name=t.batch
        LEFT JOIN reviews r ON r.id=(
            SELECT MAX(id) FROM reviews WHERE batch=t.batch AND trial=t.trial
        ) ORDER BY b.rowid, t.trial
    ''').fetchall()
    output = []
    for row in rows:
        item = dict(row)
        item['review_id'] = item['review_id'] or 0
        item['status'] = {'label': 'reviewed', 'exclude': 'excluded'}.get(item['action'], 'unreviewed')
        item['effective_label'] = item['reviewed_label'] if item['status'] == 'reviewed' else item['pred']
        item['needs_review'] = item['status'] == 'unreviewed' and (
            item['prob'] < .6 or item['verdict'] == 'Outside reference')
        output.append(item)
    return output


def search(conn, query='', status='all', queue=False):
    if status not in {'all', 'reviewed', 'unreviewed', 'excluded'}:
        raise ValueError('Unknown review status')
    text = query.casefold().strip()
    return [r for r in records(conn)
            if (status == 'all' or r['status'] == status)
            and (not queue or r['needs_review'])
            and (not text or text in f"{r['batch']} {r['reviewer'] or ''} {r['reason'] or ''}".casefold())]


def add_review(conn, batch, trial, action, label, reviewer, reason, expected_version):
    """Reject stale edits; preserve the prediction and every prior review event."""
    if action not in {'label', 'exclude', 'reopen'}:
        raise ValueError('Unknown review action')
    if action == 'label' and (type(label) is not int or label not in (0, 1)):
        raise ValueError('A reviewed label must be 0 or 1')
    if action != 'label' and label is not None:
        raise ValueError('Only a label review may supply a label')
    if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 100:
        raise ValueError('Reviewer is required, up to 100 characters')
    if not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 2000:
        raise ValueError('Evidence or reason is required, up to 2000 characters')
    if type(expected_version) is not int or type(trial) is not int:
        raise ValueError('Trial and review version must be integers')
    try:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM trials WHERE batch=? AND trial=?', (batch, trial)).fetchone() is None:
            raise ValueError('Trial not found')
        latest = conn.execute('SELECT COALESCE(MAX(id), 0) FROM reviews WHERE batch=? AND trial=?',
                              (batch, trial)).fetchone()[0]
        if latest != expected_version:
            raise ValueError('Record changed since you opened it; reload before reviewing')
        cursor = conn.execute('INSERT INTO reviews (batch,trial,action,reviewed_label,reviewer,reason,created) '
                              'VALUES (?,?,?,?,?,?,?)',
                              (batch, trial, action, label, reviewer.strip(), reason.strip(),
                               datetime.now(timezone.utc).isoformat()))
        conn.commit()
        return cursor.lastrowid
    except Exception:
        conn.rollback()
        raise


def reference_snapshot(conn, policy='predicted'):
    """Reviewed labels supersede predictions; excluded records never become references."""
    if policy not in {'predicted', 'reviewed'}:
        raise ValueError('Unknown reference policy')
    own_transaction = not conn.in_transaction
    if own_transaction:
        conn.execute('BEGIN')
    try:
        selected = [r for r in records(conn) if r['status'] != 'excluded'
                    and (policy == 'predicted' or r['status'] == 'reviewed')]
        context = {'policy': policy,
                   'review_event_cutoff': conn.execute('SELECT COALESCE(MAX(id),0) FROM reviews').fetchone()[0],
                   'sources': [{'batch': r['batch'], 'trial': r['trial'], 'review_id': r['review_id']}
                               for r in selected]}
        if own_transaction:
            conn.commit()
    except Exception:
        if own_transaction:
            conn.rollback()
        raise
    return [{**r, 'model_pred': r['pred'], 'pred': r['effective_label']} for r in selected], context



def detail(conn, batch, trial, policy='predicted'):
    all_rows = records(conn)
    target = next((r for r in all_rows if r['batch'] == batch and r['trial'] == trial), None)
    if target is None:
        raise ValueError('Trial not found')
    references, _ = reference_snapshot(conn, policy)
    earlier = [r for r in references if r['batch_order'] < target['batch_order']]
    for r in earlier:
        r['distance'] = math.hypot(r['c3'] - target['c3'], r['c4'] - target['c4'])
    neighbors = sorted(earlier, key=lambda r: (r['distance'], r['batch_order'], r['trial']))[:3]
    events = [dict(r) for r in conn.execute('SELECT * FROM reviews WHERE batch=? AND trial=? ORDER BY id',
                                           (batch, trial))]
    return {'record': target, 'neighbors': neighbors, 'reviews': events,
            'retrieval_note': 'Current review state of earlier batches; not the original report snapshot.'}

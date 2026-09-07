"""Command line entry point for preparation, training, analysis and evaluation."""
import argparse
import hashlib
import json
from contextlib import closing
from pathlib import Path

import joblib

from eegmem.analyze import analyze_batch, train_model
from eegmem.compare import compare_batch
from eegmem.memory import create_db, load_all, save_batch
from eegmem.pipeline import PROJECT_ROOT, TRAIN_SUBJECTS, TEST_SUBJECTS, prepare, evaluate, subject_paths
from eegmem.report import write_report


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process_batch(conn, path, artifact, report_dir):
    """Recreate reports on repeat runs; reject changed files under an existing ID."""
    digest = file_digest(path)
    previous = conn.execute('SELECT * FROM batches WHERE name=?', (path.stem,)).fetchone()
    if previous:
        if previous['digest'] != digest:
            raise ValueError(f'{path.stem}: batch contents changed; use a new batch ID')
        results, evidence = json.loads(previous['results']), json.loads(previous['evidence'])
        added = False
    else:
        duplicate = conn.execute('SELECT name FROM batches WHERE digest=?', (digest,)).fetchone()
        if duplicate:
            raise ValueError(f'Batch content already recorded as {duplicate[0]}')
        if path.stem in artifact['training_batches'] or digest in artifact['training_digests']:
            raise ValueError('Training batches cannot be analyzed as new observations')
        trials = analyze_batch(path, artifact)
        results, evidence = compare_batch(load_all(conn), trials)
        save_batch(conn, path.stem, digest, results, evidence)
        added = True
    report = write_report(report_dir, path.stem, results, evidence)
    return report, added


def main():
    parser = argparse.ArgumentParser(description='EEG State Memory Assistant')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT,
                        help='Project output root, including data, models and reports')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('prepare', help='Download and preprocess subjects 1-10')
    commands.add_parser('train', help='Train on subjects 1-4 and save a frozen model')
    run = commands.add_parser('run', help='Analyze subjects 5-10 or one NPZ batch')
    run.add_argument('batch', nargs='?', type=Path)
    run.add_argument('--database', type=Path, help='Separate memory database for a new experiment')
    stats = commands.add_parser('stats', help='Show persistent memory counts')
    stats.add_argument('--database', type=Path)
    commands.add_parser('evaluate', help='Reproduce metrics and figure using isolated memory')
    args = parser.parse_args()
    root = args.root.resolve()
    model_path = root / 'models' / 'assistant.joblib'
    database = getattr(args, 'database', None) or root / 'data' / 'assistant.sqlite3'
    try:
        if args.command == 'prepare':
            prepare(root)
        elif args.command == 'train':
            paths = subject_paths(root, TRAIN_SUBJECTS)
            artifact = train_model(paths)
            artifact['training_digests'] = [file_digest(path) for path in paths]
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix('.joblib.tmp')
            joblib.dump(artifact, temporary)
            temporary.replace(model_path)
            print(f"Saved model trained on {artifact['training_trials']} trials: {model_path}")
        elif args.command == 'evaluate':
            evaluate(root)
        elif args.command == 'stats':
            if not database.exists():
                print('No memory yet. Run train, then run.')
                return
            with closing(create_db(database)) as conn:
                batches = conn.execute('SELECT COUNT(*) FROM batches').fetchone()[0]
                rows = load_all(conn)
                print(f'{len(rows)} trials in {batches} batches')
                for label, name in enumerate(['Left', 'Right']):
                    print(f'{name} predictions: {sum(r["pred"] == label for r in rows)}')
        else:
            if not model_path.exists():
                raise ValueError('Model not found. Run train first.')
            paths = [args.batch.resolve()] if args.batch else subject_paths(root, TEST_SUBJECTS)
            artifact = joblib.load(model_path)
            with closing(create_db(database, file_digest(model_path))) as conn:
                for path in paths:
                    report, added = process_batch(conn, path, artifact, root / 'reports' / database.stem)
                    print(f'{"Recorded" if added else "Already recorded"}: {path.stem} -> {report}')
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()

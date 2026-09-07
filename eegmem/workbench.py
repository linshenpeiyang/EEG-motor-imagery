"""Local single-user review workspace backed by a separate memory database."""
import json
import secrets
import sqlite3
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from eegmem.memory import create_db
from eegmem.review import add_review, detail, records, search

STATIC_DIR = Path(__file__).parent / 'web'


def workbench_database(root):
    target = Path(root) / 'data' / 'workbench.sqlite3'
    if not target.exists():
        source = Path(root) / 'data' / 'assistant.sqlite3'
        if not source.exists():
            raise ValueError('No assistant records found; run train and run first')
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
            with closing(sqlite3.connect(target)) as dst:
                src.backup(dst)
    with closing(create_db(target)):
        pass
    return target


def make_server(database, port=8765):
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, status, content, kind='application/json'):
            body = json.dumps(content).encode() if kind == 'application/json' else content
            self.send_response(status)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get('Host') == f'127.0.0.1:{self.server.server_port}'

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {'error': 'Use the local address printed in the terminal'})
            url = urlsplit(self.path)
            assets = {'/': ('index.html', 'text/html'), '/app.js': ('app.js', 'text/javascript'),
                      '/style.css': ('style.css', 'text/css')}
            if url.path in assets:
                name, kind = assets[url.path]
                return self.respond(200, (STATIC_DIR / name).read_bytes(), kind)
            query = parse_qs(url.query)
            try:
                with closing(create_db(database)) as conn:
                    if url.path == '/api/records':
                        all_rows = records(conn)
                        visible = search(conn, query.get('q', [''])[0], query.get('status', ['all'])[0],
                                         query.get('queue', ['0'])[0] == '1')
                        # Benchmark labels are withheld from the review workspace.
                        for row in visible:
                            row.pop('label', None)
                        metadata = conn.execute("SELECT value FROM metadata WHERE key='model'").fetchone()
                        payload = {'records': visible, 'token': token,
                                   'summary': {'trials': len(all_rows),
                                               'batches': len({r['batch'] for r in all_rows}),
                                               'reviewed': sum(r['status'] == 'reviewed' for r in all_rows),
                                               'excluded': sum(r['status'] == 'excluded' for r in all_rows),
                                               'queue': sum(r['needs_review'] for r in all_rows)},
                                   'model': metadata[0][:12] if metadata else 'unregistered'}
                    elif url.path == '/api/detail':
                        payload = detail(conn, query['batch'][0], int(query['trial'][0]),
                                         query.get('policy', ['predicted'])[0])
                        payload['record'].pop('label', None)
                        for row in payload['neighbors']:
                            row.pop('label', None)
                    else:
                        return self.respond(404, {'error': 'Not found'})
                self.respond(200, payload)
            except (ValueError, KeyError) as error:
                self.respond(400, {'error': str(error)})

        def do_POST(self):
            origin = f'http://127.0.0.1:{self.server.server_port}'
            if (not self.valid_host() or self.headers.get('Origin', origin) != origin
                    or not secrets.compare_digest(self.headers.get('X-Review-Token', ''), token)):
                return self.respond(403, {'error': 'Invalid local review token or origin'})
            if self.path != '/api/review':
                return self.respond(404, {'error': 'Not found'})
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 16384 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected a small JSON review request')
                payload = json.loads(self.rfile.read(size))
                if not isinstance(payload, dict):
                    raise ValueError('Expected an object')
                with closing(create_db(database)) as conn:
                    event = add_review(conn, payload['batch'], payload['trial'], payload['action'],
                                       payload.get('label'), payload['reviewer'], payload['reason'],
                                       payload['expected_version'])
                self.respond(200, {'review_event': event})
            except (ValueError, KeyError, TypeError, sqlite3.Error) as error:
                self.respond(400, {'error': str(error)})

    return ThreadingHTTPServer(('127.0.0.1', port), Handler)


def serve(root, port):
    database = workbench_database(root)
    with make_server(database, port) as server:
        print(f'Memory workbench: http://127.0.0.1:{server.server_port}', flush=True)
        print(f'Review database: {database}\nLocal single-user workspace. Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass

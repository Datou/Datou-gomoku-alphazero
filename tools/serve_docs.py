"""Local preview server for docs/ with the MIME types ONNX Runtime Web needs.

Python's built-in http.server serves .mjs as text/plain on some versions, and
browsers refuse to import a module with that MIME type. GitHub Pages serves the
correct type, but this script makes the README's local preview work everywhere.
"""
import functools
import http.server
import mimetypes
from pathlib import Path
import sys

mimetypes.add_type('text/javascript', '.mjs')
mimetypes.add_type('application/wasm', '.wasm')

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'docs'
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIRECTORY))
print(f'Serving {DIRECTORY} at http://127.0.0.1:{PORT}/')
http.server.ThreadingHTTPServer(('127.0.0.1', PORT), handler).serve_forever()

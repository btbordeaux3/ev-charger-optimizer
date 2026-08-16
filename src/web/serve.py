"""Locally-host the EV charger dashboard with only the Python stdlib.

Serves whatever directory you point it at (default: the dashboard output dir)
over HTTP so the interactive map, charts, and tables are viewable in a browser.

Usage:
    python -m src.web.serve [PORT] [DIR]

    python -m src.web.serve 8000 data/output
    # -> http://localhost:8000/index.html
"""
from __future__ import annotations

import argparse
import functools
import http.server
import socketserver
import sys
from pathlib import Path


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Default handler but quieter (no per-request log spam)."""

    def log_message(self, fmt, *args):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve the EV dashboard locally")
    ap.add_argument("port", nargs="?", type=int, default=8000)
    ap.add_argument("dir", nargs="?", default="data/output")
    args = ap.parse_args(argv)

    directory = str(Path(args.dir).resolve())
    handler = functools.partial(QuietHandler, directory=directory)

    with socketserver.TCPServer(("", args.port), handler) as httpd:
        print(f"▶ EV dashboard serving at:")
        print(f"  http://localhost:{args.port}/index.html")
        print(f"  (root: {directory})  — Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
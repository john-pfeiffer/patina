"""`patine-sidecar` entry point.

    patine-sidecar --data-dir <dir>      # production: random port, random token
    patine-sidecar --dev                 # fixed port 47621, token "dev", reload

On startup the process prints exactly one line to stdout:

    PATINE_READY {"url": "http://127.0.0.1:PORT", "token": "..."}

which the shell parses (apps/desktop/src-tauri/src/lib.rs)."""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import socket
import sys
from pathlib import Path

DEV_PORT = 47621
DEV_TOKEN = "dev"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="patine-sidecar", description="Patine analysis/conform engine")
    p.add_argument("--data-dir", type=Path, default=None, help="library, render cache and models directory")
    p.add_argument("--port", type=int, default=None, help="port to bind (default: random; --dev: 47621)")
    p.add_argument("--token", default=None, help="API token (default: random; --dev: 'dev')")
    p.add_argument("--dev", action="store_true", help="development mode: fixed port/token, auto-reload, repo data dir")
    p.add_argument("--workers", type=int, default=2, help="background worker threads")
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    from .paths import DataPaths, default_data_dir, repo_dev_data_dir

    data_dir = args.data_dir or (repo_dev_data_dir() if args.dev else default_data_dir())
    port = args.port or (DEV_PORT if args.dev else _free_port())
    token = args.token or (DEV_TOKEN if args.dev else secrets.token_urlsafe(24))
    level = args.log_level or ("DEBUG" if args.dev else "INFO")
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    # The server reads its configuration from the environment so uvicorn's
    # reloader (a child process) sees the same values.
    os.environ["PATINE_DATA_DIR"] = str(data_dir)
    os.environ["PATINE_TOKEN"] = token
    os.environ["PATINE_WORKERS"] = str(args.workers)
    os.environ["PATINE_PORT"] = str(port)
    DataPaths(Path(data_dir)).ensure()

    import uvicorn

    url = f"http://127.0.0.1:{port}"
    # Announce before serving: the shell starts polling /health as soon as it
    # sees this line, and uvicorn binds within milliseconds.
    sys.stdout.write(f'PATINE_READY {{"url": "{url}", "token": "{token}"}}\n')
    sys.stdout.flush()

    if args.dev:
        uvicorn.run(
            "patine.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=port,
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parent)],
            log_level="info",
        )
    else:
        from .server import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

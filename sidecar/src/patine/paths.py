"""Where the sidecar keeps its data.

One data directory holds the SQLite database, the render cache and downloaded
model weights. The shell passes it explicitly; `--dev` defaults to a folder
inside the repository so development never touches the real library.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_data_dir() -> Path:
    env = os.environ.get("PATINE_DATA_DIR")
    if env:
        return Path(env).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Patine"
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "Patine"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "patine"


def repo_dev_data_dir() -> Path:
    # sidecar/src/patine/paths.py -> repo root is four levels up
    return Path(__file__).resolve().parents[3] / ".patine-dev"


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def db(self) -> Path:
        return self.root / "library.sqlite3"

    @property
    def renders(self) -> Path:
        return self.root / "renders"

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def ensure(self) -> DataPaths:
        for p in (self.root, self.renders, self.models, self.exports, self.tmp):
            p.mkdir(parents=True, exist_ok=True)
        return self

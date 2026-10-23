# Patine sidecar

Python engine spawned by the desktop shell. Owns the SQLite library, the analysis
pipeline, the conform renderer, the render cache and export. Exposes a local
HTTP/JSON API on 127.0.0.1.

Run from source:

    uv run patine-sidecar --dev        # fixed port 47621, token "dev", reload on

See `docs/ARCHITECTURE.md` at the repository root.

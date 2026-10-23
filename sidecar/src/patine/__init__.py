"""Patine sidecar. Conform the recording, not the music."""

__version__ = "0.1.0"

# Bump when analysis outputs change shape or meaning; stored on each sample so
# stale analyses can be detected and re-run on request.
ANALYSIS_VERSION = 1

# Bump when the conform/temporal renderers change behaviour; part of the
# render-cache key so old renders are never served for new code.
RENDER_VERSION = 1

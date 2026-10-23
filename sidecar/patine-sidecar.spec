# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for the Patine sidecar (one-file executable).
# Heavy, lazily-imported packages are collected whole so librosa/scipy/numba
# work inside the bundle. Model weights are never bundled (handoff §1): they
# are fetched on first run into the app data directory.
from PyInstaller.utils.hooks import collect_all, collect_submodules

hidden = []
datas = []
binaries = []
for pkg in ["librosa", "scipy", "sklearn", "numba", "llvmlite", "soxr", "python_stretch", "pyloudnorm", "soundfile", "lazy_loader", "pooch", "mido"]:
    try:
        d, b, h = collect_all(pkg)
    except Exception:  # package may be absent in a slim env
        continue
    datas += d
    binaries += b
    hidden += h
hidden += collect_submodules("patine")
hidden += ["uvicorn.logging", "uvicorn.loops.auto", "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto", "uvicorn.lifespan.on"]

a = Analysis(
    ["src/patine/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["torch", "torchaudio", "demucs", "beat_this", "onnxruntime", "matplotlib", "tkinter", "IPython", "jupyter", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="patine-sidecar",
    debug=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)

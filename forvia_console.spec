# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


REPO = SPECPATH


def _dir(path: str, dest: str):
    return (path, dest) if os.path.isdir(path) else None


def _file(path: str, dest: str):
    return (path, dest) if os.path.isfile(path) else None


datas = []
general_cfgs = {
    "epump2_general.yaml": "epump2_general.yaml",
    "epump3_general.yaml": "epump3_general.yaml",
    "epump4_general.yaml": "epump4_general.yaml",
    "etilt1_general.yaml": "etilt1_general.yaml",
}
for item in (
    _dir(os.path.join(REPO, "web"), "web"),
    _dir(os.path.join(REPO, "web", "forvia_label_v2", "backend"), "web/forvia_label_v2/backend"),
    _dir(os.path.join(REPO, "web", "forvia_train_v2", "backend"), "web/forvia_train_v2/backend"),
    _dir(os.path.join(REPO, "web", "forvia_label_v2"), "web/forvia_label_v2"),
    _dir(os.path.join(REPO, "web", "forvia_train_v2"), "web/forvia_train_v2"),
    _dir(os.path.join(REPO, "cfg", "core"), "cfg/core"),
    _dir(os.path.join(REPO, "data_manager"), "data_manager"),
    _dir(os.path.join(REPO, "data_augmentation"), "data_augmentation"),
    _dir(os.path.join(REPO, "ml"), "ml"),
    _dir(os.path.join(REPO, "quick_test"), "quick_test"),
):
    if item:
        datas.append(item)
for target, source_name in general_cfgs.items():
    path = os.path.join(REPO, "cfg", source_name)
    if os.path.isfile(path):
        # PyInstaller's destination is a directory.  Passing the filename
        # here creates cfg/examples/<name>.yaml/<name>.yaml in _internal.
        datas.append((path, "cfg/examples"))

sample_view = os.path.join(REPO, "web", "forvia_label_v2", "vendor", "sample_view")
if os.path.isdir(sample_view):
    datas.append((sample_view, "sample_view"))


def _no_tests(name: str) -> bool:
    parts = set(name.split("."))
    return not ({"tests", "testing", "conftest"} & parts or name.endswith(".__main__"))


hiddenimports = [
    "fastapi", "starlette", "pydantic", "pydantic_core", "multipart",
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "anyio",
    "yaml", "numpy", "pandas", "openpyxl", "joblib",
    "nptdms", "zstandard", "soundfile",
    "plotly", "plotly.graph_objs", "plotly.subplots", "plotly.io",
    "scipy", "scipy.io", "scipy.io.wavfile", "scipy.signal", "scipy.stats",
    "pywt", "PyEMD", "librosa", "numba", "llvmlite",
    "sklearn", "xgboost", "lightgbm", "matplotlib",
    "forvia_label_v2.backend.main", "forvia_train_v2.backend.main",
]

for pkg in (
    "forvia_label_v2.backend",
    "forvia_train_v2.backend",
    "forvia_label_v2.backend.cards",
    "forvia_label_v2.backend.tasks",
    "backend.cards",
    "backend.tasks",
    "ml",
    "data_augmentation",
):
    try:
        hiddenimports += collect_submodules(pkg, filter=_no_tests)
    except Exception:
        pass

binaries = []
for pkg in ("uvicorn", "plotly", "librosa", "PyEMD", "matplotlib"):
    datas += collect_data_files(pkg, include_py_files=False)
    hiddenimports += collect_submodules(pkg, filter=_no_tests)

for pkg in (
    "numpy", "pandas", "scipy", "pywt", "nptdms", "numba", "llvmlite",
    "sklearn", "xgboost", "lightgbm", "joblib", "openpyxl",
):
    datas += collect_data_files(pkg, include_py_files=False)
    binaries += collect_dynamic_libs(pkg)

for pkg in ("librosa", "numba", "llvmlite", "scikit-learn", "xgboost", "lightgbm"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass


a = Analysis(
    [os.path.join(REPO, "forvia_console_launcher.py")],
    pathex=[REPO, os.path.join(REPO, "web"), os.path.join(REPO, "web", "forvia_label_v2"), os.path.join(REPO, "web", "forvia_train_v2")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "streamlit", "torch", "tensorflow", "catboost",
        "pyarrow", "pyarrow.*", "cv2", "skimage", "statsmodels", "IPython", "jedi",
        "azure", "azure.*", "google.cloud", "google.cloud.*", "boto3", "botocore", "s3fs", "gcsfs",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="ForviaAI2Console",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=True, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ForviaAI2Console")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Forvia AI2.0 Console.app",
        icon=None,
        bundle_identifier="com.forvia.ai2.console",
        info_plist={"LSBackgroundOnly": False, "NSHighResolutionCapable": True},
    )

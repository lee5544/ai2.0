# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


datas = [
    ("forvia_label", "forvia_label"),
    ("data_manager", "data_manager"),
    ("ml", "ml"),
    ("cfg", "cfg"),
]
hiddenimports = [
    "numpy",
    "pandas",
    "yaml",
    "nptdms",
    "zstandard",
    "plotly",
    "plotly.graph_objs",
    "plotly.subplots",
    "plotly.io",
    "scipy",
    "scipy.io",
    "scipy.io.wavfile",
    "scipy.signal",
    "scipy.stats",
    "pywt",
    "PyEMD",
    "librosa",
    "soundfile",
    "numba",
    "llvmlite",
    "sklearn",
    "sklearn.base",
    "sklearn.preprocessing",
    "sklearn.decomposition",
    "sklearn.feature_extraction",
    "sklearn.metrics",
]
binaries = []


def _exclude_tests(name: str) -> bool:
    parts = set(name.split("."))
    return not (
        "tests" in parts
        or "testing" in parts
        or "conftest" in parts
        or name.endswith(".__main__")
        or ".tests." in name
    )


for package in ("streamlit", "altair", "plotly", "librosa"):
    datas += collect_data_files(package, include_py_files=False)
    hiddenimports += collect_submodules(package, filter=_exclude_tests)

for package in ("numpy", "pandas", "scipy", "pywt", "nptdms", "numba", "llvmlite"):
    datas += collect_data_files(package, include_py_files=False)
    binaries += collect_dynamic_libs(package)

datas += collect_data_files("streamlit", include_py_files=False)
datas += copy_metadata("streamlit")


a = Analysis(
    ["scripts/forvia_external_folder_label_launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ForviaExternalFolderLabel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="ForviaExternalFolderLabel",
)

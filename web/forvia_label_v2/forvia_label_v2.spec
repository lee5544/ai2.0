# -*- mode: python ; coding: utf-8 -*-
# Forvia 标注 v2 · PyInstaller 打包（双击运行，无需装 Python/Docker）。
# 在仓库根 AI-2.0 下执行： pyinstaller web/forvia_label_v2/forvia_label_v2.spec
import os
from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata,
)

# 路径一律用绝对路径，避免 PyInstaller 把 spec 里的相对路径相对“spec 目录”解析而出错。
HERE = SPECPATH                       # .../web/forvia_label_v2（spec 所在目录）
REPO = os.path.dirname(os.path.dirname(HERE))  # .../AI-2.0（仓库根）

# 动态发现的卡片 / 任务模块（discover() 用 importlib 加载，PyInstaller 不会自动包含）→ 显式列出
_dyn = []
for sub in ("cards", "tasks"):
    d = os.path.join(HERE, "backend", sub)
    if os.path.isdir(d):
        _dyn += [f"backend.{sub}.{f[:-3]}" for f in os.listdir(d)
                 if f.endswith(".py") and not f.startswith("__")]

# 随包资源：前端 + 复用的 v1 子集（保持目录结构，供 config_paths / sys.path 解析）
# 复用的 v1 子集：data_manager / cfg 用仓库根那份（运行时 sys.path 指向 REPO_ROOT 用的就是它们）；
# sample_view 仓库根没有，只在 web/forvia_label_v2/vendor 下有（data_manager 里按需懒导入），用 vendor 那份。
_SV = os.path.join(HERE, "vendor", "sample_view")
if not os.path.isdir(_SV):
    _SV = os.path.join(REPO, "sample_view")        # 兜底：若以后挪到仓库根
datas = [
    (os.path.join(HERE), "web/forvia_label_v2"),
    (os.path.join(REPO, "data_manager"), "data_manager"),
    (os.path.join(REPO, "cfg"), "cfg"),
]
if os.path.isdir(_SV):
    datas.append((_SV, "sample_view"))
hiddenimports = [
    # Web 框架
    "fastapi", "starlette", "pydantic", "pydantic_core", "multipart",
    "uvicorn", "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on", "anyio",
    # 数据 / 信号 / 频谱
    "numpy", "pandas", "yaml", "nptdms", "zstandard",
    "plotly", "plotly.graph_objs", "plotly.subplots", "plotly.io",
    "scipy", "scipy.io", "scipy.io.wavfile", "scipy.signal", "scipy.stats",
    "pywt", "PyEMD", "librosa", "soundfile", "numba", "llvmlite",
] + _dyn
binaries = []


def _no_tests(name: str) -> bool:
    parts = set(name.split("."))
    return not ({"tests", "testing", "conftest"} & parts or name.endswith(".__main__"))


for pkg in ("uvicorn", "plotly", "librosa", "PyEMD"):
    datas += collect_data_files(pkg, include_py_files=False)
    hiddenimports += collect_submodules(pkg, filter=_no_tests)

for pkg in ("numpy", "pandas", "scipy", "pywt", "nptdms", "numba", "llvmlite"):
    datas += collect_data_files(pkg, include_py_files=False)
    binaries += collect_dynamic_libs(pkg)

for pkg in ("librosa", "numba", "llvmlite"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass


a = Analysis(
    [os.path.join(HERE, "launcher.py")],
    pathex=[REPO, os.path.dirname(HERE), HERE],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter", "matplotlib", "streamlit", "torch", "tensorflow",
        # Heavy optional deps pulled by pandas/librosa ecosystems; label v2 does not use them.
        "pyarrow", "pyarrow.*", "cv2", "skimage", "statsmodels", "IPython", "jedi",
        "azure", "azure.*", "google.cloud", "google.cloud.*", "boto3", "botocore", "s3fs", "gcsfs",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Forvia标注v2",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=True, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="Forvia标注v2")

# .app 仅 macOS 有意义；Windows/Linux 上跳过，产物就是 dist/Forvia标注v2/Forvia标注v2(.exe)
import sys as _sys
if _sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="Forvia标注v2.app",
        icon=None,
        bundle_identifier="com.forvia.label.v2",
        info_plist={"LSBackgroundOnly": False, "NSHighResolutionCapable": True},
    )

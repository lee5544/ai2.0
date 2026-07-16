"""Forvia 标注 v2 · 双击启动器（打包成 .app/.exe 用）。

- 自动找一个空闲端口启动本地服务（uvicorn + FastAPI）。
- 把打包进来的 data_manager / sample_view / cfg 注入路径（FORVIA_REPO_ROOT）。
- 标注库 / 任务配置写到用户可写目录（FORVIA_DATA_HOME），不写进只读的程序包。
- 启动后自动打开浏览器到本地页面。
"""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def _bundle_root() -> Path:
    # PyInstaller 解包目录；开发时为仓库根
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[2]  # 源码运行：仓库根 AI-2.0


def _user_data_home() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "ForviaLabelV2"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / "ForviaLabelV2"
    else:
        base = Path.home() / ".forvia_label_v2"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _free_port(preferred: int = 8000) -> int:
    for port in (preferred, 8123, 8200, 8765, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
        except OSError:
            continue
    return preferred


def main() -> None:
    root = _bundle_root()
    # 复用的 v1 子集（data_manager / sample_view / cfg）随包；让 config.py 找到它们
    os.environ.setdefault("FORVIA_REPO_ROOT", str(root))
    for extra in (str(root), str(root / "web"), str(root / "web" / "forvia_label_v2")):
        if extra not in sys.path:
            sys.path.insert(0, extra)

    # 可写的数据目录（标注库 / 任务配置 / 缓存）
    data_home = _user_data_home()
    os.environ.setdefault("FORVIA_DATA_HOME", str(data_home))
    os.environ.setdefault("FORVIA_TDMS_CACHE_DIR", str(data_home / "tdms_cache"))

    # 打包成 .app 后没有可见终端，把 stdout/stderr（含 traceback）落到日志文件，便于排错
    log_path = data_home / "app.log"
    try:
        _logf = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _logf
        sys.stderr = _logf
    except Exception:
        pass

    from backend.main import app  # noqa: E402  （触发 sys.path 已就绪后再导入）
    import uvicorn

    port = _free_port(8000)
    url = f"http://127.0.0.1:{port}"
    print(f"[启动] FORVIA_REPO_ROOT={os.environ.get('FORVIA_REPO_ROOT')} "
          f"_MEIPASS={getattr(sys, '_MEIPASS', None)} 日志={log_path}")

    def _open():
        time.sleep(1.5)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()
    print(f"Forvia 标注 v2 运行中：{url}\n（关闭此窗口即退出程序）")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

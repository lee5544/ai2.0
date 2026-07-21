"""原生本地文件/文件夹选择对话框（在运行后端的本机弹出）。

macOS 用 osascript，其它平台用 tkinter。仅适用于本地运行；
远程/容器部署时对话框无法弹出，前端可退回手动粘贴路径。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


class PickCanceled(Exception):
    pass


def _mac_choose(kind: str, title: str, initial: str | None) -> str | None:
    start = str(Path(initial).expanduser()) if initial else str(Path.home())
    prompt = title.replace('"', '\\"')
    posix = start.replace('"', '\\"')
    if kind == "dir":
        lines = [
            f'set d to POSIX file "{posix}"',
            f'set f to choose folder with prompt "{prompt}" default location d',
            "return POSIX path of f",
        ]
    else:  # file（限定 csv）
        lines = [
            f'set d to POSIX file "{posix}"',
            f'set f to choose file with prompt "{prompt}" of type {{"csv","public.comma-separated-values-text"}} default location d',
            "return POSIX path of f",
        ]
    proc = subprocess.run(
        ["osascript", *sum([["-e", x] for x in lines], [])],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
    )
    if proc.returncode == 0:
        return (proc.stdout or "").strip() or None
    err = (proc.stderr or "").strip()
    if "User canceled" in err or "-128" in err:
        raise PickCanceled()
    raise RuntimeError(f"osascript: {err or proc.returncode}")


def _tk_choose(kind: str, title: str, initial: str | None) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    start = str(Path(initial).expanduser()) if initial else str(Path.home())
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    if kind == "dir":
        sel = filedialog.askdirectory(title=title, initialdir=start, mustexist=True)
    else:
        sel = filedialog.askopenfilename(
            title=title, initialdir=start,
            filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
    root.destroy()
    if not sel:
        raise PickCanceled()
    return str(Path(sel).expanduser())


def pick(kind: str, title: str = "", initial: str | None = None) -> str | None:
    """kind: 'file' | 'dir'。返回路径；用户取消抛 PickCanceled。"""
    title = title or ("选择目录" if kind == "dir" else "选择文件")
    if sys.platform == "darwin":
        return _mac_choose(kind, title, initial)
    return _tk_choose(kind, title, initial)

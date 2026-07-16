from __future__ import annotations

import os
import runpy
import shutil
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn


LABEL_PORT = 8012
TRAIN_PORT = 8001
APP_NAME = "ForviaAI2Console"
BUILTIN_EXAMPLES = {
    "epump2_general.yaml": "epump2_general.yaml",
    "epump3_general.yaml": "epump3_general.yaml",
    "epump4_general.yaml": "epump4_general.yaml",
    "etilt1_general.yaml": "etilt1_general.yaml",
}


def _bundle_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass).resolve() if meipass else Path(__file__).resolve().parent


def _data_home() -> Path:
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / APP_NAME
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / APP_NAME
    else:
        base = Path.home() / f".{APP_NAME.lower()}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _copy_default_cfg(root: Path, data_home: Path) -> None:
    dst = data_home / "cfg"
    core_src = root / "cfg" / "core"
    core_dst = dst / "core"
    if core_src.exists():
        core_dst.mkdir(parents=True, exist_ok=True)
        for src in core_src.iterdir():
            target = core_dst / src.name
            if target.exists():
                continue
            if src.is_dir():
                shutil.copytree(src, target)
            else:
                shutil.copy2(src, target)
    examples_dst = dst / "examples"
    examples_dst.mkdir(parents=True, exist_ok=True)
    for target, source_name in BUILTIN_EXAMPLES.items():
        for src in (root / "cfg" / "examples" / target, root / "cfg" / source_name):
            if src.exists():
                shutil.copy2(src, examples_dst / target)
                break


def _setup_runtime(root: Path, data_home: Path) -> None:
    _copy_default_cfg(root, data_home)
    os.environ.setdefault("FORVIA_REPO_ROOT", str(root))
    os.environ.setdefault("FORVIA_DATA_HOME", str(data_home))
    os.environ.setdefault("FORVIA_CONSOLE_FROZEN", "1" if getattr(sys, "frozen", False) else "0")
    os.environ.setdefault("FORVIA_CFG_DIR", str(data_home / "cfg"))
    os.environ.setdefault("FORVIA_RESULTS_DIR", str(data_home / "results"))
    os.environ.setdefault("FORVIA_TRAIN_V2_STATE_DIR", str(data_home / "train_state"))
    os.environ.setdefault("FORVIA_TDMS_CACHE_DIR", str(data_home / "tdms_cache"))
    for item in (root, root / "web", root / "web" / "forvia_label_v2", root / "web" / "forvia_train_v2"):
        text = str(item)
        if item.exists() and text not in sys.path:
            sys.path.insert(0, text)


def _dispatch_python(root: Path, data_home: Path) -> bool:
    if len(sys.argv) < 2 or sys.argv[1] != "--forvia-python":
        return False
    _setup_runtime(root, data_home)
    args = sys.argv[2:]
    if not args:
        raise SystemExit("--forvia-python 缺少模块或脚本参数")
    if args[0] == "-m":
        if len(args) < 2:
            raise SystemExit("-m 缺少模块名")
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__")
        return True
    script = Path(args[0])
    if not script.is_absolute():
        script = root / script
    sys.argv = [str(script), *args[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return True


def _port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _serve(app, port: int) -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    uvicorn.Server(config).run()


def _start_server(import_path: str, port: int) -> None:
    if _port_open(port):
        return
    module_name, app_name = import_path.split(":", 1)
    module = __import__(module_name, fromlist=[app_name])
    app = getattr(module, app_name)
    thread = threading.Thread(target=_serve, args=(app, port), daemon=True)
    thread.start()


def _redirect_logs(data_home: Path) -> None:
    log_path = data_home / "forvia_console.log"
    log = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = log
    sys.stderr = log


def main() -> None:
    root = _bundle_root()
    data_home = _data_home()
    if _dispatch_python(root, data_home):
        return
    _setup_runtime(root, data_home)
    _redirect_logs(data_home)
    print(f"\n===== Forvia Console start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"root={root}")
    print(f"data_home={data_home}")
    _start_server("forvia_label_v2.backend.main:app", LABEL_PORT)
    _start_server("forvia_train_v2.backend.main:app", TRAIN_PORT)
    for _ in range(60):
        if _port_open(LABEL_PORT) and _port_open(TRAIN_PORT):
            break
        time.sleep(0.25)
    webbrowser.open(f"http://127.0.0.1:{TRAIN_PORT}/console")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bundle_root() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parents[1]


def main() -> None:
    root = _bundle_root()
    app_path = root / "forvia_label" / "external_folder_app.py"
    os.environ["FORVIA_LABEL_EXTERNAL_FOLDER_ONLY"] = "1"
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    for extra in (root, root / "forvia_label"):
        extra_str = str(extra)
        if extra_str not in sys.path:
            sys.path.insert(0, extra_str)

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--browser.gatherUsageStats=false",
        "--server.headless=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()

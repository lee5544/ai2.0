from __future__ import annotations

import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("FORVIA_REPO_ROOT", APP_DIR.parent)).expanduser()
FRONTEND_DIR = APP_DIR / "frontend"
CFG_DIR = Path(os.environ.get("FORVIA_CFG_DIR", PROJECT_ROOT / "cfg")).expanduser()
RESULTS_DIR = Path(os.environ.get("FORVIA_RESULTS_DIR", PROJECT_ROOT / "results")).expanduser()
STATE_DIR = Path(os.environ.get("FORVIA_TRAIN_V2_STATE_DIR", APP_DIR / "state")).expanduser()
RUN_DIR = STATE_DIR / "runs"
DATABASE_PATH = STATE_DIR / "forvia_train_v2.db"


def ensure_dirs() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

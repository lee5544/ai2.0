from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from .run_manager import _execute
from .store import get_run, now, update_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Forvia Train v2 独立运行任务执行器")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    try:
        run = get_run(args.run_id)
        if not run:
            raise KeyError(f"运行记录不存在: {args.run_id}")
        update_run(args.run_id, supervisor_pid=os.getpid())

        config_path = Path(str(run["config_path"])).expanduser()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(config, dict):
            raise ValueError(f"配置文件格式错误: {config_path}")

        log_path = Path(str(run.get("log_path") or "")).expanduser()
        job_path = log_path.parent / "job.json"
        command = None
        if job_path.is_file():
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if isinstance(job, dict) and isinstance(job.get("command"), list):
                command = [str(part) for part in job["command"]]

        _execute(args.run_id, config, command)
    except Exception as exc:
        if get_run(args.run_id):
            update_run(
                args.run_id,
                status="failed",
                error=str(exc),
                pid=None,
                supervisor_pid=None,
                finished_at=now(),
            )
        raise


if __name__ == "__main__":
    main()

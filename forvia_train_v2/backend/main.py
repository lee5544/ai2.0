from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from ml.train import list_registered_model_types
from ml.training.app_api import (
    config_path_for_project,
    data_manager_defaults,
    dataset_summary,
    dump_yaml,
    line_reference_overview,
    model_id,
    normalize_project_payload,
    parse_yaml,
    save_project_config,
    training_app_options,
    training_data_options,
    validate_config,
    xlsx_model_overview,
)
from ml.training.results import collect_results, find_model_dir
from data_augmentation import AUGMENTATION_METHODS

from . import database_jobs, run_manager, store
from .config import (
    CFG_DIR,
    FRONTEND_DIR,
    PROJECT_ROOT,
    RUN_DIR,
    ensure_dirs,
)


app = FastAPI(title="Forvia Train v2")


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    store.init_db()
    run_manager.reconcile_active_runs()
    _sync_cfg_projects()


def _sync_cfg_projects() -> None:
    indexed_paths = {
        str(Path(project.get("config_path") or "").expanduser().resolve())
        for project in store.list_projects()
        if project.get("config_path")
    }
    for path in sorted(CFG_DIR.glob("*.yaml")):
        resolved = str(path.resolve())
        if resolved in indexed_paths:
            continue
        try:
            config = parse_yaml(path.read_text(encoding="utf-8"))
            line_name = str(config.get("line_name") or "").strip()
            model_name = str((config.get("model") or {}).get("model_name") or "").strip()
            model_type = str((config.get("train") or {}).get("model_type") or "xgb").strip()
            if not line_name or not model_name or model_type not in list_registered_model_types():
                continue
            payload = normalize_project_payload(
                {
                    "name": path.stem,
                    "line_name": line_name,
                    "model_name": model_name,
                    "model_type": model_type,
                    "config": config,
                }
            )
            payload["config_path"] = resolved
            store.create_project(payload)
            indexed_paths.add(resolved)
        except Exception:
            continue
    for project in store.list_projects():
        if project.get("config_path"):
            continue
        payload = normalize_project_payload(project)
        target = config_path_for_project(payload)
        if target.is_file():
            payload["config"] = parse_yaml(target.read_text(encoding="utf-8"))
            payload["config_path"] = str(target.resolve())
        else:
            payload = save_project_config(payload)
        store.update_project(project["id"], payload)


def _require_project(project_id: str) -> dict:
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


class ProjectReq(BaseModel):
    name: str = ""
    line_name: str
    model_name: str
    model_type: str = "xgb"
    config: dict = {}
    config_path: str = ""   # 可在界面修改配置路径；变化时重命名/迁移 yaml


@app.get("/api/options")
def api_options():
    options = training_app_options()
    options["run_stages"] = {
        kind: [{"id": stage_id, "title": title} for stage_id, title, _ in stages]
        for kind, stages in run_manager.STAGES.items()
    }
    options["run_stages"]["augmentation"] = [{"id": "augmentation", "title": "原始信号增强"}]
    options["augmentation_methods"] = [
        {"id": method_id, "name": name}
        for method_id, name in AUGMENTATION_METHODS.items()
    ]
    return options


@app.get("/api/projects")
def api_projects():
    _sync_cfg_projects()
    return {"projects": store.list_projects()}


@app.post("/api/projects")
def api_create_project(req: ProjectReq):
    try:
        payload = save_project_config(normalize_project_payload(req.model_dump()))
        return {"project": store.create_project(payload)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/projects/{project_id}")
def api_get_project(project_id: str):
    project = _require_project(project_id)
    return {"project": project, "yaml": dump_yaml(project["config"])}


@app.put("/api/projects/{project_id}")
def api_update_project(project_id: str, req: ProjectReq):
    existing = _require_project(project_id)
    try:
        old_path = str(existing.get("config_path") or "")
        desired = str(req.config_path or "").strip()
        # 用户在界面改了配置路径 → 写到新路径，并删除旧 yaml（迁移/重命名）。
        if desired and (not old_path or Path(desired).expanduser().resolve() != Path(old_path).expanduser().resolve()):
            payload = save_project_config(
                normalize_project_payload(req.model_dump()),
                str(Path(desired).expanduser()),
            )
            new_path = Path(str(payload.get("config_path") or "")).resolve()
            if old_path:
                old_p = Path(old_path).expanduser()
                if old_p.exists() and old_p.resolve() != new_path:
                    try:
                        old_p.unlink()
                    except OSError:
                        pass
        else:
            payload = save_project_config(
                normalize_project_payload(req.model_dump()),
                old_path,
            )
        return {"project": store.update_project(project_id, payload)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/projects/{project_id}")
def api_delete_project(project_id: str):
    project = _require_project(project_id)
    active = [r for r in store.list_runs(project_id) if r["status"] in ("queued", "running")]
    if active:
        raise HTTPException(status_code=409, detail="项目仍有运行中的任务，请先取消任务")
    config_path = Path(str(project.get("config_path") or "")).expanduser()
    try:
        resolved = config_path.resolve()
        if resolved.is_file() and CFG_DIR.resolve() in resolved.parents:
            resolved.unlink()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"删除配置 YAML 失败: {exc}")
    return {"ok": store.delete_project(project_id)}


class DuplicateReq(BaseModel):
    name: str = ""
    model_name: str = ""


@app.post("/api/projects/{project_id}/duplicate")
def api_duplicate_project(project_id: str, req: DuplicateReq):
    project = _require_project(project_id)
    model_name = req.model_name.strip() or f"{project['model_name']}_copy"
    config = copy.deepcopy(project["config"])
    config.setdefault("model", {})["model_name"] = model_name
    payload = normalize_project_payload(
        {
            "name": req.name.strip() or f"{project['name']} 副本",
            "line_name": project["line_name"],
            "model_name": model_name,
            "model_type": project["model_type"],
            "config": config,
        }
    )
    return {"project": store.create_project(save_project_config(payload))}


class YamlReq(BaseModel):
    yaml: str


@app.put("/api/projects/{project_id}/yaml")
def api_update_yaml(project_id: str, req: YamlReq):
    project = _require_project(project_id)
    try:
        config = parse_yaml(req.yaml)
        payload = normalize_project_payload(
            {
                "name": project["name"],
                "line_name": config.get("line_name") or project["line_name"],
                "model_name": (config.get("model") or {}).get("model_name") or project["model_name"],
                "model_type": (config.get("train") or {}).get("model_type") or project["model_type"],
                "config": config,
            }
        )
        payload = save_project_config(payload, str(project.get("config_path") or ""))
        return {"project": store.update_project(project_id, payload), "yaml": dump_yaml(payload["config"])}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/projects/{project_id}/validate")
def api_validate(project_id: str):
    project = _require_project(project_id)
    return validate_config(project["config"])


@app.post("/api/projects/{project_id}/validate-yaml")
def api_validate_yaml(project_id: str, req: YamlReq):
    _require_project(project_id)
    try:
        config = parse_yaml(req.yaml)
        return validate_config(config)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/projects/{project_id}/dataset-summary")
def api_dataset_summary(project_id: str):
    project = _require_project(project_id)
    return dataset_summary(project["config"])


@app.get("/api/projects/{project_id}/line-reference-overview")
def api_line_reference_overview(project_id: str):
    project = _require_project(project_id)
    return line_reference_overview(project["config"])


@app.get("/api/projects/{project_id}/xlsx-model-overview")
def api_xlsx_model_overview(project_id: str):
    project = _require_project(project_id)
    return xlsx_model_overview(project["config"])


@app.get("/api/projects/{project_id}/training-data-options")
def api_training_data_options(project_id: str):
    project = _require_project(project_id)
    return training_data_options(project["config"])


class DatabaseActionReq(BaseModel):
    action: str
    source_folder: str = ""
    label_csvs: list[str] = Field(default_factory=list)
    source_label_db: str = ""
    output_data_root: str = ""
    label_records_db_path: str = ""
    tdms_root: str = ""
    storage_root: str = "factory_raw"
    line: str = ""
    file_mode: str = "manual"
    workers: int = 0


@app.post("/api/projects/{project_id}/database-actions")
def api_database_action(project_id: str, req: DatabaseActionReq):
    project = _require_project(project_id)
    try:
        return {"job": database_jobs.start(project, req.model_dump())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/database-actions/{job_id}")
def api_database_action_status(job_id: str):
    job = database_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="数据库任务不存在")
    return {"job": job}


class RunReq(BaseModel):
    kind: str


class AugmentationFolderReq(BaseModel):
    path: str
    count: int = 1


class AugmentationReq(BaseModel):
    input_folders: list[AugmentationFolderReq] = Field(default_factory=list)
    output_dir: str = ""
    methods: list[str] = Field(default_factory=list)
    line: str = ""
    seed: int | None = None


@app.post("/api/projects/{project_id}/runs")
def api_start_run(project_id: str, req: RunReq):
    try:
        return {"run": run_manager.start(project_id, req.kind)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/projects/{project_id}/augmentation")
def api_start_augmentation(project_id: str, req: AugmentationReq):
    project = _require_project(project_id)
    if not req.methods:
        raise HTTPException(status_code=400, detail="至少选择一种数据增强方法")
    folder_counts: dict[Path, int] = {}
    for item in req.input_folders:
        raw_path, raw_count = item.path, item.count
        if not raw_path.strip():
            continue
        count = int(raw_count)
        if count < 1:
            raise HTTPException(status_code=400, detail="每个文件夹的 sample 生成数量必须大于 0")
        folder_counts[Path(raw_path).expanduser().resolve()] = count
    if not folder_counts:
        raise HTTPException(status_code=400, detail="请至少选择一个包含 TDMS / TDMS.ZST 的文件夹")
    invalid_folders = [folder for folder in folder_counts if not folder.is_dir()]
    if invalid_folders:
        raise HTTPException(status_code=400, detail=f"输入文件夹不存在: {invalid_folders[0]}")
    configured_root = Path(str(project["config"].get("results_path") or "./results/")).expanduser()
    results_root = configured_root if configured_root.is_absolute() else PROJECT_ROOT / configured_root
    output_dir = (results_root / model_id(project["config"]) / "dataset_csv").resolve()
    command = [
        "-m", "data_augmentation.direct_features",
        "--config", "{config}",
        "--output-dir", str(output_dir),
        "--methods", ",".join(req.methods),
    ]
    for input_folder, count in folder_counts.items():
        command.extend(["--input-folder", str(input_folder), "--folder-count", str(count)])
    if req.seed is not None:
        command.extend(["--seed", str(req.seed)])
    if req.line.strip():
        command.extend(["--line", req.line.strip()])
    try:
        return {"run": run_manager.start(project_id, "augmentation", command=command)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/runs")
def api_runs(project_id: str = ""):
    return {"runs": store.list_runs(project_id)}


@app.get("/api/runs/{run_id}")
def api_run(run_id: str):
    run = store.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return {"run": run, "log": run_manager.tail_log(run_id)}


@app.post("/api/runs/{run_id}/cancel")
def api_cancel_run(run_id: str):
    run = run_manager.cancel(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="运行不存在")
    return {"run": run}


@app.get("/api/projects/{project_id}/results")
def api_results(project_id: str):
    project = _require_project(project_id)
    latest = store.latest_successful_run(project_id, ("train", "full"))
    result = collect_results(project["config"])
    result["latest_run"] = latest
    return result


@app.get("/api/artifact")
def api_artifact(path: str):
    target = Path(path).expanduser().resolve()
    allowed = [RUN_DIR.resolve(), (PROJECT_ROOT / "results").resolve()]
    if not any(target == root or root in target.parents for root in allowed):
        raise HTTPException(status_code=403, detail="不允许访问该路径")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)


class PredictReq(BaseModel):
    mode: str = "single"
    input_path: str
    output_path: str = ""
    threshold: float | None = None


class NativePickerReq(BaseModel):
    mode: str = "file"
    start_path: str = ""


@app.post("/api/projects/{project_id}/predict")
def api_predict(project_id: str, req: PredictReq):
    project = _require_project(project_id)
    model_dir = find_model_dir(project["config"])
    input_path = Path(req.input_path).expanduser()
    if req.mode == "single":
        script = model_dir / "main_predict.py"
        if not input_path.is_file():
            raise HTTPException(status_code=400, detail="输入 TDMS / TDMS.ZST 文件不存在")
        command = [str(script), "--tdms", str(input_path), "--line", project["line_name"]]
    else:
        script = model_dir / "quick_test" / "predict_folder.py"
        if not input_path.is_dir():
            raise HTTPException(status_code=400, detail="输入文件夹不存在")
        command = [
            str(script), "--folder", str(input_path), "--model-dir", str(model_dir),
            "--line", project["line_name"],
        ]
        if req.output_path:
            command.extend(["--output", req.output_path])
    if not script.is_file():
        raise HTTPException(status_code=400, detail=f"推理脚本不存在: {script}")
    if req.threshold is not None:
        command.extend(["--threshold", str(req.threshold)])
    return {"run": run_manager.start(project_id, "predict", command=command)}


@app.get("/api/browse")
def api_browse(path: str = ""):
    base = Path(path).expanduser() if path else Path.home()
    try:
        base = base.resolve()
        entries = list(base.iterdir())[:3000]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    dirs = sorted([e.name for e in entries if e.is_dir() and not e.name.startswith(".")], key=str.lower)
    files = sorted(
        [e.name for e in entries if e.is_file() and not e.name.startswith(".") and e.suffix.lower() in {".yaml", ".yml", ".csv", ".db", ".tdms", ".zst"}],
        key=str.lower,
    )
    return {"path": str(base), "parent": str(base.parent) if base.parent != base else "", "dirs": dirs, "files": files}


@app.post("/api/native-picker")
def api_native_picker(req: NativePickerReq):
    if sys.platform != "darwin":
        raise HTTPException(status_code=501, detail="当前系统不支持原生文件选择器")
    if req.mode not in {"file", "dir", "save"}:
        raise HTTPException(status_code=400, detail="不支持的选择类型")

    start = Path(req.start_path).expanduser() if req.start_path else Path.home()
    if req.mode in {"file", "save"} and not start.is_dir():
        start = start.parent
    if not start.is_dir():
        start = Path.home()
    script = """
on run argv
    set pickerMode to item 1 of argv
    set startFolder to POSIX file (item 2 of argv)
    try
        if pickerMode is "dir" then
            set selectedItem to choose folder with prompt "选择文件夹" default location startFolder
        else if pickerMode is "save" then
            set selectedItem to choose file name with prompt "选择输出文件" default location startFolder
        else
            set selectedItem to choose file with prompt "选择文件" default location startFolder
        end if
        return POSIX path of selectedItem
    on error number -128
        return ""
    end try
end run
"""
    try:
        result = subprocess.run(
            ["osascript", "-e", script, req.mode, str(start.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise HTTPException(status_code=500, detail=f"无法打开系统文件选择器: {detail.strip()}") from exc
    path = result.stdout.strip()
    return {"path": path, "cancelled": not bool(path)}


@app.get("/api/browse-roots")
def api_browse_roots():
    defaults = data_manager_defaults()
    roots = [
        {"label": "用户目录", "path": str(Path.home())},
        {"label": "项目目录", "path": str(PROJECT_ROOT)},
    ]
    data_root = str(defaults.get("data_root") or "").strip()
    db_path = str(defaults.get("label_records_db_path") or "").strip()
    if data_root:
        roots.append({"label": "默认数据根目录", "path": data_root})
    if db_path:
        roots.append({"label": "默认数据库目录", "path": str(Path(db_path).expanduser().parent)})
    return {"roots": roots}


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

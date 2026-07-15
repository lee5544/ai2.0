from __future__ import annotations

import copy
import csv
from pathlib import Path
import subprocess
import sys
import sqlite3
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import yaml
from data_manager.tdms_read import (
    is_compressed_tdms_path,
    is_uncompressed_tdms_path,
    iter_tdms_files,
)
from ml.training.app_api import (
    config_filename,
    data_manager_defaults,
    database_distribution,
    dataset_summary,
    dump_yaml,
    line_reference_overview,
    _model_xlsx_path_from_manifest,
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
from data_manager.line_rules import reload_line_rules

from . import database_jobs, run_manager, store
from .config import (
    CFG_DIR,
    FRONTEND_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    RUN_DIR,
    ensure_dirs,
)


app = FastAPI(title="Forvia Train v2")

LABEL_RULES_PATH = CFG_DIR / "core" / "label_rules.yaml"
LINE_RULES_PATH = CFG_DIR / "core" / "line_rules.yaml"
DEFAULT_LABEL_RULES_PATH = CFG_DIR / "core" / "label_rules.default.yaml"


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()
    store.init_db()
    run_manager.reconcile_active_runs()


def _require_project(project_id: str) -> dict:
    project = store.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except Exception:
        return str(left) == str(right)


def _project_config_path(line_name: str, model_name: str, old_path: str = "") -> Path:
    directory = Path(old_path).expanduser().parent if old_path else CFG_DIR
    return directory / config_filename(line_name, model_name)


def _save_project_config_auto_path(payload: dict, old_path: str = "") -> dict:
    target_path = _project_config_path(payload["line_name"], payload["model_name"], old_path)
    if target_path.exists() and (not old_path or not _same_path(target_path, old_path)):
        raise ValueError(f"配置 YAML 已存在，不能覆盖: {target_path}")
    payload = save_project_config(payload, str(target_path))
    if old_path:
        old_p = Path(old_path).expanduser()
        new_path = Path(str(payload.get("config_path") or "")).resolve()
        if old_p.exists() and old_p.resolve() != new_path:
            try:
                old_p.unlink()
            except OSError:
                pass
    return payload


class ProjectReq(BaseModel):
    name: str = ""
    line_name: str
    model_name: str
    model_type: str = "xgb"
    config: dict = {}
    config_path: str = ""


class OpenProjectYamlReq(BaseModel):
    path: str


class LabelRuleItem(BaseModel):
    key: str = ""
    id: int | None = None
    name: str = ""
    parent: str = ""
    alias: list[str] = Field(default_factory=list)
    note: str = ""


class LabelRulesReq(BaseModel):
    results: list[LabelRuleItem] = Field(default_factory=list)
    reasons: list[LabelRuleItem] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)


class LineRulesReq(BaseModel):
    yaml_text: str


class LineRulesPreviewReq(BaseModel):
    yaml_text: str
    line: str = ""
    filename: str = ""


def _default_line_rules_text() -> str:
    default_path = CFG_DIR / "core" / "line_rules.default.yaml"
    return default_path.read_text(encoding="utf-8")


def _validate_line_rules_text(text: str) -> dict:
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"line_rules.yaml YAML 格式错误: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ValueError("line_rules.yaml 顶层必须是非空对象")
    for line, rule in data.items():
        if not isinstance(line, str) or not line.strip():
            raise ValueError("产线名称不能为空")
        if not isinstance(rule, dict):
            raise ValueError(f"产线 {line} 的规则必须是对象")
        filename = rule.get("filename")
        if filename is not None:
            if not isinstance(filename, dict):
                raise ValueError(f"产线 {line}.filename 必须是对象或 null")
            for key in ("split", "sn_index", "reference_index", "time_index", "time_order"):
                if key not in filename:
                    raise ValueError(f"产线 {line}.filename 缺少 {key}")
            if not isinstance(filename["time_index"], list) or len(filename["time_index"]) != 2:
                raise ValueError(f"产线 {line}.filename.time_index 必须是两个元素的列表")
        channels = rule.get("channels")
        if channels is not None and not isinstance(channels, dict):
            raise ValueError(f"产线 {line}.channels 必须是对象或 null")
        if isinstance(channels, dict) and "conditional" in channels:
            conditions = channels["conditional"]
            if not isinstance(conditions, list) or not conditions:
                raise ValueError(f"产线 {line}.channels.conditional 必须是非空列表")
            for index, condition in enumerate(conditions, start=1):
                if not isinstance(condition, dict) or not isinstance(condition.get("when"), dict):
                    raise ValueError(f"产线 {line}.channels.conditional[{index}] 缺少 when 对象")
    return data


def _line_rules_payload() -> dict:
    try:
        text = LINE_RULES_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        text = _default_line_rules_text()
    data = _validate_line_rules_text(text)
    return {"path": str(LINE_RULES_PATH), "yaml_text": text, "lines": list(data)}


def _read_label_rules() -> dict:
    try:
        data = yaml.safe_load(LABEL_RULES_PATH.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        data = yaml.safe_load(DEFAULT_LABEL_RULES_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("label_rules.yaml 顶层必须是对象")
    return data


def _rules_to_rows(items: dict, *, include_parent: bool) -> list[dict]:
    rows = []
    for key, value in (items or {}).items():
        raw = value if isinstance(value, dict) else {}
        row = {
            "key": str(key),
            "id": raw.get("id"),
            "name": str(raw.get("name") or ""),
            "alias": [str(x) for x in (raw.get("alias") or [])],
            "note": str(raw.get("note") or ""),
        }
        if include_parent:
            row["parent"] = str(raw.get("parent") or "")
        rows.append(row)
    rows.sort(key=lambda x: (999999 if x.get("id") is None else int(x.get("id")), x["key"]))
    return rows


def _slug_key(text: str, prefix: str, used: set[str]) -> str:
    import re

    base = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip()).strip("_").lower()
    if not base:
        base = prefix
    if base[0].isdigit():
        base = f"{prefix}_{base}"
    key = base
    index = 2
    while key in used:
        key = f"{base}_{index}"
        index += 1
    used.add(key)
    return key


def _normalize_rule_rows(rows: list[LabelRuleItem], *, include_parent: bool, key_prefix: str) -> dict:
    used: set[str] = set()
    explicit_ids = [int(row.id) for row in rows if row.id is not None]
    next_id = (max(explicit_ids) + 1) if explicit_ids else 0
    out: dict[str, dict] = {}
    for row in rows:
        name = row.name.strip()
        if not name:
            raise ValueError("标签名称不能为空")
        key = row.key.strip()
        if key:
            if key in used:
                raise ValueError(f"标签 key 重复: {key}")
            used.add(key)
        else:
            key = _slug_key(name, key_prefix, used)
        item_id = int(row.id) if row.id is not None else next_id
        if row.id is None:
            next_id += 1
        item: dict[str, object] = {
            "id": item_id,
            "name": name,
            "alias": [str(x).strip() for x in row.alias if str(x).strip()],
        }
        if include_parent:
            parent = row.parent.strip()
            if not parent:
                raise ValueError(f"原因标签 {name} 缺少 parent")
            item["parent"] = parent
        elif row.note.strip():
            item["note"] = row.note.strip()
        out[key] = item
    return out


def _label_rules_payload() -> dict:
    data = _read_label_rules()
    return {
        "path": str(LABEL_RULES_PATH),
        "results": _rules_to_rows(data.get("results") if isinstance(data.get("results"), dict) else {}, include_parent=False),
        "reasons": _rules_to_rows(data.get("reasons") if isinstance(data.get("reasons"), dict) else {}, include_parent=True),
        "meta": data.get("meta") if isinstance(data.get("meta"), dict) else {},
    }


def _write_label_rules(payload: dict) -> None:
    LABEL_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABEL_RULES_PATH.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _database_paths_from_root(data_root_text: str) -> dict[str, str]:
    if not str(data_root_text or "").strip():
        raise ValueError("请先选择数据库根目录 data_root")
    data_root = Path(str(data_root_text or "")).expanduser()
    metadata = data_root / "metadata"
    return {
        "data_root": str(data_root),
        "label_records_db_path": str(metadata / "label_records.db"),
        "manifest_path": str(metadata / "tdms_manifest.csv"),
        "tdms_root": str(data_root / "factory_raw"),
    }


def _distribution_overview_from_paths(paths: dict[str, str], *, with_models: bool) -> dict:
    distribution = database_distribution(paths["label_records_db_path"], paths["manifest_path"])
    if with_models:
        return {
            "xlsx_path": str(_model_xlsx_path_from_manifest(Path(paths["manifest_path"]))),
            "line_rows": [
                {
                    "line": row.get("line", ""),
                    "models": copy.deepcopy(row.get("models", [])),
                }
                for row in distribution.get("line_rows", [])
            ],
            "total": copy.deepcopy(distribution.get("total", {})),
            "filter_rule": distribution.get("filter_rule", ""),
        }
    line_rows = []
    for row in distribution.get("line_rows", []):
        line_rows.append(
            {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key != "models"
            }
        )
    return {
        "line_rows": line_rows,
        "total": copy.deepcopy(distribution.get("total", {})),
        "filter_rule": distribution.get("filter_rule", ""),
    }


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


@app.get("/api/label-rules")
def api_label_rules():
    try:
        return _label_rules_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/label-rules")
def api_save_label_rules(req: LabelRulesReq):
    try:
        result_keys = {row.key.strip() for row in req.results if row.key.strip()}
        payload = {
            "results": _normalize_rule_rows(req.results, include_parent=False, key_prefix="result"),
            "reasons": _normalize_rule_rows(req.reasons, include_parent=True, key_prefix="reason"),
            "meta": req.meta or {},
        }
        invalid_parents = [
            f"{key}:{value.get('parent')}"
            for key, value in payload["reasons"].items()
            if str(value.get("parent") or "") not in result_keys and str(value.get("parent") or "") not in payload["results"]
        ]
        if invalid_parents:
            raise ValueError("原因标签 parent 不存在: " + ", ".join(invalid_parents))
        _write_label_rules(payload)
        return _label_rules_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/label-rules/restore-default")
def api_restore_label_rules():
    try:
        text = DEFAULT_LABEL_RULES_PATH.read_text(encoding="utf-8")
        payload = yaml.safe_load(text) or {}
        _write_label_rules(payload)
        return _label_rules_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/line-rules")
def api_line_rules():
    try:
        return _line_rules_payload()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/line-rules/validate")
def api_validate_line_rules(req: LineRulesReq):
    try:
        data = _validate_line_rules_text(req.yaml_text)
        return {"ok": True, "lines": list(data), "message": "规则格式正确"}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.put("/api/line-rules")
def api_save_line_rules(req: LineRulesReq):
    try:
        data = _validate_line_rules_text(req.yaml_text)
        LINE_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
        LINE_RULES_PATH.write_text(req.yaml_text.rstrip() + "\n", encoding="utf-8")
        reload_line_rules()
        return {"path": str(LINE_RULES_PATH), "yaml_text": req.yaml_text.rstrip() + "\n", "lines": list(data)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/line-rules/restore-default")
def api_restore_line_rules():
    try:
        text = _default_line_rules_text()
        _validate_line_rules_text(text)
        LINE_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
        LINE_RULES_PATH.write_text(text, encoding="utf-8")
        reload_line_rules()
        default_data = _validate_line_rules_text(text)
        return {"path": str(LINE_RULES_PATH), "yaml_text": text, "lines": list(default_data)}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


EXAMPLE_CFG_DIR = CFG_DIR / "examples"
BUILTIN_EXAMPLES = {
    "epump2_general.yaml": "epump2_general.yaml",
    "epump3_general.yaml": "epump3_general.yaml",
    "epump4_general.yaml": "epump4_general.yaml",
    "etilt1_general.yaml": "etilt1_general.yaml",
}


def _example_path(filename: str) -> Path | None:
    source_name = BUILTIN_EXAMPLES.get(filename)
    if not source_name:
        return None
    path = EXAMPLE_CFG_DIR / filename
    if path.exists():
        return path
    path = PROJECT_ROOT / "cfg" / source_name
    return path if path.exists() else None


def _example_payload(path: Path) -> dict:
    config = parse_yaml(path.read_text(encoding="utf-8"))
    line_name = str(config.get("line_name") or "").strip()
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    model_name = str(model.get("model_name") or path.stem.replace(f"{line_name}_", "", 1)).strip()
    model_type = str(train.get("model_type") or "xgb").strip().lower()
    return normalize_project_payload({
        "line_name": line_name,
        "model_name": model_name,
        "model_type": model_type,
        "config": config,
    })


def _ensure_builtin_projects() -> None:
    existing = {model_id(project["config"]) for project in store.list_projects()}
    for filename in BUILTIN_EXAMPLES:
        path = _example_path(filename)
        if not path:
            continue
        payload = _example_payload(path)
        project_id = model_id(payload["config"])
        if project_id in existing:
            continue
        payload["origin"] = "ui"
        payload["config_path"] = str(path.resolve())
        store.create_project(payload)
        existing.add(project_id)


@app.get("/api/projects")
def api_projects():
    _ensure_builtin_projects()
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


@app.post("/api/projects/open-yaml")
def api_open_project_yaml(req: OpenProjectYamlReq):
    source = Path(req.path).expanduser()
    if source.suffix.lower() not in {".yaml", ".yml"} or not source.is_file():
        raise HTTPException(status_code=400, detail="请选择存在的 YAML 配置文件")
    try:
        payload = _example_payload(source)
        project_id = model_id(payload["config"])
        existing = next((p for p in store.list_projects() if model_id(p["config"]) == project_id), None)
        if existing:
            return {"project": existing, "yaml": dump_yaml(existing["config"])}
        payload["origin"] = "ui"
        target = CFG_DIR / config_filename(payload["line_name"], payload["model_name"])
        if source.resolve() == target.resolve() or target.is_file():
            payload["config_path"] = str((target if target.is_file() else source).resolve())
        else:
            payload = save_project_config(payload)
        project = store.create_project(payload)
        return {"project": project, "yaml": dump_yaml(project["config"])}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"打开模型项目失败: {exc}")


@app.put("/api/projects/{project_id}")
def api_update_project(project_id: str, req: ProjectReq):
    existing = _require_project(project_id)
    try:
        payload = _save_project_config_auto_path(
            normalize_project_payload(req.model_dump()),
            str(existing.get("config_path") or ""),
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
        payload = _save_project_config_auto_path(payload, str(project.get("config_path") or ""))
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
    update_kind: str = ""
    label_source_type: str = "none"
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


class DatabasePrecheckReq(BaseModel):
    action: str = ""
    update_kind: str = ""
    label_source_type: str = "none"
    data_root: str = ""
    source_folder: str = ""
    label_csvs: list[str] = Field(default_factory=list)
    source_label_db: str = ""
    label_records_db_path: str = ""
    tdms_root: str = ""
    storage_root: str = "factory_raw"
    line: str = ""
    file_mode: str = "manual"


def _format_bytes(num_bytes: int) -> str:
    value = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.1f} {units[idx]}"


def _path_inside(child: Path, parent: Path) -> bool:
    try:
        child.expanduser().resolve().relative_to(parent.expanduser().resolve())
        return True
    except Exception:
        return False


def _read_csv_preview_rows(path: Path, limit: int = 5000) -> tuple[list[dict[str, str]], list[str]]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="", errors="replace") as stream:
                reader = csv.DictReader(stream)
                rows = []
                for idx, row in enumerate(reader):
                    if idx >= limit:
                        break
                    rows.append(dict(row))
                return rows, list(reader.fieldnames or [])
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"无法识别 CSV 编码: {path}")


def _scan_source_folder(source_folder: str, *, data_root: Path, storage_root: str) -> dict:
    source_text = str(source_folder or "").strip()
    empty = {
        "path": source_text,
        "exists": False,
        "tdms_count": 0,
        "tdms_zst_count": 0,
        "raw_tdms_count": 0,
        "need_compress_count": 0,
        "total_size_bytes": 0,
        "total_size_text": "0.0 B",
        "line_counts": [],
        "inside_factory_raw": False,
        "recommended_file_mode": "",
    }
    if not source_text:
        return empty
    source = Path(source_text).expanduser()
    if not source.is_dir():
        return empty
    factory_root = data_root / storage_root
    files = sorted(iter_tdms_files(source))
    line_counter: Counter[str] = Counter()
    total_size = 0
    raw_count = 0
    zst_count = 0
    for path in files:
        try:
            total_size += int(path.stat().st_size)
        except Exception:
            pass
        if is_compressed_tdms_path(path):
            zst_count += 1
        if is_uncompressed_tdms_path(path):
            raw_count += 1
        try:
            relative = path.resolve().relative_to(source.resolve())
            line = relative.parts[0] if relative.parts else ""
        except Exception:
            line = ""
        if line:
            line_counter[line] += 1
    inside = _path_inside(source, factory_root)
    return {
        **empty,
        "exists": True,
        "tdms_count": len(files),
        "tdms_zst_count": zst_count,
        "raw_tdms_count": raw_count,
        "need_compress_count": raw_count,
        "total_size_bytes": total_size,
        "total_size_text": _format_bytes(total_size),
        "line_counts": [{"line": line, "count": int(count)} for line, count in line_counter.most_common()],
        "inside_factory_raw": inside,
        "recommended_file_mode": "manual" if inside else "copy",
    }


def _db_label_preview(path: Path) -> dict:
    if not path.is_file():
        return {"exists": False, "label_count": 0, "distribution": []}
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as con:
            label_count = int(con.execute("SELECT COUNT(*) FROM label_events").fetchone()[0])
            distribution = [
                {"label": str(label or "未填写"), "count": int(count)}
                for label, count in con.execute(
                    """
                    SELECT COALESCE(NULLIF(reason_name, ''), NULLIF(reason_key, ''), NULLIF(result_name, ''), '未填写'), COUNT(*)
                    FROM label_events
                    WHERE status='confirmed'
                    GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 20
                    """
                )
            ]
        return {"exists": True, "label_count": label_count, "distribution": distribution}
    except Exception as exc:
        return {"exists": True, "label_count": 0, "distribution": [], "error": str(exc)}


def _csv_label_preview(paths: list[str]) -> dict:
    distribution: Counter[str] = Counter()
    total_rows = 0
    sample_like = 0
    files = []
    for item in paths:
        path = Path(str(item or "").strip()).expanduser()
        if not str(item or "").strip():
            continue
        report = {"path": str(path), "exists": path.is_file(), "rows": 0, "format": "unknown"}
        if path.is_file():
            try:
                rows, headers = _read_csv_preview_rows(path)
                report["rows"] = len(rows)
                header_set = set(headers)
                internal_required = {
                    "view_name", "line", "sn", "sample_id", "result_key", "result_id",
                    "result_name", "reason_key", "reason_id", "reason_name",
                    "reason_confidence", "label_version", "note", "timestamp", "source",
                }
                if internal_required.issubset(header_set):
                    report["format"] = "internal_label_event"
                    for row in rows:
                        label = str(
                            row.get("reason_name")
                            or row.get("reason_key")
                            or row.get("result_name")
                            or row.get("result_key")
                            or "未填写"
                        ).strip()
                        distribution[label] += 1
                    sample_like += len(rows)
                elif {"sn", "sample_id"}.issubset(header_set):
                    report["format"] = "sample_view_like"
                    for row in rows:
                        label = str(
                            row.get("reason_name")
                            or row.get("reason_key")
                            or row.get("result_name")
                            or row.get("result_key")
                            or "未填写"
                        ).strip()
                        distribution[label] += 1
                    sample_like += len(rows)
                else:
                    label_cols = [name for name in headers if name.startswith("label_")]
                    report["format"] = "employee_operator_wide" if label_cols else "wide"
                    for row in rows:
                        for name in label_cols:
                            label = str(row.get(name) or "").strip()
                            if label:
                                distribution[label] += 1
                                sample_like += 1
                total_rows += len(rows)
            except Exception as exc:
                report["error"] = str(exc)
        files.append(report)
    return {
        "exists": any(item["exists"] for item in files),
        "label_count": int(sum(distribution.values()) or sample_like or total_rows),
        "matched_tdms_count": None,
        "unmatched_count": None,
        "distribution": [{"label": key, "count": int(value)} for key, value in distribution.most_common(20)],
        "files": files,
    }


def _database_status(paths: dict[str, str]) -> dict:
    db_path = Path(paths["label_records_db_path"]).expanduser()
    manifest_path = Path(paths["manifest_path"]).expanduser()
    tdms_root = Path(paths["tdms_root"]).expanduser()
    label_rules_ok = False
    try:
        rules = _read_label_rules()
        label_rules_ok = isinstance(rules.get("results"), dict) and isinstance(rules.get("reasons"), dict)
    except Exception:
        label_rules_ok = False
    sampling_total = 0
    sampling_missing = 0
    if db_path.is_file():
        try:
            with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as con:
                sampling_total = int(con.execute("SELECT COUNT(*) FROM samples WHERE is_active=1").fetchone()[0])
                sampling_missing = int(
                    con.execute(
                        "SELECT COUNT(*) FROM samples WHERE is_active=1 AND (sampling_rate IS NULL OR sampling_rate='')"
                    ).fetchone()[0]
                )
        except Exception:
            pass
    sampling_complete = sampling_total > 0 and sampling_missing == 0
    return {
        "db_exists": db_path.is_file(),
        "manifest_exists": manifest_path.is_file(),
        "factory_raw_exists": tdms_root.is_dir(),
        "label_rules_ok": label_rules_ok,
        "sampling_total": sampling_total,
        "sampling_missing": sampling_missing,
        "sampling_complete": sampling_complete,
        "sampling_rate_text": f"{sampling_total - sampling_missing}/{sampling_total}" if sampling_total else "0/0",
    }


def _precheck_database(req: DatabasePrecheckReq) -> dict:
    data_root_text = str(req.data_root or "").strip()
    if not data_root_text:
        raise ValueError("数据库目录 data_root 不能为空")
    data_root = Path(data_root_text).expanduser()
    storage_root = str(req.storage_root or "factory_raw").strip().strip("/") or "factory_raw"
    paths = _database_paths_from_root(str(data_root))
    source_scan = _scan_source_folder(req.source_folder, data_root=data_root, storage_root=storage_root)
    label_source = str(req.label_source_type or "none").strip().lower()
    csv_paths = [str(path).strip() for path in req.label_csvs if str(path).strip()]
    if label_source == "db":
        label_preview = _db_label_preview(Path(req.source_label_db).expanduser())
    elif label_source in {"internal", "forvia"}:
        label_preview = _csv_label_preview(csv_paths)
    else:
        label_preview = {"exists": True, "label_count": 0, "distribution": [], "files": []}
    status = _database_status(paths)
    warnings = []
    errors = []
    action = str(req.action or "").strip().lower()
    update_kind = str(req.update_kind or "").strip().lower()
    if action == "update" and not str(req.source_folder or "").strip():
        repair_scan_root = data_root / storage_root
        line = str(req.line or "").strip()
        if line and (repair_scan_root / line).is_dir():
            repair_scan_root = repair_scan_root / line
        source_scan = _scan_source_folder(
            str(repair_scan_root),
            data_root=data_root,
            storage_root=storage_root,
        )
    if action == "create":
        if not source_scan["exists"]:
            errors.append("源文件夹不存在")
        elif not source_scan["tdms_count"]:
            errors.append("源文件夹中没有 TDMS / TDMS.ZST")
        if status["db_exists"] or status["manifest_exists"]:
            errors.append("新建数据库目标 metadata 已存在，请改用更新数据库或选择新的 data_root")
        if source_scan["exists"] and not source_scan["inside_factory_raw"]:
            warnings.append("源文件夹不在 data_root/factory_raw 下，执行前需要选择复制或移动")
        if source_scan["need_compress_count"]:
            warnings.append(f"检测到 {source_scan['need_compress_count']} 个 .tdms，注册流程会升级为 .tdms.zst")
    elif action == "update":
        if not status["db_exists"]:
            errors.append("更新数据库需要已有 label_records.db")
        if not status["factory_raw_exists"]:
            errors.append("TDMS 根目录不存在")
        if source_scan["need_compress_count"]:
            warnings.append(f"更新数据库会先压缩 {source_scan['need_compress_count']} 个 .tdms 为 .tdms.zst")
    if label_source in {"internal", "forvia"} and not csv_paths:
        errors.append("选择内部表或 Forvia 表作为标签来源时，需要添加标签 CSV")
    if label_source == "db" and not str(req.source_label_db or "").strip():
        errors.append("选择 DB 作为标签来源时，需要选择标签 DB")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "paths": paths,
        "source_scan": source_scan,
        "label_preview": label_preview,
        "status": status,
    }


@app.post("/api/database/precheck")
def api_database_precheck(req: DatabasePrecheckReq):
    try:
        return _precheck_database(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/database/paths")
def api_database_paths(data_root: str):
    try:
        return _database_paths_from_root(data_root)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/database/line-reference-overview")
def api_database_line_reference_overview(data_root: str):
    try:
        return _distribution_overview_from_paths(_database_paths_from_root(data_root), with_models=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/database/xlsx-model-overview")
def api_database_xlsx_model_overview(data_root: str):
    try:
        return _distribution_overview_from_paths(_database_paths_from_root(data_root), with_models=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/database-actions")
def api_global_database_action(req: DatabaseActionReq):
    try:
        return {"job": database_jobs.start_global(req.model_dump())}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
    allowed = [RUN_DIR.resolve(), RESULTS_DIR.resolve()]
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


@app.get("/console")
def console_index():
    return FileResponse(PROJECT_ROOT / "web" / "forvia_console_preview.html")


@app.get("/forvia_console_modules.js")
def console_modules():
    return FileResponse(PROJECT_ROOT / "web" / "forvia_console_modules.js", headers={"Cache-Control": "no-store"})


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

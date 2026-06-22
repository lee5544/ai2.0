from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_MANAGER_CONFIG_PATH = PROJECT_ROOT / "cfg" / "core" / "data_manager.yaml"

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        raise RuntimeError(
            f"PyYAML is required to load config file: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return cfg


def _pick_cfg_value(cfg: Dict[str, Any], key: str, default: str) -> str:
    value = cfg.get(key)
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _expand_path_template(raw: str, *, data_root: str, tdms_storage_root: str) -> str:
    return (
        str(raw)
        .replace("{data_root}", str(data_root))
        .replace("{tdms_storage_root}", str(tdms_storage_root))
        .replace("{project_root}", str(PROJECT_ROOT))
    )


def load_data_manager_config() -> Dict[str, str]:
    cfg = _read_yaml(DATA_MANAGER_CONFIG_PATH)

    data_root = _pick_cfg_value(cfg, "data_root", "./data_root")
    tdms_storage_root = _pick_cfg_value(cfg, "tdms_storage_root", "factory_raw").strip().strip("/")
    if not tdms_storage_root:
        tdms_storage_root = "factory_raw"

    label_records_db_default = str(
        Path(data_root).expanduser() / "metadata" / "label_records.db"
    )
    label_root_default = str(Path(data_root).expanduser() / tdms_storage_root)
    label_root_raw = (
        _pick_cfg_value(cfg, "label_root_path", label_root_default)
    )
    label_records_db_raw = _pick_cfg_value(
        cfg,
        "label_records_db_path",
        str(cfg.get("sample_records_db_path") or label_records_db_default),
    )
    legacy_path = Path(label_records_db_raw).expanduser()
    if legacy_path.name == "sample_records.db":
        label_records_db_raw = str(legacy_path.with_name("label_records.db"))

    label_root_path = str(
        Path(
            _expand_path_template(
                label_root_raw,
                data_root=data_root,
                tdms_storage_root=tdms_storage_root,
            )
        ).expanduser()
    )
    label_records_db_path = str(
        Path(
            _expand_path_template(
                label_records_db_raw,
                data_root=data_root,
                tdms_storage_root=tdms_storage_root,
            )
        ).expanduser()
    )

    result = {
        "config_path": str(DATA_MANAGER_CONFIG_PATH),
        "data_root": data_root,
        "tdms_storage_root": tdms_storage_root,
        "label_root_path": label_root_path,
        "label_records_db_path": label_records_db_path,
    }
    for key in ("line_rules_path", "label_rules_path", "model_registry_path"):
        raw = _pick_cfg_value(cfg, key, "")
        if not raw:
            raise ValueError(f"data_manager.yaml 缺少配置: {key}")
        result[key] = str(Path(_expand_path_template(
            raw, data_root=data_root, tdms_storage_root=tdms_storage_root
        )).expanduser())
    return result


CONFIG = load_data_manager_config()
LINE_RULES_PATH = Path(CONFIG["line_rules_path"])
LABEL_RULES_PATH = Path(CONFIG["label_rules_path"])
MODEL_REGISTRY_PATH = Path(CONFIG["model_registry_path"])

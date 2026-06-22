from __future__ import annotations

import csv
from pathlib import Path

from .app_api import model_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def _read_csv(path: Path, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            with path.open("r", encoding=enc, errors="replace", newline="") as f:
                return [dict(row) for _, row in zip(range(limit), csv.DictReader(f))]
        except Exception:
            continue
    return []


def find_model_dir(config: dict) -> Path:
    configured = Path(str(config.get("results_path") or RESULTS_DIR)).expanduser()
    root = configured if configured.is_absolute() else PROJECT_ROOT / configured
    current = root / model_id(config)
    if current.exists():
        return current
    train = config.get("train") if isinstance(config.get("train"), dict) else {}
    model_type = str(train.get("model_type") or "").strip()
    legacy = root / f"{model_id(config)}_{model_type}"
    return legacy if model_type and legacy.exists() else current


def _chart(title: str, path: Path, kind: str) -> dict:
    return {"title": title, "path": str(path), "kind": kind}


def collect_results(config: dict) -> dict:
    model_dir = find_model_dir(config)
    eval_dir = model_dir / "eval"
    summary = _read_csv(eval_dir / "summary.csv")
    stats = _read_csv(eval_dir / "summary_stats.csv")
    best = summary[0] if summary else {}
    best_seed = str(best.get("seed") or "")
    featured_charts: list[dict] = []
    other_charts: list[dict] = []
    if best_seed:
        base = eval_dir / "best_model_eval" / f"seed_{best_seed}"
        featured = [
            ("训练集混淆矩阵", base / "train.png", "train_confusion"),
            ("测试集混淆矩阵", base / "test.png", "test_confusion"),
            ("验证集混淆矩阵", base / "validation.png", "validation_confusion"),
            ("Loss 曲线", base / "training_loss.png", "training_loss"),
        ]
        others = [
            ("测试集 ROC", base / "test_ROC.png", "test_roc"),
            ("验证集 ROC", base / "validation_ROC.png", "validation_roc"),
            ("训练集 ROC", base / "train_ROC.png", "train_roc"),
            ("数据集指标对比", base / "diagnostics" / "split_metrics.png", "split_metrics"),
            ("训练特征重要性", base / "importance" / "train_feature_importance_gain.png", "feature_importance"),
            ("最佳 Seed 混淆矩阵与 ROC", base / "seed_confusion_roc.png", "seed_confusion_roc"),
        ]
        featured_charts.extend(_chart(title, path, kind) for title, path, kind in featured if path.exists())
        other_charts.extend(_chart(title, path, kind) for title, path, kind in others if path.exists())
    summary_charts = [
        ("多 Seed 混淆矩阵与 ROC 汇总", eval_dir / "summary_confusion_roc.png", "summary_confusion_roc"),
        ("多 Seed 最终报告", eval_dir / "final_seed_report_summary.png", "final_seed_report"),
    ]
    other_charts.extend(_chart(title, path, kind) for title, path, kind in summary_charts if path.exists())
    image_paths = [chart["path"] for chart in [*featured_charts, *other_charts]]
    return {
        "model_id": model_id(config),
        "model_dir": str(model_dir),
        "exists": model_dir.exists(),
        "model_exists": (model_dir / "model.pkl").exists(),
        "predict_config_exists": (model_dir / "predict_config.yaml").exists(),
        "summary": summary,
        "stats": stats,
        "best": best,
        "images": image_paths,
        "featured_charts": featured_charts,
        "other_charts": other_charts,
        "misclassified_path": str(eval_dir / "validation_test_misclassified_sample_view.csv"),
    }

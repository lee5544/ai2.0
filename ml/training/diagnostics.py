from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def _split_metrics_row(
    *,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
    model_classes: np.ndarray,
    ok_label: int = 0,
) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba_arr = None if y_proba is None else np.asarray(y_proba)

    max_prob_mean = float(np.max(y_proba_arr, axis=1).mean()) if y_proba_arr is not None and y_proba_arr.size else np.nan
    margin_mean = np.nan
    if y_proba_arr is not None and y_proba_arr.ndim == 2 and y_proba_arr.shape[1] >= 2:
        sorted_scores = np.sort(y_proba_arr, axis=1)
        margin_mean = float((sorted_scores[:, -1] - sorted_scores[:, -2]).mean())

    logloss_value = np.nan
    if y_proba_arr is not None and y_proba_arr.size:
        try:
            logloss_value = float(log_loss(y_true, y_proba_arr, labels=model_classes))
        except Exception:
            logloss_value = np.nan

    quality_metrics = compute_quality_gate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        ok_label=ok_label,
    )

    return {
        "split": split_name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        **quality_metrics,
        "logloss": logloss_value,
        "max_prob_mean": max_prob_mean,
        "margin_mean": margin_mean,
    }


def compute_quality_gate_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ok_label: int = 0,
) -> dict[str, Any]:
    """
    Collapse multiclass predictions into the production quality gate:
    OK vs NOK. The critical miss is actual NOK predicted as OK.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ok_label = int(ok_label)

    true_ok = y_true == ok_label
    true_nok = ~true_ok
    pred_ok = y_pred == ok_label
    pred_nok = ~pred_ok

    ok_total = int(true_ok.sum())
    nok_total = int(true_nok.sum())
    pred_ok_total = int(pred_ok.sum())
    pred_nok_total = int(pred_nok.sum())

    ok_as_ok = int((true_ok & pred_ok).sum())
    ok_as_nok = int((true_ok & pred_nok).sum())
    nok_as_nok = int((true_nok & pred_nok).sum())
    nok_as_ok = int((true_nok & pred_ok).sum())

    def _safe_ratio(num: int, den: int) -> float:
        return float(num / den) if den > 0 else np.nan

    return {
        "ok_label": ok_label,
        "ok_samples": ok_total,
        "nok_samples": nok_total,
        "pred_ok_samples": pred_ok_total,
        "pred_nok_samples": pred_nok_total,
        "ok_recall": _safe_ratio(ok_as_ok, ok_total),
        "nok_recall": _safe_ratio(nok_as_nok, nok_total),
        "nok_precision": _safe_ratio(nok_as_nok, pred_nok_total),
        "false_ok_count": nok_as_ok,
        "false_ok_rate": _safe_ratio(nok_as_ok, nok_total),
        "false_reject_count": ok_as_nok,
        "false_reject_rate": _safe_ratio(ok_as_nok, ok_total),
    }


def _report_rows(
    *,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mlabel_mtype_dict: dict[int, str] | None,
) -> list[dict[str, Any]]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows: list[dict[str, Any]] = []
    for label_name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        try:
            class_id = int(label_name)
        except Exception:
            class_id = None
        display_name = (
            mlabel_mtype_dict.get(class_id, label_name)
            if class_id is not None and isinstance(mlabel_mtype_dict, dict)
            else label_name
        )
        rows.append(
            {
                "split": split_name,
                "label": label_name,
                "label_name": display_name,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "f1_score": float(metrics.get("f1-score", 0.0)),
                "support": float(metrics.get("support", 0.0)),
            }
        )
    return rows


def _format_table_frame(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    for col in formatted.columns:
        if col == "split":
            continue
        if col in {
            "samples",
            "ok_label",
            "ok_samples",
            "nok_samples",
            "pred_ok_samples",
            "pred_nok_samples",
            "false_ok_count",
            "false_reject_count",
        }:
            formatted[col] = formatted[col].astype(int).astype(str)
            continue
        formatted[col] = formatted[col].map(lambda x: "-" if pd.isna(x) else f"{float(x):.4f}")
    return formatted


def _save_metrics_overview_png(metrics_df: pd.DataFrame, output_path: Path) -> None:
    display_df = _format_table_frame(metrics_df)
    fig, ax = plt.subplots(figsize=(18, 2.8))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=list(display_df.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.6)
    ax.set_title("Training diagnostics by split", fontsize=14, pad=12)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_training_diagnostics(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    y_pred_train: np.ndarray,
    y_proba_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_pred_val: np.ndarray,
    y_proba_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    y_proba_test: np.ndarray,
    model_classes: np.ndarray,
    mlabel_mtype_dict: dict[int, str] | None,
    output_dir: str | Path,
    seed: int,
    ok_label: int = 0,
) -> dict[str, str]:
    del X_train, X_val, X_test

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = [
        _split_metrics_row(
            split_name="train",
            y_true=y_train,
            y_pred=y_pred_train,
            y_proba=y_proba_train,
            model_classes=model_classes,
            ok_label=ok_label,
        ),
        _split_metrics_row(
            split_name="val",
            y_true=y_val,
            y_pred=y_pred_val,
            y_proba=y_proba_val,
            model_classes=model_classes,
            ok_label=ok_label,
        ),
        _split_metrics_row(
            split_name="test",
            y_true=y_test,
            y_pred=y_pred_test,
            y_proba=y_proba_test,
            model_classes=model_classes,
            ok_label=ok_label,
        ),
    ]
    metrics_df = pd.DataFrame(split_rows)
    metrics_csv_path = out_dir / "split_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    report_rows: list[dict[str, Any]] = []
    report_rows.extend(_report_rows(split_name="train", y_true=y_train, y_pred=y_pred_train, mlabel_mtype_dict=mlabel_mtype_dict))
    report_rows.extend(_report_rows(split_name="val", y_true=y_val, y_pred=y_pred_val, mlabel_mtype_dict=mlabel_mtype_dict))
    report_rows.extend(_report_rows(split_name="test", y_true=y_test, y_pred=y_pred_test, mlabel_mtype_dict=mlabel_mtype_dict))
    report_df = pd.DataFrame(report_rows)
    report_csv_path = out_dir / "split_classification_report.csv"
    report_df.to_csv(report_csv_path, index=False, encoding="utf-8-sig")

    summary_png_path = out_dir / "split_metrics.png"
    _save_metrics_overview_png(metrics_df, summary_png_path)

    print(
        f"[INFO] 训练诊断已保存: seed={seed} | "
        f"metrics={metrics_csv_path} | report={report_csv_path} | png={summary_png_path}"
    )
    return {
        "metrics_csv": str(metrics_csv_path),
        "report_csv": str(report_csv_path),
        "summary_png": str(summary_png_path),
    }

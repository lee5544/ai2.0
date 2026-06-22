"""
本地特征测试脚本 —— 在你的机器上运行，对 v2/v3/v4 做对照。

用途
----
1. 读取真实的 tdms / tdms.zst（支持 `/Volumes/18555440521/fault/data_root`
   以及仓库内 `漏判的震颤件/part1/*.tdms`）。
2. 同时跑 v2 / v3 / v4 特征提取，落盘成 CSV。
3. 对关键新增特征（crest_factor、mel_band_wide_std、mel_shock_high_avg_kurt、
   env_mod_peak_freq...）做分组统计，按 reason 分组画分布箱线图。

环境 & 依赖
    conda activate fault
    # 如需补齐依赖：
    # pip install nptdms scipy pywavelets numpy pandas matplotlib zstandard

用法示例
--------
# 只跑仓库里 14 条漏判样本（最快自检）
conda activate fault
python AI-2.0/test_v4_features.py --preset missed

# 跑外接硬盘里的 epump2 full set，按 reason 筛选
python AI-2.0/test_v4_features.py \
    --data-root /Volumes/18555440521/fault/data_root \
    --line epump2 \
    --reasons 震颤 秒表 正常 \
    --limit 60 \
    --out-dir ./results/v4_feature_test

# 用 label_records.db 驱动（只跑已标注样本，更精确）
python AI-2.0/test_v4_features.py \
    --label-records-db 漏判的震颤件/part1/metadata/label_records.db \
    --data-root /Volumes/18555440521/fault/data_root \
    --line epump2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# 保证能 import 到项目模块（python 从任何位置起都能跑）
PROJECT_ROOT = Path(__file__).resolve().parent  # .../AI-2.0
REPO_ROOT = PROJECT_ROOT.parent                  # .../2025-forvia-异常检测
sys.path.insert(0, str(PROJECT_ROOT))

from data_manager.tdms_read import read_tdms  # noqa: E402
from data_manager.label_database import load_label_dataframe  # noqa: E402

# v2 / v3 / v4 extractors（按需 lazy import）
from ml.features import (                      # noqa: E402
    DEFAULT_FEATURE_VERSION,
    get_feature_extractor,
    normalize_feature_version,
)


# ====================================================================
# 1) 样本集合的获取
# ====================================================================
def _iter_from_sample_records(label_records_db: Path, data_root: Path) -> list[dict]:
    """从 label_records.db 读取 (sample_id, reason, expected_tdms_path)。"""
    df = load_label_dataframe(label_records_db)
    # 每台电机一条（sample_id 形如 'OG6A0331_up'），去重后只保留 sn
    df["sn"] = df["sample_id"].str.split("_").str[0]
    df = df.drop_duplicates(subset=["sn"], keep="first")

    items: list[dict] = []
    for row in df.itertuples(index=False):
        sn = row.sn
        reason = row.reason_name
        matches = list(data_root.rglob(f"*_{sn}_*.tdms")) + list(
            data_root.rglob(f"*_{sn}_*.tdms.zst")
        )
        if not matches:
            continue
        items.append({"tdms_path": str(matches[0]), "sn": sn, "reason": reason})
    return items


def _iter_from_directory(
    directory: Path,
    line: str,
    reasons: Iterable[str] | None,
    limit: int,
) -> list[dict]:
    """从一个目录里扫描 tdms / tdms.zst 文件，reason 留空。"""
    all_files = sorted(
        list(directory.rglob("*.tdms")) + list(directory.rglob("*.tdms.zst"))
    )
    items: list[dict] = []
    for f in all_files:
        sn = f.stem.split("_")[1] if "_" in f.stem else f.stem
        items.append({"tdms_path": str(f), "sn": sn, "reason": ""})
        if limit and len(items) >= limit:
            break
    if reasons:
        # 没有 reason 信息 —— 打印警告
        print(
            "[warn] --reasons 指定了过滤条件，但是仅扫描目录时无 reason 信息，"
            "已忽略；请改用 --label-records-db。"
        )
    return items


# ====================================================================
# 2) 核心：对单条 tdms 跑三个版本
# ====================================================================
def _run_one(tdms_path: Path, line: str) -> dict:
    """读取一条 tdms，按 up/down 通道分别跑 v2/v3/v4，合并为一行。"""
    info = read_tdms(str(tdms_path), line=line)
    sr_up = int(info.get("up_sampling_rate") or info.get("sampling_rate") or 0)
    sr_down = int(info.get("down_sampling_rate") or info.get("sampling_rate") or 0)
    up = info.get("up_data")
    down = info.get("down_data")

    row: dict = {
        "tdms_path": str(tdms_path),
        "sn": info.get("sn"),
        "reference": info.get("reference"),
        "up_len": int(0 if up is None else up.shape[0]),
        "down_len": int(0 if down is None else down.shape[0]),
        "sr_up": sr_up,
        "sr_down": sr_down,
    }

    for version in ("v2", "v3", "v4"):
        extractor = get_feature_extractor(version)
        for channel_name, signal, sr in (
            ("up", up, sr_up),
            ("down", down, sr_down),
        ):
            if signal is None or sr <= 0:
                continue
            t0 = time.perf_counter()
            try:
                feats = extractor(signal, sr)
            except Exception as err:  # noqa: BLE001
                row[f"{version}_{channel_name}__ERROR"] = f"{type(err).__name__}: {err}"
                continue
            dt_ms = (time.perf_counter() - t0) * 1000.0
            row[f"{version}_{channel_name}__elapsed_ms"] = round(dt_ms, 2)
            if isinstance(feats, dict):
                for k, v in feats.items():
                    row[f"{version}_{channel_name}__{k}"] = v
            elif isinstance(feats, (list, tuple, np.ndarray)):
                for i, v in enumerate(np.asarray(feats).ravel()):
                    row[f"{version}_{channel_name}__f{i:03d}"] = float(v)
            else:
                row[f"{version}_{channel_name}__raw"] = repr(feats)
    return row


# ====================================================================
# 3) Cohen's d 摘要
# ====================================================================
_KEY_NEW_FEATURES = [
    "crest_factor",
    "peak_to_rms_q95",
    "zero_cross_rate",
    "spectral_flatness",
    "frame_energy_entropy",
    "env_mod_peak_freq",
    "env_mod_peak_ratio",
    "env_mod_top3_ratio_sum",
    "mel_band_wide_std",
    "mel_band_hml_balance",
    "mel_shock_high_ratio",
    "mel_shock_high_avg_kurt",
    "mel_shock_med_avg_kurt",
    "mel_shock_high_avg_prom",
    "mel_shock_med_avg_prom",
    "mel_shock_high_interval_cv",
    "mel_shock_high_peak_count_per_sec",
    "dwt_2_shock_interval_cv",
    "dwt_3_shock_interval_cv",
]


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sa, sb = np.std(a, ddof=1), np.std(b, ddof=1)
    if sa == 0 and sb == 0:
        return 0.0
    pooled = np.sqrt((sa**2 + sb**2) / 2.0)
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def summarize_discriminators(df: pd.DataFrame, reason_col: str = "reason") -> pd.DataFrame:
    """对 v4 新增特征做 震颤 vs 正常、秒表 vs 正常、秒表 vs 震颤 的 Cohen's d。"""
    rows: list[dict] = []
    reasons_unique = df[reason_col].dropna().unique().tolist()
    if not any(r in reasons_unique for r in ("震颤", "秒表", "正常")):
        print("[warn] 数据里没有同时包含 震颤/秒表/正常，跳过 Cohen's d 摘要")
        return pd.DataFrame()
    for chan in ("up", "down"):
        for feat in _KEY_NEW_FEATURES:
            col = f"v4_{chan}__{feat}"
            if col not in df.columns:
                continue
            g_normal = df.loc[df[reason_col] == "正常", col].to_numpy(dtype=float)
            g_tremor = df.loc[df[reason_col] == "震颤", col].to_numpy(dtype=float)
            g_stop   = df.loc[df[reason_col] == "秒表", col].to_numpy(dtype=float)
            rows.append(
                {
                    "channel": chan,
                    "feature": feat,
                    "mean_normal": float(np.nanmean(g_normal)) if g_normal.size else np.nan,
                    "mean_tremor": float(np.nanmean(g_tremor)) if g_tremor.size else np.nan,
                    "mean_stop":   float(np.nanmean(g_stop))   if g_stop.size   else np.nan,
                    "d_tremor_vs_normal": _cohens_d(g_tremor, g_normal),
                    "d_stop_vs_normal":   _cohens_d(g_stop,   g_normal),
                    "d_stop_vs_tremor":   _cohens_d(g_stop,   g_tremor),
                }
            )
    return pd.DataFrame(rows).sort_values("d_stop_vs_tremor", key=np.abs, ascending=False)


# ====================================================================
# 4) 主流程
# ====================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=None,
                   help="顶层数据目录；传 --label-records-db 时用它定位 tdms 文件")
    p.add_argument("--label-records-db", type=Path, default=None,
                   help="label_records.db，推荐方式（带 reason 信息）")
    p.add_argument("--preset", choices=["missed"], default=None,
                   help="快捷方式：missed 跑仓库里 漏判的震颤件/part1/*.tdms")
    p.add_argument("--line", default="epump2")
    p.add_argument("--reasons", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out-dir", type=Path,
                   default=REPO_ROOT / "results_v4_feature_test")
    p.add_argument("--feature-versions", nargs="+",
                   default=["v2", "v3", "v4"],
                   help="默认跑全部三个版本；排障时可只跑 v4")
    args = p.parse_args()

    # 1) 确定样本集合
    if args.preset == "missed":
        missed_dir = REPO_ROOT / "漏判的震颤件" / "part1"
        items = _iter_from_directory(missed_dir, args.line, None, args.limit)
        # 补一下 reason
        label_records_db = missed_dir / "metadata" / "label_records.db"
        if label_records_db.exists():
            df_lh = load_label_dataframe(label_records_db)
            sn2reason = dict(
                df_lh.drop_duplicates("sample_id")
                .assign(sn=lambda d: d["sample_id"].str.split("_").str[0])
                .groupby("sn")["reason_name"]
                .first()
                .items()
            )
            for it in items:
                it["reason"] = sn2reason.get(it["sn"], "")
    elif args.label_records_db and args.data_root:
        items = _iter_from_sample_records(args.label_records_db, args.data_root)
    elif args.data_root:
        items = _iter_from_directory(
            args.data_root, args.line, args.reasons, args.limit
        )
    else:
        p.error("必须提供 --preset missed / --label-records-db / --data-root 其中之一")
        return

    if args.reasons and any(it.get("reason") for it in items):
        items = [it for it in items if it["reason"] in set(args.reasons)]
    if args.limit:
        items = items[: args.limit]

    print(f"[info] 共 {len(items)} 条待处理样本")

    # 2) 跑特征
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    errors: list[dict] = []
    for idx, item in enumerate(items, 1):
        try:
            row = _run_one(Path(item["tdms_path"]), args.line)
            row.update(
                {"sn_hint": item["sn"], "reason": item.get("reason", "")}
            )
            rows.append(row)
            # 实时打印一行关键特征
            keys_to_show = [
                "v4_up__crest_factor",
                "v4_up__mel_band_wide_std",
                "v4_up__mel_shock_high_avg_kurt",
                "v4_up__env_mod_peak_freq",
            ]
            quick = {k.split("__")[-1]: row.get(k) for k in keys_to_show}
            print(
                f"[{idx}/{len(items)}] {item['sn']}  reason={item.get('reason') or '?'}  "
                + "  ".join(
                    f"{k}={v:.3g}" if isinstance(v, (int, float)) and v is not None
                    else f"{k}=NA"
                    for k, v in quick.items()
                )
            )
        except Exception as err:  # noqa: BLE001
            tb = traceback.format_exc()
            errors.append({"tdms_path": item["tdms_path"], "error": str(err), "traceback": tb})
            print(f"[err] {item['tdms_path']}: {err}")

    if not rows:
        print("[fatal] 没有任何样本成功跑完特征，检查路径/依赖")
        sys.exit(1)

    df = pd.DataFrame(rows)
    feat_path = args.out_dir / "features_v2v3v4.csv"
    df.to_csv(feat_path, index=False)
    print(f"[info] 特征表 -> {feat_path}  shape={df.shape}")

    # 3) Cohen's d 摘要
    summary = summarize_discriminators(df)
    if not summary.empty:
        summary_path = args.out_dir / "discriminator_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[info] Cohen's d 摘要 -> {summary_path}")
        with pd.option_context("display.float_format", "{:+.2f}".format):
            print(summary.head(20).to_string(index=False))

    # 4) 可视化（可选；matplotlib 不可用就跳过）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_features = [
            "v4_up__mel_band_wide_std",
            "v4_up__mel_shock_high_avg_kurt",
            "v4_up__crest_factor",
            "v4_up__env_mod_peak_freq",
        ]
        plot_features = [f for f in plot_features if f in df.columns]
        if plot_features and df["reason"].astype(bool).any():
            fig, axes = plt.subplots(
                1, len(plot_features), figsize=(4 * len(plot_features), 4),
                constrained_layout=True,
            )
            if len(plot_features) == 1:
                axes = [axes]
            grouped = df[df["reason"].astype(bool)].groupby("reason")
            for ax, feat in zip(axes, plot_features):
                data = [g[feat].dropna().to_numpy() for _, g in grouped]
                labels = [name for name, _ in grouped]
                ax.boxplot(data, tick_labels=labels, showmeans=True)
                ax.set_title(feat.split("__")[-1])
                ax.set_yscale("symlog", linthresh=1e-3)
                ax.grid(True, alpha=0.3)
            fig.suptitle("v4 key features by reason")
            fig_path = args.out_dir / "distribution_boxplot.png"
            fig.savefig(fig_path, dpi=130)
            plt.close(fig)
            print(f"[info] 箱线图 -> {fig_path}")
    except Exception as err:  # noqa: BLE001
        print(f"[info] 跳过绘图: {err}")

    if errors:
        err_path = args.out_dir / "errors.json"
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] {len(errors)} 条样本出错 -> {err_path}")

    print("[done]")


if __name__ == "__main__":
    main()

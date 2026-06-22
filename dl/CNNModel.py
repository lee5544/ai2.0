"""DL CNN 分类器模型类。

镜像 ml/XGBModel 的对外接口，使 dl/train.run_training 能以与 ML
完全一致的方式调度：
  - __init__(config_path)：解析配置、解析目录与超参、暴露切分所需属性
  - train_and_evaluate(split)：构建张量→训练→评估→存档（支持 seed_runs）
  - train_and_cross_validate / train_and_evaluate_grid：占位，后续补全
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl import train_core as core
from dl.train import DLSplit
from dl.models import build_dl_model


class CNNModel:
    """基于窗口 mel 特征的 CNN 分类器（默认 cnn1d 架构）。"""

    def __init__(self, config_path: str) -> None:
        self.config_path = str(config_path)
        cfg = core._read_yaml(self.config_path)
        self.cfg = cfg

        dl_cfg = core._resolve_dl_cfg(cfg)
        self.dl_cfg = dl_cfg
        self.dl_train_cfg = dl_cfg.get("train") if isinstance(dl_cfg.get("train"), dict) else {}
        self.dl_model_cfg = dl_cfg.get("model") if isinstance(dl_cfg.get("model"), dict) else {}
        self.global_train_cfg = cfg.get("train") if isinstance(cfg.get("train"), dict) else {}
        # 与 ML 接口对齐：dataset_split/registry 通过 train_cfg 读取部分参数
        self.train_cfg = self.dl_train_cfg

        # 架构
        self.model_arch = core._resolve_model_arch(cfg, None)
        self.model_tag = core._model_tag_from_arch(self.model_arch)

        # 目录
        self.results_root = core._resolve_results_root(cfg, None, self.model_arch)
        self.dataset_dir = core._resolve_dataset_dir(cfg, None, self.model_arch)
        self.train_output_dir = self.results_root / "dl_train"
        self.train_output_dir.mkdir(parents=True, exist_ok=True)
        # main_train._copy_config_to_results / 标签复核钩子使用
        self.results_path = self.train_output_dir

        # 特征 schema 与批次文件
        self.schema = core._load_feature_schema(self.dataset_dir)
        extract_cfg = dl_cfg.get("extract") if isinstance(dl_cfg.get("extract"), dict) else {}
        self.output_file_prefix = (
            core._to_text(self.schema.get("output_file_prefix"))
            or core._to_text(extract_cfg.get("output_file_prefix"))
            or core.DEFAULT_OUTPUT_FILE_PREFIX
        )
        self.batch_format = core._normalize_batch_format(
            self.schema.get("output_format")
            or extract_cfg.get("output_format")
            or dl_cfg.get("output_format")
            or core.DEFAULT_BATCH_FORMAT
        )
        self.dataset_files = core._resolve_dataset_files(
            dataset_dir=self.dataset_dir,
            output_file_prefix=self.output_file_prefix,
            batch_format=self.batch_format,
        )

        # 超参
        self.epochs = int(self.dl_train_cfg.get("epochs") or 300)
        self.batch_size = int(self.dl_train_cfg.get("batch_size") or 256)
        self.learning_rate = float(self.dl_train_cfg.get("learning_rate") or 1e-3)
        self.weight_decay = float(self.dl_train_cfg.get("weight_decay") or 1e-4)
        self.num_workers = int(self.dl_train_cfg.get("num_workers", 0))
        self.use_amp = bool(self.dl_train_cfg.get("use_amp", True))
        self.random_state = int(
            self.dl_train_cfg.get("random_state")
            or self.global_train_cfg.get("random_state")
            or 42
        )
        self.split_random_state = self.random_state
        self.test_size = float(self.dl_train_cfg.get("test_size") or 0.1)
        self.val_size = float(self.dl_train_cfg.get("val_size") or 0.125)
        self.group_split_trials = int(
            self.dl_train_cfg.get("group_split_trials")
            or self.global_train_cfg.get("group_split_trials")
            or 32
        )
        self.group_sample_trials = int(
            self.dl_train_cfg.get("group_sample_trials")
            or self.global_train_cfg.get("group_sample_trials")
            or 24
        )
        self.seed_runs = max(1, int(self.dl_train_cfg.get("seed_runs") or 1))
        self.split_strategy = core._to_text(self.global_train_cfg.get("split_strategy")) or "reference_in"
        self.device = core._resolve_device(
            self.dl_train_cfg.get("device") or dl_cfg.get("device")
        )

        if self.epochs <= 0:
            raise ValueError(f"epochs 必须 > 0，当前: {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size 必须 > 0，当前: {self.batch_size}")

        # DL 暂不生成 ML 风格标签复核清单，跳过 main_train 的默认钩子
        self._label_audit_generated = True

    # ------------------------------------------------------------------
    # 标签复核钩子（占位，避免 main_train 调用 ML 默认实现）
    # ------------------------------------------------------------------
    def _generate_label_audit_outputs(self) -> None:
        print("[INFO] DL 流程暂未生成标签复核清单（后续里程碑补全）。")

    # ------------------------------------------------------------------
    # 训练入口
    # ------------------------------------------------------------------
    def train_and_evaluate(self, split: DLSplit) -> Dict[str, Any]:
        wall_t0 = time.perf_counter()
        selected_font = core._configure_matplotlib_for_chinese()

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        channel_names = split.channel_names
        columns_by_channel = split.columns_by_channel
        seq_len = int(split.seq_len)
        class_name_by_id = split.class_name_by_id

        x_train = core._build_feature_tensor(
            split.train_df,
            channel_names=channel_names,
            columns_by_channel=columns_by_channel,
            compact_feature_column=split.compact_feature_column,
        )
        x_val = core._build_feature_tensor(
            split.val_df,
            channel_names=channel_names,
            columns_by_channel=columns_by_channel,
            compact_feature_column=split.compact_feature_column,
        )
        x_test = core._build_feature_tensor(
            split.test_df,
            channel_names=channel_names,
            columns_by_channel=columns_by_channel,
            compact_feature_column=split.compact_feature_column,
        )
        y_train = split.train_df["label"].to_numpy(dtype=np.int64)
        y_val = split.val_df["label"].to_numpy(dtype=np.int64)
        y_test = split.test_df["label"].to_numpy(dtype=np.int64)
        num_classes = int(max(y_train.max(), y_val.max(), y_test.max()) + 1)
        if num_classes < 2:
            raise RuntimeError(f"类别数不足 2，无法训练 DL 模型。当前 num_classes={num_classes}")

        train_dataset = core.WindowFeatureDataset(x_train, y_train)
        val_dataset = core.WindowFeatureDataset(x_val, y_val)
        test_dataset = core.WindowFeatureDataset(x_test, y_test)
        _nw = max(0, self.num_workers)
        _pin = self.device.type == "cuda"
        _persistent = _nw > 0
        _prefetch = 2 if _nw > 0 else None
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=_nw, pin_memory=_pin,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=_nw, pin_memory=_pin,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size * 2, shuffle=False,
            num_workers=_nw, pin_memory=_pin,
            persistent_workers=_persistent, prefetch_factor=_prefetch,
        )

        class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        class_weights = np.ones(num_classes, dtype=np.float32)
        valid_mask = class_counts > 0
        class_weights[valid_mask] = float(len(y_train)) / (float(num_classes) * class_counts[valid_mask])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=self.device))

        print(f"特征目录: {self.dataset_dir}")
        print(f"训练输出目录: {self.train_output_dir}")
        print(f"数据文件数: {len(self.dataset_files)}")
        print(f"device: {self.device}")
        print(f"model_arch: {self.model_arch} ({self.model_tag})")
        if selected_font:
            print(f"matplotlib 中文字体: {selected_font}")
        else:
            print("[WARN] 未找到常见中文字体，图中的中文可能显示为方框。")
        print(
            f"epochs={self.epochs}, batch_size={self.batch_size}, lr={self.learning_rate}, "
            f"weight_decay={self.weight_decay}, seed_runs={self.seed_runs}"
        )

        # ---------------- 多 seed 复跑，取验证集 macro_f1 最优 ----------------
        seeds = [self.random_state + i for i in range(self.seed_runs)]
        best_run: Dict[str, Any] | None = None
        seed_overview_rows: List[Dict[str, Any]] = []

        for run_idx, seed in enumerate(seeds, start=1):
            print("")
            print(f"===== seed run {run_idx}/{self.seed_runs} (seed={seed}) =====")
            run_result = self._train_single_seed(
                seed=seed,
                num_classes=num_classes,
                channel_names=channel_names,
                seq_len=seq_len,
                criterion=criterion,
                train_loader=train_loader,
                val_loader=val_loader,
                class_name_by_id=class_name_by_id,
                live_plot=(self.seed_runs == 1),
            )
            seed_overview_rows.append(
                {
                    "seed": int(seed),
                    "best_epoch": int(run_result["best_epoch"]),
                    "val_macro_f1": float(run_result["best_score"]),
                }
            )
            if best_run is None or run_result["best_score"] > best_run["best_score"] + 1e-8:
                best_run = run_result

        assert best_run is not None
        best_seed = int(best_run["seed"])
        best_epoch = int(best_run["best_epoch"])
        best_score = float(best_run["best_score"])
        history_rows = best_run["history_rows"]
        resolved_model_config = best_run["resolved_model_config"]

        # 用最优模型重建并评估
        model = best_run["model"]
        model.load_state_dict(best_run["best_state"]["model_state"])
        print("")
        print(f"使用最佳模型评估: seed={best_seed}, epoch={best_epoch}, val_macro_f1={best_score:.4f}")
        train_eval = core._run_epoch(model=model, loader=train_loader, device=self.device, criterion=criterion, optimizer=None, desc="best train eval", num_classes=num_classes)
        val_eval = core._run_epoch(model=model, loader=val_loader, device=self.device, criterion=criterion, optimizer=None, desc="best valid eval", num_classes=num_classes)
        test_eval = core._run_epoch(model=model, loader=test_loader, device=self.device, criterion=criterion, optimizer=None, desc="best test eval", num_classes=num_classes)

        self._save_outputs(
            split=split,
            model=model,
            history_rows=history_rows,
            class_name_by_id=class_name_by_id,
            channel_names=channel_names,
            columns_by_channel=columns_by_channel,
            seq_len=seq_len,
            num_classes=num_classes,
            resolved_model_config=resolved_model_config,
            best_epoch=best_epoch,
            train_eval=train_eval,
            val_eval=val_eval,
            test_eval=test_eval,
            seed_overview_rows=seed_overview_rows,
            best_seed=best_seed,
        )

        total_sec = float(time.perf_counter() - wall_t0)
        print("")
        print(f"DL 训练完成: model_arch={self.model_arch}, best_seed={best_seed}, best_epoch={best_epoch}")
        print(f"train_output_dir: {self.train_output_dir}")
        print(
            f"test metrics: accuracy={test_eval['accuracy']:.4f}, "
            f"macro_precision={test_eval['macro_precision']:.4f}, "
            f"macro_recall={test_eval['macro_recall']:.4f}, "
            f"macro_f1={test_eval['macro_f1']:.4f}"
        )
        print(f"总耗时: {total_sec:.3f}s")
        return {"test": test_eval, "validation": val_eval, "train": train_eval, "best_seed": best_seed}

    # ------------------------------------------------------------------
    def _train_single_seed(
        self,
        *,
        seed: int,
        num_classes: int,
        channel_names: List[str],
        seq_len: int,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_name_by_id: Dict[int, str],
        live_plot: bool,
    ) -> Dict[str, Any]:
        core._set_random_seed(int(seed))

        model_arch, model, resolved_model_config = build_dl_model(
            arch=self.model_arch,
            in_channels=len(channel_names),
            sequence_length=seq_len,
            num_classes=num_classes,
            model_cfg=self.dl_model_cfg,
            train_cfg=self.dl_train_cfg,
        )
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scaler = (
            torch.cuda.amp.GradScaler()
            if self.use_amp and self.device.type == "cuda"
            else None
        )
        if scaler is not None:
            print(f"[INFO] AMP (混合精度) 已启用")

        live_history_plotter = None
        if live_plot:
            live_history_plotter = core._LiveHistoryPlotter(out_path=self.train_output_dir / "training_curve.png")

        best_state: Dict[str, Any] | None = None
        best_epoch = -1
        best_score = float("-inf")
        best_val_loss = float("inf")
        history_rows: List[Dict[str, Any]] = []

        for epoch in range(1, self.epochs + 1):
            print("")
            print(f"[seed {seed}] Epoch {epoch}/{self.epochs}")
            train_metrics = core._run_epoch(model=model, loader=train_loader, device=self.device, criterion=criterion, optimizer=optimizer, desc=f"train epoch {epoch}", num_classes=num_classes, scaler=scaler)
            val_metrics = core._run_epoch(model=model, loader=val_loader, device=self.device, criterion=criterion, optimizer=None, desc=f"valid epoch {epoch}", num_classes=num_classes)

            history_row = {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": float(val_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "train_macro_f1": float(train_metrics["macro_f1"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
            core._append_per_class_metric_history(history_row=history_row, prefix="train", metrics=train_metrics, num_classes=num_classes, metric_key="accuracy", column_suffix="class_acc")
            core._append_per_class_metric_history(history_row=history_row, prefix="val", metrics=val_metrics, num_classes=num_classes, metric_key="accuracy", column_suffix="class_acc")
            core._append_per_class_metric_history(history_row=history_row, prefix="train", metrics=train_metrics, num_classes=num_classes, metric_key="recall", column_suffix="class_recall")
            core._append_per_class_metric_history(history_row=history_row, prefix="val", metrics=val_metrics, num_classes=num_classes, metric_key="recall", column_suffix="class_recall")
            history_rows.append(history_row)
            print(
                f"epoch={epoch} | "
                f"train_loss={train_metrics['loss']:.4f}, train_f1={train_metrics['macro_f1']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['macro_f1']:.4f}"
            )

            improved = False
            if float(val_metrics["macro_f1"]) > best_score + 1e-8:
                improved = True
            elif abs(float(val_metrics["macro_f1"]) - best_score) <= 1e-8 and float(val_metrics["loss"]) < best_val_loss:
                improved = True
            if improved:
                best_score = float(val_metrics["macro_f1"])
                best_val_loss = float(val_metrics["loss"])
                best_epoch = int(epoch)
                best_state = {"model_state": model.state_dict(), "epoch": best_epoch}

            if live_history_plotter is not None:
                live_history_plotter.update(
                    pd.DataFrame(history_rows),
                    best_epoch=best_epoch if best_epoch > 0 else None,
                    best_score=best_score if np.isfinite(best_score) else None,
                )

        if live_history_plotter is not None:
            live_history_plotter.close()
        if best_state is None:
            raise RuntimeError("训练未产生有效模型状态。")

        return {
            "seed": int(seed),
            "model": model,
            "best_state": best_state,
            "best_epoch": int(best_epoch),
            "best_score": float(best_score),
            "history_rows": history_rows,
            "resolved_model_config": resolved_model_config,
            "model_arch": model_arch,
        }

    # ------------------------------------------------------------------
    def _save_outputs(
        self,
        *,
        split: DLSplit,
        model: nn.Module,
        history_rows: List[Dict[str, Any]],
        class_name_by_id: Dict[int, str],
        channel_names: List[str],
        columns_by_channel: Dict[str, List[str]],
        seq_len: int,
        num_classes: int,
        resolved_model_config: Dict[str, Any],
        best_epoch: int,
        train_eval: Dict[str, Any],
        val_eval: Dict[str, Any],
        test_eval: Dict[str, Any],
        seed_overview_rows: List[Dict[str, Any]],
        best_seed: int,
    ) -> None:
        out_dir = self.train_output_dir
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
        core._save_history_plot(history_df, out_dir / "training_curve.png")
        core._save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_accuracy_curve.png", column_suffix="class_acc", ylabel="Accuracy")
        core._save_per_class_metric_plot(history_df=history_df, class_name_by_id=class_name_by_id, out_path=out_dir / "per_class_recall_curve.png", column_suffix="class_recall", ylabel="Recall")

        if len(seed_overview_rows) > 1:
            pd.DataFrame(seed_overview_rows).to_csv(out_dir / "seed_overview.csv", index=False, encoding="utf-8-sig")

        model_config_payload = {"model_arch": self.model_arch, "model_tag": self.model_tag, **resolved_model_config}

        checkpoint = {
            "model_state": model.state_dict(),
            "model_arch": self.model_arch,
            "model_config": model_config_payload,
            "channel_names": channel_names,
            "feature_columns": columns_by_channel,
            "sequence_length": int(seq_len),
            "num_classes": int(num_classes),
            "mlabel_mtype_dict": {int(k): str(v) for k, v in split.mlabel_mtype_dict.items()},
            "class_names": class_name_by_id,
            "mlabel_to_reason_ids": {int(k): [int(x) for x in v] for k, v in split.mlabel_to_reason_ids.items()},
            "device": str(self.device),
            "best_epoch": int(best_epoch),
            "best_seed": int(best_seed),
        }
        torch.save(checkpoint, out_dir / "best_model.pt")

        shutil.copy2(Path(self.config_path).expanduser(), out_dir / "train_config.yaml")
        schema_path = self.dataset_dir / core.SCHEMA_FILENAME
        if schema_path.exists():
            shutil.copy2(schema_path, out_dir / core.SCHEMA_FILENAME)

        split_summary = {
            "train_rows": int(len(split.train_df)),
            "val_rows": int(len(split.val_df)),
            "test_rows": int(len(split.test_df)),
            "train_label_counts": {int(k): int(v) for k, v in split.train_df["label"].value_counts().sort_index().items()},
            "val_label_counts": {int(k): int(v) for k, v in split.val_df["label"].value_counts().sort_index().items()},
            "test_label_counts": {int(k): int(v) for k, v in split.test_df["label"].value_counts().sort_index().items()},
        }
        with (out_dir / "split_summary.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(split_summary, f, allow_unicode=True, sort_keys=False)

        runtime_meta = {
            "model_arch": self.model_arch,
            "model_config": model_config_payload,
            "mlabel_mtype_dict": {int(k): str(v) for k, v in split.mlabel_mtype_dict.items()},
            "class_names": class_name_by_id,
            "label_to_mlabel_map": {int(k): int(v) for k, v in split.label_to_mlabel_map.items()},
            "label_sample_dict": {int(k): int(v) for k, v in split.label_sample_dict.items()},
            "mlabel_to_reason_ids": {int(k): [int(x) for x in v] for k, v in split.mlabel_to_reason_ids.items()},
        }
        with (out_dir / "label_runtime.json").open("w", encoding="utf-8") as f:
            json.dump(runtime_meta, f, ensure_ascii=False, indent=2)

        metrics_payload = {
            "device": str(self.device),
            "best_epoch": int(best_epoch),
            "best_seed": int(best_seed),
            "model_arch": self.model_arch,
            "model_config": model_config_payload,
            "class_names": class_name_by_id,
            "feature_channels": list(channel_names),
            "feature_sequence_length": int(seq_len),
            "train": train_eval,
            "validation": val_eval,
            "test": test_eval,
        }
        with (out_dir / "metrics.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(metrics_payload, f, allow_unicode=True, sort_keys=False)

    # ------------------------------------------------------------------
    def train_and_cross_validate(self, cv_data: Any) -> None:  # pragma: no cover
        raise NotImplementedError(
            "DL CNN 暂未实现 cross 模式，请使用 train_mode=normal（后续里程碑补全）。"
        )

    def train_and_evaluate_grid(self, split_data: Any, grid_cv_data: Any) -> None:  # pragma: no cover
        raise NotImplementedError(
            "DL CNN 暂未实现 grid 模式，请使用 train_mode=normal（后续里程碑补全）。"
        )

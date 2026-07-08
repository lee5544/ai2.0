# dl 模块逻辑

深度学习异常检测流水线。整体分**两个阶段**：先「制作数据集」，再「训练」。
两个阶段各有独立入口，通过磁盘上的 `sample_view.csv` 和 mel 特征批次文件衔接。

```
阶段一  制作数据集 (dl/main_dataset.py)
        cfg ──▶ 样本筛选 ──▶ 标签筛选 ──▶ sample_view.csv ──▶ 特征提取 ──▶ mel_spec_batch_*.pkl
阶段二  训练       (dl/main_train.py)
        mel_spec_batch_*.pkl ──▶ 加载 ──▶ 合并标签+按sn划分 ──▶ 训练 ──▶ 模型+曲线+指标落盘
```

---

## 目录职责（四层 + 结果层）

| 目录/文件 | 职责 | 关键内容 |
|---|---|---|
| `dataset/` | 制作数据集（样本/标签筛选 + 特征生成） | sample_filter / label_filter / build |
| `training/` | 数据集加载 + 划分 + 结果绘图 | loader.py（加载/布局）、split.py（标签runtime/划分）、results.py（绘图） |
| `features/` | 特征提取算法（统一预处理 preprocess） | extract_mel / extract_pcen / extract_raw（按 `dl.feature_type` 分发） |
| `models/` | 网络模型 + 公共接口 | base_model（基类+绘图接口）+ cnn1d/cnn2d/lstm/tcn/resnet |
| `train.py` | 总训练流程（过程式，无编排器类） | train_from_config：加载+划分 → 训练 → 落盘 |

依赖方向（单向无环，且整个 dl **不依赖 ml**）：

```
train.py ──▶ training.split ──▶ training.loader ──▶ (data_manager)
   │  └────▶ training.results ──▶ models.base_model
   └───────▶ models（build_dl_model）
dataset.build ──▶ dataset.sample_filter ──▶ dataset.label_filter
              └─▶ dataset.label_filter
              └─▶ features（提特征）
```

---

## 阶段一：制作数据集

入口 `dl/main_dataset.py` → `dataset/build.py:main()`，两步（`--step all|sample-view|extract`）：

**Step 1 生成 sample_view.csv**（`build.generate`）
1. `sample_filter.filter_samples(cfg)` — 按 cfg 的 `data.line/folders/reference/key_sns` 从
   `tdms_manifest.csv` 筛 manifest，再与 `label_records.db` 里 active 的 sample 按 `line+sn`
   inner join，得到候选样本（只筛样本，不定标签 y）。
2. `label_filter.filter_sample_view_dataframe(候选, 标签库, cfg)` — 为每个样本定训练标签 y：
   - 有 expert 标签 → 取最新 expert
   - ≥2 个员工标签且一致 → 取最新
   - ≥2 个不一致 / 单个 / 无 → 不产生 y（丢弃）
   - 再按 `train.label_mapping` 的每类目标数对 sample_view 下采样
3. 输出标准列 `sample_view.csv` 到 `results/{line}_{model}/dl_dataset_csv/`。

**Step 2 提取特征**（`build._run_extract` → `features.run`）
- 读 sample_view.csv 里每个样本对应的 tdms 信号，按 `dl.feature_type` 调对应提取器：
  STFT 分帧（n_fft=256, hop=128）→ mel 滤波（n_mels=13）→ log → **固定到 max_frames=1024**
  （不足补零、超出截断）。
- 每个样本得到固定 `[13, 1024]` 的 log-mel，连同 y 一起存为 `dl_mel_spec_batch_*.pkl`。

> 制作数据集**只在 dataset 中完成**，train 不负责制作。

---

## 阶段二：训练

入口 `dl/main_train.py` → `train.py:main()` → **`train_from_config(config_path)`**（主流程函数，无编排器类）：

1. **`build_train_config`**：解析 cfg → `TrainConfig`（纯数据：超参 + 路径 + mel 批次文件 + schema）。
2. **加载 + 划分数据集**（调 `training/split.py:prepare_train_val_test_for_dl(tc)`）：
   - `_load_feature_dataframe` 读所有 mel 批次拼成一个 DataFrame
   - `build_label_runtime` 按 label_rules/cfg 算出「原始标签→合并类别 mlabel、每类目标数、类名」
   - `_load_and_split_dataset` **按 sn 分组**切分 train/val/test（相同 sn 不跨集），
     多次试切挑分布最均衡的方案；切完只对训练集按类别目标数重采样
   - `_resolve_feature_layout` 还原通道数和序列长度，打包成 `DLSplit`
   - **训练侧定长**：`build_dataloaders` 按全体样本「时间轴长度的 95 分位」定 `target_length`，
     每条特征沿时间轴裁剪(取前段)/补零到 target_length（mel/pcen 本就定长 → no-op；raw 不定长 → 生效）
3. **训练**（`run_training(tc, split)`，教科书式分层）：
   - `build_dataloaders` 把 split 变成三个 DataLoader；`build_criterion/optimizer/scheduler` 备齐组件
   - 多 seed：每个 `fit_one_run` 跑「准备 → epoch 循环 → 最优」
     - 每 epoch：`train_one_epoch`（5 步：zero_grad→forward→loss→backward→step）+ `evaluate`（`@torch.no_grad`）
     - 可选 scheduler、early stopping；`training.results.LiveHistoryPlotter` 实时刷新曲线
     - 按 val macro-f1 存最优 checkpoint
4. **落盘结果**（`_save_outputs(tc, ...)` + 混淆矩阵）：
   - `history.csv`、checkpoint、`save_history_plot`、`save_per_class_metric_plot`
   - **三张混淆矩阵（计数版）**：`confusion_matrix_train.png` / `confusion_matrix_test.png` /
     `confusion_matrix_val_plus_rest.png`（val + 采样剩余 = 所有未进 train/test 的样本）

---

## models 接口

- `BaseModel(nn.Module)`（`models/base_model.py`）：统一公共接口
  - 构造时集中校验 `in_channels>0 / num_classes>1`
  - `forward` 统一校验 `[B, C, T]` 输入后转交子类 `_forward_impl`
  - 元数据：`arch_name` / `hyperparameters()` / `num_parameters()` / `describe()`
  - 自带绘图：`record_epoch` / `plot_loss` / `plot_history`（底层与 results 同源）
- 具体模型继承 BaseModel：`cnn1d` `cnn2d` `lstm` `tcn` `resnet`
- `build_dl_model(arch, ...)`（`models/registry.py`）按 arch 字符串分发构造

---

## 关键配置（cfg YAML）

- `dl.model_type` — 架构（cnn1d/lstm/tcn…）
- `dl.feature_type` — `mel`(默认) | `pcen` | `raw`（决定调哪个提取器）
- `dl.extract.mel` — `n_fft / hop_length / n_mels / max_frames / fmin / fmax`
- `dl.train` — `epochs / batch_size / learning_rate / weight_decay / test_size(0.1) / val_size(0.125) /
  group_split_trials(32) / group_sample_trials(24) / random_state(42) / seed_runs /
  scheduler(none|cosine|step|plateau) / early_stopping{enable,patience} / length_percentile(0.95) /
  loss{type:ce|focal, label_smoothing, focal_gamma} / class_weight`
- `train.label_mapping` — 类别合并规则与每类目标样本数

---

## 两条命令

```bash
# 阶段一：制作数据集（sample_view + mel 特征）
python dl/main_dataset.py --config cfg/xxx.yaml

# 阶段二：训练
python dl/main_train.py --config cfg/xxx.yaml --model-type cnn1d
```

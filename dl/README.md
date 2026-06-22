# DL 模块说明

基于 PyTorch 的深度学习训练流程，支持 CNN1D / CNN2D / LSTM / ResNet2D / TCN 等架构。

## 目录结构

```
AI-2.0/
└── dl/
    ├── main_train.py      # 训练入口
    ├── main_dataset.py    # 数据集生成入口
    ├── train.py           # 模型注册表 + 训练调度 + CLI
    ├── train_core.py      # 训练循环核心（loss / 优化器 / 评估）
    ├── CNNModel.py        # CNN 模型类（实现 train_and_evaluate 接口）
    ├── features/
    │   ├── __init__.py
    │   └── extract_mel.py # mel 频谱特征提取
    ├── dataset/
    │   ├── __init__.py
    │   └── build.py       # 样本筛选 + 标签筛选 + generate + 数据集 CLI
    └── models/            # PyTorch 网络架构（cnn1d / cnn2d / lstm / resnet2d / tcn）
```

---

## 完整流程

### Step 1 — 生成 sample_view.csv

从 `label_records.db` + `tdms_manifest.csv` 按筛选规则生成训练样本清单：

```bash
python dl/main_dataset.py --config cfg/<your_config>.yaml --step sample-view
```

输出：`results/{line}_{model}_{model_type}/dl_dataset_csv/sample_view.csv`

---

### Step 2 — 提取 mel 窗口特征

从 TDMS 文件提取 mel 频谱特征，保存为 pickle 批次文件：

```bash
python dl/main_dataset.py --config cfg/<your_config>.yaml --step extract
```

常用可选参数：

```bash
python dl/main_dataset.py \
  --config cfg/<your_config>.yaml \
  --step extract \
  --num-workers 8 \        # 并行 worker 数（默认 min(8, cpu_count)）
  --batch-size 2000        # 每个批次文件的样本数（默认 2000）
```

输出：`results/{line}_{model}_{model_type}/dl_dataset_csv/dl_mel_spec_batch_*.pkl`

---

### Step 1 + 2 合并执行

```bash
python dl/main_dataset.py --config cfg/<your_config>.yaml
```

---

### Step 3 — 训练模型

```bash
python dl/main_train.py --config cfg/<your_config>.yaml
```

可选参数：

```bash
python dl/main_train.py \
  --config cfg/<your_config>.yaml \
  --model-type cnn          # 模型类型：cnn / cnn1d / cnn2d / lstm / resnet / tcn（默认读 YAML dl.model_type）
```

训练完成后输出到 `results/{line}_{model}_{model_type}/`：

```
results/{line}_{model}_{model_type}/
├── model.pt               # 最佳模型权重
├── train_config.yaml      # 本次训练配置副本
└── eval/                  # 评估结果（混淆矩阵、指标等）
```

---

## 单独运行特征提取（直接调用 dl.features）

```bash
python -m dl.features \
  --config cfg/<your_config>.yaml \
  --output-feature-folder results/<model_id>/dl_dataset_csv \
  --n-fft 256 \
  --hop-length 128 \
  --n-mels 13 \
  --max-frames 1024 \
  --num-workers 8
```

---

## YAML 配置关键字段

```yaml
line_name: epump2
results_path: ./results

model:
  model_name: general_20260607

dl:
  model_type: cnn          # cnn / cnn1d / cnn2d / lstm / resnet / tcn
  mel:
    n_fft: 256
    hop_length: 128
    n_mels: 13
    max_frames: 1024
    fmin: 0
    fmax: null             # null = sr/2
  train:
    train_mode: normal     # normal / cross / grid
    epochs: 50
    batch_size: 64
    learning_rate: 0.001
```

### TCN 专用超参（model_type: tcn 时生效）

```yaml
dl:
  model_type: tcn
  train:
    # TCN block 通道数列表（长度 = block 数，膨胀系数 2^i 自动递增）
    tcn_num_channels: [64, 64, 128, 128]   # 4 个 block，感受野 = 1 + 2*(k-1)*(2^4-1)
    # 或仅指定 block 数 + 基础通道（等宽配置）
    # tcn_num_blocks: 6
    # tcn_base_channels: 64
    kernel_size: 3          # 膨胀卷积核大小（建议 3 或 5）
    dropout: 0.2
    hidden_dim: 128
    tcn_causal: false       # true = 因果模式（实时推理）；false = 非因果（分类任务）
```

感受野速查（kernel_size=3）：

| num_blocks | 膨胀系数序列            | 感受野（步数）|
|-----------|------------------------|------------|
| 4         | 1, 2, 4, 8             | 61         |
| 6         | 1, 2, 4, 8, 16, 32     | 253        |
| 8         | 1, 2, …, 128           | 1021       |

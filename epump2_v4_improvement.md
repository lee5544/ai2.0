# epump2 模型与特征改进建议（v4）

面向 `cfg/epump2_general_20260407.yaml`，针对 **震颤 / 秒表** 两类高价值 NOK
的识别，对特征工程与模型训练环节做一次系统性的审视。

本文分为四部分：
1. 现状诊断（数据分布、现有特征、现有模型）
2. v4 新增特征（经验证据 + 代码位置）
3. 模型层面的改进建议（保持 XGBoost 的前提下）
4. 后续可选方向（数据/标签/两段式结构）

---

## 1. 现状诊断

### 1.1 标签分布高度不均衡
基于 `results/epump2_general_20260407_xgb/dataset_csv/features_batch_*.csv`
（训练/验证总计 **7196 条**）得到的样本数：

| 类别 | 数量 | 占比 |
|------|------|------|
| 正常 | 5243 | 72.9% |
| 秒表 | 692  | 9.6% |
| 摩擦 | ~450 | 6.2% |
| 震颤 | 452  | 6.3% |
| 传感器错误 | ~150 | 2.1% |
| 干扰/边界/其它 | <100 each | — |
| 马达 | 9    | 0.13% |

结论：
- 秒表 / 震颤 样本数虽然比马达大一个数量级，但相对正常仍是 **~1:10**，
  且它们与正常的差异主要体现在**局部冲击 / 周期性**，很容易被全局统计的主轴
  所淹没。
- `XGBModel.py` 当前 `fit()` **没有设置** `sample_weight` / `scale_pos_weight`，
  也没有启用 `class_weight`（参见 "现有模型" 分析）。

### 1.2 现有特征（v2 / v3）的盲区
通过对 177 维 v2 特征做 Cohen's d 分析：

| 方向 | 秒表 vs 正常 最强特征 | |d| | 震颤 vs 正常 最强特征 | |d| |
|------|---------------------|----|---------------------|----|
| 能量/幅值 | mel_band_high_db_mean | 1.18 | mel_band_mid_db_std | 2.31 |
| 冲击数 | mel_shock_high_count | 1.27 | mel_shock_med_count | 1.64 |
| 形态   | mel_shock_high_kurt_mean | 0.60 | mel_shock_med_prom_mean | 0.85 |

共同盲区：
1. **"每个冲击有多尖"** 在 v2 里只用了一阶矩（mean/max），缺少对尖锐度的
   稳健估计（avg_kurt = Σkurt / count、avg_prom）。秒表是离散的机械咔哒，
   峰值 kurt 通常很大；震颤是连续抖动，数量多但每个冲击 kurt 偏小 —— v2
   把二者搅在一起。
2. **周期性**。秒表 ~1Hz，震颤连续 > 10Hz，v2 里只有"总数 / 总宽度"，
   没有"间隔的 CV 系数 / 秒均冲击数"。v3 已经在 **DWT 维度**加了这些
   （interval_mean/std/cv、peak_count_per_sec），但是 v4 早期为了减少维度把
   这一批剪掉了，秒表/震颤的可区分度被牺牲。
3. **全局冲击性**。Crest factor、spectral flatness、帧能量熵这类"一眼可辨"
   的标准信号质量指标，之前一直缺失。
4. **调制谱**。秒表 & 某些震颤会在**包络**上呈现清晰的周期主峰，经典轴承
   故障诊断路数。v2/v3 完全没有用到 Hilbert 包络。

### 1.3 现有模型
`XGBModel.py` 关键点：
- `XGBClassifier(n_estimators=150, max_depth=8, eta=0.1, subsample=0.8, colsample_bytree=0.8)`
- 训练时**不传** `sample_weight`，`XGBClassifier` 默认 `scale_pos_weight=1`。
  对于多分类，这意味着每个样本权重相同，少数类被主类别淹没。
- `predict_with_threshold_rule`（`prediction_logic.py`）的阈值逻辑**只针对
  OK 类兜底**：当最大类概率 < 阈值、且 `1-ok_prob ≤ ok_threshold` 时强制输出
  正常。这个兜底是"从 NOK 翻回 OK"的单向操作，没有"从 OK 翻到 NOK"的对称
  机制 —— 当模型对某条漏判的震颤件只输出 0.55 正常 / 0.45 震颤 时，
  规则层不会给出任何修正。
- 最新 `summary_stats.csv`：`test_acc=0.942`、`test_nok_recall=0.872`、
  `test_false_ok_rate=0.128` —— 漏判率偏高，符合"少数类召回不足"的典型症状。

---

## 2. v4 新增特征（已写入 `ml/features/extract_features_v4.py`）

> 所有新增特征都加入了 `SELECTED_FEATURE_NAMES`，训练时会自动被 XGBoost 使用。
> v4 保留了 v2/v3 所有原 89 维精选特征，新增 ~30 维 —— 总维度约 120。

### 2.1 per-peak 形态平均
```
mel_shock_{low,med,high,mada,mada2}_avg_kurt = Σkurt / count
mel_shock_{low,med,high,mada,mada2}_avg_prom = Σprom / count
```
经验证据：
- **mel_shock_high_avg_kurt** 区分秒表 vs 震颤 **|d|=2.04**
  （秒表虽然冲击少但每个都尖，震颤虽然多但每个都钝）
- mel_shock_med_avg_prom 区分震颤 vs 正常 **|d|=1.44**
- mel_shock_high_avg_prom 区分震颤 vs 正常 **|d|=1.14**

实现：`_summarize_mel_peak_features()` 同时输出 `kurtosis_sum`
和 `prominences_sum`（保留以便派生），并直接计算 `avg_kurt` / `avg_prom`。

### 2.2 带宽与能量分布
```
mel_band_wide_std   = mel_band_mid_db_std + mel_band_high_db_std
mel_band_hml_balance = high_db_ratio - low_db_ratio
mel_shock_high_ratio = count_high / (count_low + count_med + count_high + ε)
```
**mel_band_wide_std** 区分 震颤 vs 正常 **|d|=2.74**，是本次分析发现的
最强单维震颤指标（持续性宽带能量波动）。

### 2.3 全局冲击性（新）
```
crest_factor          = |x|_max / rms           # 经典冲击指数
peak_to_rms_q95       = |x|_{q95} / rms         # 稳健版 crest，抗离群
zero_cross_rate       = 过零率
spectral_flatness     = exp(mean(log S)) / mean(S)       # Wiener 熵
frame_energy_entropy  = -Σ pᵢ log pᵢ, pᵢ = eᵢ / Σeⱼ     # 帧能量熵
```
实现：`_global_impulsivity_features()`，一次 STFT 覆盖 spectral_flatness
和 frame_energy_entropy，其余时域计算。

### 2.4 包络调制谱（新）
```
env_mod_peak_freq     = envelope 0.5–500Hz 区间主峰频率
env_mod_peak_ratio    = peak_power / total_power
env_mod_top3_ratio_sum = (top3 peaks sum) / total_power
```
实现：`_envelope_modulation_features()` 用 Hilbert 变换取高通（>=20Hz）信号
的瞬时幅度包络 → 降采到 1kHz → FFT → 在 0.5–500Hz 区间寻主峰。
- 秒表期望 `env_mod_peak_freq ≈ 1Hz`
- 震颤期望 `env_mod_peak_freq ∈ [20Hz, 200Hz]`，且 `peak_ratio` 较高
- 正常曲线平坦，`peak_ratio` 低

### 2.5 周期性（回归 v3 特征）
```
mel_shock_{high,med}_interval_cv
mel_shock_{high,med}_peak_count_per_sec
dwt_{2,3,4}_shock_interval_cv
dwt_{2,3,4}_shock_peak_count_per_sec
```
v3 已经实现的 `_effective_sr_for_wavedec_coeff` 路径保持不变，v4 继续调用；
**mel-level** 的间隔 CV 是新加的 —— 秒表的 mel_high 冲击间隔 CV 很低
（极规律），震颤的间隔 CV 很高（随机密集）。

---

## 3. 模型训练层面的改进建议

### 3.1 处理类不平衡（**优先级最高**）
`XGBModel.py` 中 fit 时增加 `sample_weight`：
```python
# 建议用 "median_count / class_count" 做 sqrt 平滑
counts = np.bincount(y_train)
w = np.sqrt(np.median(counts) / counts[y_train])
model.fit(X_train, y_train, sample_weight=w, ...)
```
或者通过 YAML 暴露：
```yaml
train:
  class_weight:
    mode: sqrt_balanced   # 或 balanced / manual
    manual:
      震颤: 3.0
      秒表: 2.0
      马达: 5.0
```
预期收益：`nok_recall` 上升 3–5 pt，代价是 OK→NOK 的假报略升，可以通过
调 `ok_threshold` 弥补。

### 3.2 threshold 规则双向化
当前 `predict_with_threshold_rule` 只允许 NOK→OK 的兜底。建议增加：
```
if max(p_nok_classes) > nok_boost_threshold (e.g. 0.35)
   and p_ok < 0.55:
       y_pred = argmax(p_nok_classes)
```
即当任何一个 NOK 类概率超过阈值（且 OK 不强势）时，强制选 NOK。阈值
由 `results/{exp}/eval/*` 里按每类做 ROC 后选出。

### 3.3 震颤类先验：per-class threshold
建议在 `eval/` 阶段新增一份 `per_class_threshold.json`，记录每个类的
F1-optimal threshold，推理时按类使用。当前单一 `ok_threshold` 无法
对震颤（连续弱信号）和秒表（稀疏强信号）同时最优。

### 3.4 噪声标签处理
`漏判的震颤件/part1/metadata/label_history.csv` 里专家已经纠偏 14 条；
从全数据看估计有 3–5% 的标签噪声（正常 ↔ 边界 ↔ 干扰）。
建议两步走：
- **短期**：训练时启用 `train.ignore_low_confidence`，把
  `reason_confidence < 0.5` 的样本权重降为 0.3。
- **中期**：训练后用 OOF 预测回扫一轮，挑出 `p_true_label < 0.2` 的样本
  交人工复核，落盘到 `label_history.csv`；下一轮训练采用纠正后的标签
  （这正是 `data_manager/label.py` 已经支持的 expert-override 流程）。

### 3.5 两段式结构（可选）
```
Stage A:  OK vs NOK（二分类，样本均衡）
Stage B:  NOK 分 6 类（小模型，震颤/秒表/摩擦/马达/传感器错误/哒哒）
```
Stage A 用更强的正则（比如加 Focal Loss）压漏判；Stage B 可以完全不看
正常样本、不担心类别不平衡。两个模型用同一套 v4 特征即可。

---

## 4. 后续可选方向

### 4.1 特征层面
- **震颤专用频谱斜率**：对 200Hz–2kHz 段做线性回归，斜率 + 残差标准差。
- **Envelope Spectrum Kurtosis (SK)**：定位最具冲击性的频带。
- **Cepstrum C1/C2**：提取周期结构的倒谱峰，秒表特别敏感。
- **log-Mel 对比度（SpectralContrast）**：与 spectral_flatness 互补。
- **短时 RMS 的 CV 系数**：简易且稳健的"连续抖动"量化。

上述特征**未**加入 v4，因为需要真实 tdms 样本先行验证有效性。可以在
`test_v4_features.py`（见本任务第二个产出）里快速试。

### 4.2 数据层面
- epump2 目前只有 1 条产线，cross-line 泛化观察不到。建议**reference-out**
  split 作为主要评估，不要再看 sample-level split —— 样本级 split 会泄露
  "同一台电机"的相关性，训练集分数虚高。
- 建议对漏判的震颤件生成合成样本：用小幅振幅缩放（±10%）+ 时间平移
  （±0.1s）增强，对训练集扩 2–3 倍震颤，推理端保持原样。

### 4.3 工程层面
- `extract_features_v4.py` 当前冷启动要做 Mel + DWT + Hilbert + STFT，
  单条样本耗时估计 ~0.15s。若后续要上线，建议把 `_butter_coeffs`、
  `_mel_filterbank_cached`（v2 已有）等缓存机制扩展到包络路径。
- `XGBModel.py` 建议把 `feature_importances_` 持久化到
  `results/{exp}/feature_importance.csv`，便于每轮对照"新增特征是否真被
  模型用上"。

---

## 附录：主要发现的数据证据（Cohen's d）

基于 7196 条训练样本的 v2 特征（未含 v4 新增）：

| 特征 | 秒表 vs 正常 | 震颤 vs 正常 | 秒表 vs 震颤 |
|------|:-----------:|:-----------:|:-----------:|
| mel_band_wide_std* | 1.12 | **2.74** | -0.92 |
| mel_shock_med_count | 0.98 | **1.64** | -0.42 |
| mel_shock_high_count | **1.27** | 1.38 | -0.09 |
| mel_shock_high_avg_kurt* | 0.94 | -0.55 | **2.04** |
| mel_band_high_db_mean | 1.18 | 0.71 | 0.48 |
| mel_band_low_db_ratio | -0.88 | -0.42 | -0.41 |

（*标注表示 v4 新增；其余由 v2 原始特征计算）

结论：
- 震颤的"第一把刀"是 **mel_band_wide_std**（连续宽带抖动）
- 秒表的"第一把刀"是 **mel_shock_high_count + mel_shock_high_avg_kurt**
  （稀疏 + 尖锐）
- 二者最强分离轴是 **mel_shock_high_avg_kurt**，其次
  mel_band_wide_std（符号相反）。

# Forvia Train v2

基于 **FastAPI + Vue 3** 的模型训练 Web 应用。`forvia_train_v2` 只负责界面、
项目状态和任务编排，数据与训练算法统一调用外部接口：

- `ml.training.app_api`：项目配置、数据库统计和训练数据选项
- `python -m ml.dataset.generate`
- `data_manager/label_filter.py`
- `python -m ml.dataset.build`
- `python -m ml.train --step train`
- `ml.training.results`：训练结果解析
- 训练模型目录中的推理脚本

## 页面

```text
项目列表（模型名称） | 配置数据库 | 训练（配置参数 + 提取特征 + 训练） | 结果分析 | 模型推理
```

项目列表直接同步项目根目录下的 `cfg/*.yaml`。每次数据处理、训练和推理均生成独立运行记录。

- 项目支持创建、重命名、复制和删除。
- 配置数据库页面只负责数据库输入和查看现有数据、标签情况。
- 数据概览使用两个独立刷新按钮：按产线/reference 展示标签分布，或按 `型号数据.xlsx` 展示模型标签分布；不展示全部数据总计。
- 标签统计以独立 `line + sn + sample_id` 为单位，复用训练过滤规则，重复标签事件不重复计算。
- 训练数据分为主要数据和额外 key 数据；`data.key_sns` 只接受文件夹，通常选择 `factory_raw/prototype`。
- 创建项目时从 `cfg/core/line_rules.yaml` 读取产线下拉选项。
- 训练页按模型卡片展示对应参数，并包含数据范围、提取特征、训练和高级 YAML。
- 标签通过点击加入类别组，并可为每个标签设置训练样本数量。
- 运行详情展示阶段进度、日志和取消操作。

## 数据输入

v2 以新的 SQLite 数据库 `sample_records.db` 为主输入：

- `samples`：样本索引
- `label_events`：标签事件
- `status=confirmed`：训练使用的确认标签

同时配置 TDMS 根目录和 `tdms_manifest.csv`。v2 直接从新数据库与 manifest
生成 `sample_view.csv`，并在标签筛选、特征提取命令中显式传入数据库路径。
数据库读取使用 SQLite 只读连接，不要求数据库所在目录可写。

配置数据库页面通过 `forvia_train_v2.backend.database_service.execute_database_action` 提供三种方式：

- 新建数据库：选择 TDMS 根目录，标签来源支持 DB、内部标签事件表或 Forvia 宽表；Forvia 表会先转换为内部表再写入 DB，manifest 自动生成。
- 追加已有数据库：可追加 TDMS、多个标签 CSV，或只追加 CSV。
- 更新已有数据库：选择 line 时只更新该产线；不选择 line 时递归更新当前 TDMS 根目录下全部产线的 TDMS 路径和 sample_id。已有标签按 `line / sn / sample_id` 保留。

数据库操作作为后台任务执行，页面显示阶段与进度。当前项目的数据库路径卡片仅供查看，
只能通过上述操作修改。Train v2 只调用公共接口并保存返回路径，不包含数据库生成算法。

## 数据增强

训练页面的数据增强卡片调用外部 `data_augmentation.direct_features` 模块：

- 输入为一个或多个包含 `.tdms` / `.tdms.zst` 的文件夹。
- 支持设置每个原始 TDMS 的生成数量、增强方法和随机种子。
- 当前方法包括加噪、混响、时频遮挡、速度扰动、随机白噪声、随机裁剪、随机增加片段、随机振幅缩放、随机时序伸缩，每个方法位于独立 Python 文件。
- TDMS 信号只在内存中增强，不写增强 TDMS、manifest 或临时 sample view。
- 最终只写 `results/<model_id>/dataset_csv/features_batch_augmented.csv`。

```bash
python -m data_augmentation.direct_features \
  --config cfg/epump2_general.yaml \
  --input-folder /path/to/tdms \
  --output-dir results/epump2_general/dataset_csv \
  --count 2 \
  --methods add_noise,reverberation,time_frequency_mask,speed_perturb \
  --line epump2
```

## 运行

在项目根目录执行：

```bash
pip install -r forvia_train_v2/requirements.txt
uvicorn forvia_train_v2.backend.main:app --reload --port 8001
```

打开 `http://127.0.0.1:8001`。

## 状态与产物

```text
cfg/
├── core/label_rules.yaml
└── <line>_<model>.yaml
results/
└── <line>_<model>_<type>/
forvia_train_v2/
├── state/forvia_train_v2.db
└── workspaces/<project_id>/runs/<run_id>/
    ├── config_snapshot.yaml
    └── run.log
web/forvia_train_v2/index.html
└── 训练启动台前端页面
```

- 项目配置以 `cfg/*.yaml` 为主存储，SQLite 仅保存项目索引和运行状态。
- 标签定义统一读取 `cfg/core/label_rules.yaml`。
- 数据处理和训练产物统一写入根目录 `results/<model_id>/`。
- workspace 只保存每次运行的配置快照与日志。
- 所有训练 App 均通过 `ml.train.train_from_config` 执行训练。
  仅保留为旧调用方式的兼容入口。
- v2 的数据、训练和推理任务由独立 worker 执行；Web 后端重启不会中断正在运行的任务。

## 增加模型

模型实现和注册统一归外部 `ml` 管理，train_v2 不保存模型训练实现。

1. 在 `ml/` 中增加模型类，保持与现有模型一致的训练方法接口。
2. 在 `cfg/core/model_registry.yaml` 的 `models` 中登记 `module_name`、`class_name`、
   参数配置键及参数卡片字段。
3. 重启训练 App。新模型会自动进入项目创建下拉框和训练参数卡片。

Python 扩展也可以调用 `ml.models.registry.register_model(...)` 注册模型。所有 App
最终通过以下公开接口执行训练：

```python
from ml.train import train_from_config

train_from_config("cfg/my_model.yaml")
```

## 验证

```bash
conda run -n fault python -m pytest \
  tests/test_forvia_train_v2.py tests/test_sample_record_store.py -q
```

## 当前限制

- 新数据库负责样本与标签，`tdms_manifest.csv` 负责将样本定位到实际 TDMS。
- 当前后台执行器适用于单机部署；多实例部署需要外部任务队列。
- 前端 Vue 使用 CDN，生产离线部署时应切换到 Vite 构建。

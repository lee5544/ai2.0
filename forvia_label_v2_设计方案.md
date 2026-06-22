# Forvia 标注软件 v2 重新设计方案

> 目标：在保留现有标注体验的前提下，把入口、标签存储、频谱分析三块从"manifest + data_root + 4 套 workflow + Prototype 库"的旧结构，收敛成"两个输入路径 + 单一标签表 + 卡片化分析 + 可批处理任务"的简洁、可扩展架构。

---

## 1. 现状与问题

当前 `forvia_label` 的核心包袱（来自 `main.py` 3388 行 + `data_manager/*`）：

- **入口过重**：4 套 workflow（sample_view CSV / factory_raw 文件夹 / 外部标签导入 / 外部文件夹+线名），都依赖 `data_root` + `tdms_registry` 做路径解析。用户要先理解 data_root、storage_root、manifest 这些内部概念。
- **标签存储绕**：标注先写 `pending` 临时 CSV，再"合并"进 `data_root/metadata/label_history.csv`，依赖 manifest 反查路径、`label_source` 优先级、同日去重等一堆规则。
- **Prototype 库独立**：`PrototypeStore.add_prototype` 会另存 WAV/PNG/HTML/TDMS 副本，与标签表脱节，维护成本高。
- **频谱分析硬编码**：`render_detail_plot_buttons` 把 dwt/mel详情/pcen/mfcc/wavelet/emd 6 个分析写死在一个函数里，新增一种分析要改 `main.py` + `plot_utils.py`，无法插拔。
- **无批处理概念**：所有计算都在 Streamlit 的单次 rerun 里同步完成，无法"对全集跑一遍模型/分析"。

v2 针对这五点重构。

---

## 2. 总体架构（分层）

```
┌──────────────────────────────────────────────────────────┐
│ UI 编排层  app.py（薄）                                     │
│   InitScreen（两输入+可选标签）   LabelScreen（标注界面）     │
├──────────────────────────────────────────────────────────┤
│ 会话层  LabelSession                                        │
│   sample_view_df / tdms_root / 路径解析表 / 标签表 / 任务结果  │
├───────────────┬──────────────┬───────────────┬────────────┤
│ 数据访问       │ 标签层        │ 卡片层         │ 任务层      │
│ PathResolver   │ LabelTable    │ CardRegistry   │ TaskRunner  │
│ TdmsLoader     │ (单表+导出)   │ CardHost       │ TaskStore   │
│ FeatureCache   │ TypicalFlag   │ cards/*.py     │ tasks/*.py  │
└───────────────┴──────────────┴───────────────┴────────────┘
```

核心思想：**会话层 `LabelSession` 是唯一的状态中心**，UI 只负责渲染；数据访问、标签、卡片、任务四个子系统都只依赖 Session，互不耦合。卡片和任务都做成"目录里放一个 py 文件就能扩展"的插件形式。

---

## 3. 输入与初始化（需求 1/2/3）

### 3.1 输入界面：只有两个必填 + 一个可选

```
┌─ 开始标注 ───────────────────────────────┐
│ ① 选择 sample_view.csv      [浏览…] path  │
│ ② 选择 TDMS 存放目录(.zst)  [浏览…] path  │
│ ③ ☐ 加载已有标签                          │
│      └ 选择标签表格.csv      [浏览…] path  │  ← 勾选后才出现
│                                           │
│              [ 加载并开始标注 ]            │
└───────────────────────────────────────────┘
```

- 去掉 `data_root` / `storage_root` / manifest / 4 套 workflow。入口只有"样本清单"和"音频根目录"两个物理概念，用户无需理解内部结构。
- 第 ③ 项是开关：不勾 → 从空标签表开始；勾选 → 载入该表作为初始标签（既能续标，也能在模型/同事标签基础上复核）。

### 3.2 一次性初始化（点"加载并开始标注"时执行一次）

```python
class LabelSession:
    sample_view: pd.DataFrame      # 样本清单（原样保留）
    tdms_root: Path                # 用户选的 .zst 根目录
    path_map: dict[str, Path]      # sample_id/sn -> 实际 .tdms(.zst) 文件
    label_table: LabelTable        # 标签（空表或载入的表）
    task_results: TaskResultStore  # 任务输出（见 §6）

def init_session(sample_view_csv, tdms_root, label_csv=None) -> LabelSession:
    sv  = read_sample_view(sample_view_csv)          # 规范化列
    pm  = PathResolver(tdms_root).build(sv)          # 一次性解析所有路径
    lt  = LabelTable.load(label_csv) if label_csv else LabelTable.empty()
    lt.attach_index(sv)                              # 按 sample_id 建索引
    return LabelSession(sv, tdms_root, pm, lt, TaskResultStore())
```

- **`PathResolver`** 取代 manifest 反查：用 `tdms_root` + sample_view 里的相对路径/文件名/sn 直接定位文件，多策略兜底（精确相对路径 → 文件名匹配 → sn 模糊匹配），解析结果一次性缓存在 `path_map`，标注过程中零再解析。无法解析的样本在初始化报告里列出，不在标注流程里临时弹"选线名"。
- **标签一次性对齐**：载入的标签表按 `sample_id` 建索引，初始化即和 sample_view 对齐，每条样本的"最新标签"在进入界面前就准备好。

---

## 4. 标注界面布局（需求 4）

自上而下，与现有体验一致：

```
[标注进度]  ⬅上一条   ███████░░░ 63% (126/200)   下一条➡
─────────────────────────────────────────────
[通道面板  sn_up / sn_down]
   🔊 播放器
   📈 原始曲线
   ✂️ 裁剪曲线
   🌈 梅尔频谱          ← 以上四项是"默认卡片"
   ── [开始新的标注] ──  → 展开 reason/置信度/note + [保存标注]
   [★ 标记为典型异音]    ← 取代"加入 Prototype"
─────────────────────────────────────────────
[label_history]   [sample_view 行]   [sample_index]
─────────────────────────────────────────────
[高级频谱分析]   + 插入卡片 ▾  (dwt / pcen / mfcc / wavelet / emd …)
   每个分析 = cards/ 下一个独立 py 文件
─────────────────────────────────────────────
            [⬇ 导出标签]   （取代"合并标签"）
```

变化点：
- 顶部进度条 + 上/下一条按钮，保持现状（`render_progress_chips` + `st.progress` 复用）。
- 顶部加 **视图切换**：`列表总览 | 逐条标注` 两个标签页（见 §4.5），列表里双击进入逐条标注。
- 播放器、原始曲线、裁剪曲线、梅尔频谱：作为通道面板里 4 个**默认卡片**（见 §6），渲染逻辑等价于现有 `render_channel_panel`。
- "开始新的标注"逻辑不变（radio reason + 置信度 + note + 保存）。
- **"加入 Prototype 库" → "标记为典型异音"**：见 §5.2。
- 底部 **"合并标签" → "导出标签"**：见 §5.3。

### 4.5 样本标签总览面板（列表视图，双击进入标注）

在标注界面顶部用 `st.tabs(["📋 列表总览", "🎧 逐条标注"])` 增加一个总览面板，把"sample_view × 当前标签"按一行一个样本的表格列出，**双击某行直接跳到该样本的标注视图**。

**数据来源**：`LabelSession` 把 `sample_view` 与 `LabelTable.latest(sample_id)` 做左连接，生成总览 DataFrame（每条样本一行，已标注的带上最新标签）：

```python
def build_overview(session) -> pd.DataFrame:
    rows = []
    for idx, sv in session.sample_view.iterrows():
        sid = sv["sample_id"]; lab = session.label_table.latest(sid) or {}
        rows.append({
            "#": idx,                                   # 行号 = current_index，用于跳转
            "sn": sv["sn"], "sample_id": sid, "line": sv.get("line",""),
            "已标注": "✅" if lab else "—",
            "reason": lab.get("reason_name",""),
            "置信度": lab.get("reason_confidence",""),
            "来源": lab.get("source",""),
            "典型异音": "★" if TYPICAL_TAG in lab.get("note","") else "",
            "note": lab.get("note","").replace(TYPICAL_TAG,"").strip()[:30],
            "tdms": "✓" if sid in session.path_map else "缺失",   # 路径解析状态
        })
    return pd.DataFrame(rows)
```

**列**：行号 `#`、sn、sample_id、line、已标注、reason、置信度、来源、典型异音、note 预览、tdms（路径是否解析到）。

**筛选/排序**（面板顶部一行控件）：按 已标注/未标注、line、reason、是否典型、tdms 缺失 过滤；点列头排序。复用现有 `sidebar_list_view.py` 的筛选思路，但以主面板大表呈现。

**双击进入标注**——按 Streamlit 能力给两种实现，推荐 A：

- **A. `streamlit-aggrid`（真双击，推荐）**：用 AgGrid 渲染，监听 `cellDoubleClicked` / `onRowDoubleClicked` 事件，拿到该行 `#` → `session.set_current_index(#)` → 切到"逐条标注"标签页并 `st.rerun()`。AgGrid 还自带列筛选/排序/冻结，省去自己写控件。需加依赖 `streamlit-aggrid`。
- **B. 原生 `st.dataframe`（单击选中，零依赖兜底）**：`st.dataframe(df, on_select="rerun", selection_mode="single-row")`，选中行后点"进入标注"按钮跳转（或选中即跳）。无需额外依赖，但只能单击选中、不能真双击。

跳转逻辑统一：

```python
def goto_sample(session, row_index: int):
    session.set_current_index(int(row_index))     # 即现有 _set_current_index
    st.session_state["active_view"] = "label"     # 切到逐条标注 tab
    st.rerun()
```

**联动**：标注页保存标签后回到列表，对应行的"已标注/reason/典型异音"实时刷新（因为总览每次从 `LabelTable` 重算）；列表里也可直接显示进度统计（已标 N/总 M），与顶部进度条一致。

---

## 5. 标签存储与导出（需求 5）

> **存储格式与 `label_history.csv` 完全一致**：v2 直接复用现有 `LabelStore` 作为存储引擎，列、编码、去重、schema 迁移规则全部不变，使读写、复核、训练侧消费都保持兼容。只去掉 `pending → merge` 两段式和 manifest 依赖。

### 5.1 沿用 label_history.csv 的列（不新造 schema）

工作表就是 `label_history.csv`，列严格等于 `LabelStore.HEADER`：

```
line, sn, sample_id, timestamp, source,
result_key, result_id, result_name,
reason_key, reason_id, reason_name, reason_confidence,
label_version, note
```

`LabelTable` 只是 `LabelStore` 的薄封装，**不改列、不改文件格式**：

```python
class LabelTable:
    """薄封装 LabelStore：列 = LabelStore.HEADER（与 label_history.csv 一致）。"""
    def __init__(self, store: LabelStore): self.store = store

    @classmethod
    def load(cls, csv_path):           # 载入已有 label_history.csv（需求 2）
        return cls(LabelStore(csv_path))
    @classmethod
    def empty(cls, csv_path):           # 不载标签时，在 sample_view 同级新建同格式文件
        return cls(LabelStore(csv_path))

    def add_result(self, *, line, sn, sample_id, decision_result,
                   label_version, reason_confidence, note, source="human"):
        # 直接调用 LabelStore.add_result，写入即落盘到 label_history.csv
        ...
    def latest(self, sample_id) -> dict | None: ...
    def list_for(self, sample_id) -> list[dict]: ...     # label_history 面板用
    def set_typical(self, sample_id, flag: bool): ...
    def export(self, out_path, *, scope="all", only_typical=False): ...
```

- 复用 `data_manager/label.py` 的 `ReasonResolver` 和现有 `build_decision_result` → `result_key/result_id/result_name` + `reason_*` 这套列照常填充，**与现网 label_history.csv 逐列对齐**。
- 去掉的只是：① pending 临时 CSV；② 显式"合并"动作；③ 用 manifest 反查路径。标注 `add_result` 即直接写入 `label_history.csv`（`LabelStore` 已支持同日同源去重）。
- `source` 列继续用现有 `label_source` 取值（human/model/imported…），多源场景见 §7.1。
- 一条样本多次标注事件全部保留，`label_history` 面板照常展示该 sample_id 的全部行。

### 5.2 "标记为典型异音"——在不破坏 label_history 列的前提下

**采用方案 A：note 约定标记，零 schema 改动**，保持与 `label_history.csv` 列完全一致。

- 在当前样本最新标注行的 `note` 里写入约定前缀标记 `[[典型异音]]`，形如 `[[典型异音]] 原备注…`。文件列不变，训练侧/旧工具完全兼容。
- `set_typical(sample_id, flag)`：对该 sample 最新行的 `note` 加/去 `[[典型异音]]` 前缀（幂等，不重复添加）。
- 筛选典型样本：按 `note` 是否含 `[[典型异音]]` 过滤；导出时 `only_typical=True` 即据此筛选。
- 约定一个常量 `TYPICAL_TAG = "[[典型异音]]"` 统一管理，避免散落字符串。

**不再**用 `PrototypeStore` 另存 WAV/PNG/TDMS 副本——典型异音只是标签表上的一个标记，与标签同生命周期。

### 5.3 导出标签（取代"合并"）

```python
def export(self, out_path, *, scope="all", only_typical=False):
    # scope: all / 当前线 / 选中；only_typical: 仅典型异音
    # 输出列 == LabelStore.HEADER，与 label_history.csv 完全一致
```

- 工作文件本身已是标准 `label_history.csv`；"导出"= 按范围/典型筛选后产出一份同格式副本（带时间戳文件名，可指定路径），供训练侧直接使用。
- 不再回写到 data_root 隐藏目录；文件位置由用户在入口选择或落在 sample_view 同级。

---

## 6. 卡片插件化：梅尔频谱 / 高级频谱可插入（需求 4 + 你的第 3 个问题）

### 6.1 统一的卡片接口

每个分析（含梅尔频谱）= `cards/` 下一个**独立 py 文件**，暴露一个 `CARD`：

```python
# cards/mel.py
from card_api import Card, CardContext, register

@register
class MelCard(Card):
    id = "mel"
    title = "梅尔频谱"
    category = "spectrum"        # spectrum / waveform / feature / task
    default = True               # 是否默认出现在通道面板
    needs = ("proc", "sr")       # 声明依赖，宿主据此判断能否渲染

    def render(self, ctx: CardContext):
        fig = generate_mel_figure(ctx.proc, sr=ctx.sr)
        ctx.st.plotly_chart(fig, use_container_width=True)
```

`CardContext` 是宿主注入的只读上下文，把现在散落在 `main.py` 的参数收拢成一个对象：

```python
@dataclass
class CardContext:
    sn: str; sample_id: str; line: str
    raw: np.ndarray; proc: np.ndarray; sr: int
    features: dict                 # ml.features 提取结果（缓存）
    metadata: pd.DataFrame
    task_results: dict             # 该 sample 的任务输出（见 §7）
    cache_token: str               # 缓存键，复用现有 figure_cache 机制
    st: ModuleType                 # 注入的 streamlit 容器
```

### 6.2 注册与发现

```python
class CardRegistry:
    def discover(self, pkg="cards"):     # importlib 扫描目录，收集 @register
    def get(self, card_id) -> Card
    def by_category(self, cat) -> list[Card]
```

往 `cards/` 丢一个新 `.py`（实现 `Card`）→ 自动出现在"+ 插入卡片"菜单，**无需改 `app.py`**。把现有 6 个高级分析（dwt/pcen/mfcc/wavelet/emd/mel详情）各拆成一个文件，`generate_*_figure` 仍放 `plot_utils.py` 由卡片调用。

### 6.3 卡片宿主（CardHost）— "可插入"的关键

```python
class CardHost:
    """渲染一组卡片，支持 默认卡片 + 用户插入/移除/排序。"""
    def render(self, ctx, *, slot: str):
        layout = session.card_layout[slot]   # 每个 slot 一份有序 card_id 列表
        for cid in layout:
            with st.container(border=True):
                head, _, x = st.columns([6,1,1])
                head.markdown(registry.get(cid).title)
                if x.button("✕", key=f"rm_{slot}_{cid}"):
                    layout.remove(cid)
                registry.get(cid).render(ctx)
        # 插入控件
        choices = [c for c in registry.by_category("spectrum") if c.id not in layout]
        pick = st.selectbox("+ 插入卡片", ["—"]+[c.title for c in choices], key=f"add_{slot}")
        if pick != "—": layout.append(resolve_id(pick))
```

- **通道面板** = 一个 `slot="channel"` 的 CardHost，默认布局 `[waveform_raw, waveform_cut, mel]`。
- **高级频谱分析区** = 一个 `slot="advanced"` 的 CardHost，默认空，用户从下拉插入 dwt/pcen/… 任意张。
- 卡片布局 `card_layout` 存 `session_state`，可选持久化到本地 json，下次打开还原。
- 卡片可按 `category` 出现在不同 slot：把 mel 设 `default=True`、归类 `spectrum`，它就既是默认通道卡片、也能在高级区重复插入。

这样"梅尔频谱"和"高级频谱分析"在实现上是同一种东西——卡片，区别只是默认是否出现。

---

## 7. 任务系统：创建任务 / 执行任务（你的第 2 个问题）

### 7.1 会带来的逻辑变化（重点）

加入"创建任务/执行任务"后，有 3 处本质变化：

1. **同步 → 异步/批处理**。现在所有计算都在单条样本的 rerun 内同步完成。任务是"对一批样本跑一遍"（模型推理、批量特征、批量频谱预生成），耗时长，不能卡在 Streamlit 主线程。需要引入 **TaskRunner**：用子进程/后台线程执行，结果写盘，UI 轮询进度，与逐条标注解耦。

2. **标签从单源 → 多源**。任务可产出"模型建议标签"。于是标签表里 `source` 列真正启用（human / model / imported），出现**冲突协调**问题：同一 sample 人工标 vs 模型标。建议：模型结果写入**独立的"建议层"**（不直接污染人工标签），界面上以"建议: tick (0.92)"形式提示，人工"采纳"才落到 `LabelTable`。导出时可选"是否包含未采纳的建议"。这其实是把旧的 `label_source` 优先级概念，从"隐式合并"改成"显式采纳"。

3. **卡片成为任务结果的消费者**。任务输出（异常分数、聚类、模型注意力热图）通过 `CardContext.task_results[sample_id]` 注入，于是可以写"模型异常分数卡片""相似样本卡片"等 `category="task"` 的卡片，和频谱卡片同机制插入。逐条标注界面因此能展示批任务的结果。

### 7.2 任务抽象

```python
# tasks/anomaly_score.py
@register_task
class AnomalyScoreTask(Task):
    id = "anomaly_score"
    title = "模型异常分数"
    params = {"model_path": str, "scope": ["all","current_line","selected"]}

    def run(self, session, params, progress) -> Iterable[TaskRecord]:
        for i, sid in enumerate(session.iter_scope(params["scope"])):
            sig = session.load_proc(sid)
            yield TaskRecord(sample_id=sid, data={"score": model(sig)})
            progress(i)

class TaskRunner:        # 执行：子进程 + 进度回调 + 结果写 TaskResultStore
class TaskResultStore:   # {task_id: {sample_id: data}}，落盘 parquet/json，可重载
```

- **创建任务**：UI 选任务类型 + 范围(全集/当前线/选中) + 参数 → 入队。
- **执行任务**：TaskRunner 跑，进度条；完成后结果进 `TaskResultStore`，对应 `category="task"` 的卡片即可读取渲染；若任务产出标签，进入"建议层"等待采纳。
- `tasks/` 同样是"放一个 py 文件即扩展"，与 `cards/` 对称。

---

## 8. 还需完善的点

- **路径解析的健壮性**：sample_view 相对路径与实际目录结构不一致时的兜底与"未解析样本"报告页；大目录扫描要缓存/索引，避免每次 init 全盘 walk。
- **标签并发与撤销**：单表内存编辑需要"撤销上一步""删除某条事件"（现有 `_delete_label_history_row` 可迁移）；多人/多窗口同时编辑同一表时的覆盖检测（导出前比对 mtime）。
- **导出 schema 兼容**：导出列 == `LabelStore.HEADER`（与 label_history.csv 一致），已天然兼容训练侧；建议导出前再跑一次 schema 校验兜底。
- **"典型异音"落地方式需拍板**：§5.2 的 A（note 标记，零 schema 改动）vs B（尾部追加 `typical_abnormal` 列）。需确认训练侧能否容忍尾部新增列；若需要典型样本的图/音频快照，可在导出时按标记单独打包 WAV（保留旧 PrototypeStore 的导出能力，但只在导出时做、不在标注时做）。
- **卡片性能**：插入多张重频谱卡片会拖慢 rerun，需沿用现有 `_channel_figure_cache`（48 条 LRU）并对每张卡片懒渲染（折叠时不算）。
- **去 i18n**：删除 `i18n.py` 与所有 `T(...)`/语言切换按钮，界面文案统一硬编码中文，简化代码。
- **任务结果版本**：模型换版本后 TaskResultStore 要带 `model_version`，避免旧分数与新标签混用。
- **只读模式**：载入他人标签复核时，应支持"只看不写"开关（现有 `sample_view_read_only` 可复用）。

---

## 9. 模块清单与迁移

| v2 模块 | 来源 / 处理 |
|---|---|
| `app.py`（UI 编排，薄） | 从 `main.py` `main()` 抽取、大幅瘦身 |
| `session.py` `LabelSession` | 新增，吸收 `init_runtime` 的状态聚合职责 |
| `path_resolver.py` | 取代 `tdms_registry` 反查 + `sample_view_path_service` |
| `tdms_loader.py` | 复用 `data_manager/tdms_read.py` + `utils/plot_utils.process_data` |
| `label_table.py` | 薄封装**复用 `LabelStore`**（label_history.csv 格式不变），仅去掉 pending/merge/manifest；复用 `Label.ReasonResolver` |
| `card_api.py` + `cards/*.py` | 新增；`cards/{mel,raw,cut,dwt,pcen,mfcc,wavelet,emd}.py` 调用 `plot_utils.generate_*` |
| `card_host.py` | 新增（插入/移除/排序） |
| `overview_view.py` | 新增，样本标签总览列表 + 双击跳转（§4.5）；筛选思路复用 `sidebar_list_view.py`；推荐依赖 `streamlit-aggrid` |
| `task_api.py` + `tasks/*.py` + `task_runner.py` + `task_store.py` | 新增 |
| 删除 | `PrototypeStore`、4 套 workflow 入口、pending/merge 面板、manifest 路径反查、**`i18n.py`（中英切换全部移除，界面统一中文）** |
| 保留复用 | **`LabelStore`（label_history.csv 存储引擎）**、`plot_utils.py`、`ml/features`、`ui_primitives.py`、`Label.ReasonResolver`、`build_decision_result` |

迁移建议分四步：① 先做 `LabelSession`+两输入入口跑通逐条浏览；② 切 `LabelTable`+导出替换 merge；③ 卡片化（先把 mel/原始/裁剪改成默认卡片，再迁高级分析）；④ 最后加任务系统。每步可独立验证、不破坏标注主流程。

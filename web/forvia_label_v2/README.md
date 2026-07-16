# Forvia 标注 v2 · 样本标签总览面板（独立可运行）

技术栈：**FastAPI（后端）+ Vue 3 + AG Grid（前端）**，Web 架构、跨平台、可 Docker。
本目录是 v2 的第一个功能切片：样本标签总览列表 + 双击进入标注。

## 功能

- 列表展示 `sample_view × sample_records.db` 连接后的每条样本（一行一个 sample_id）。
- 列：行号 `#`、sn、sample_id、line、已标注、reason、置信度、来源、典型异音 ★、note、tdms 状态。
- 顶部筛选：标注状态 / 线名 / 仅典型异音 / 仅 tdms 缺失；表头可排序、按列筛选。
- **双击任意行 → 进入该样本的标注视图**（当前为占位，后续接 tdms 加载 + 波形/梅尔/高级频谱卡片）。
- 标签直接存储在 `sample_records.db`；典型异音 = note 含 `[[prototype]]`。

## 本地运行

```bash
pip install -r requirements.txt
cd forvia_label_v2
conda activate fault  
uvicorn backend.main:app --reload --port 8000
# 打开 http://localhost:8000
```

不配置数据时自动用内置演示数据（120 行），开箱即跑。

接真实数据（用环境变量注入路径）：

```bash
export SAMPLE_VIEW_PATH=/path/to/sample_view.csv
export LABEL_RECORDS_DB_PATH=/path/to/sample_records.db
export TDMS_ROOT=/path/to/tdms_zst_root                # 用于判断 tdms 是否解析到
uvicorn backend.main:app --port 8000
```

## Docker（跨平台部署）

```bash
docker compose up --build
# 打开 http://localhost:8000
```

Docker 构建前请先执行 `./scripts/run-mac.sh`，脚本会从仓库 `web/forvia_label_v2/`
暂存前端到 Docker 构建上下文，停止后自动清理暂存目录。

真实数据：把数据放进 `./data/`，在 `docker-compose.yml` 里设置
`SAMPLE_VIEW_PATH=/data/sample_view.csv`、`LABEL_RECORDS_DB_PATH=/data/sample_records.db` 等环境变量即可。

## 接口

| 方法 | 路径                    | 说明                       |
| ---- | ----------------------- | -------------------------- |
| GET  | `/`                   | 前端页面                   |
| GET  | `/api/overview`       | 总览行 + 统计 + 线名列表   |
| GET  | `/api/sample/{index}` | 单样本详情（标注视图占位） |

## 结构

```
forvia_label_v2/
├── backend/
│   ├── main.py     FastAPI 应用 + 路由
│   └── session.py  sample_view × sample_records.db 连接、筛选、典型异音标记
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
web/forvia_label_v2/
├── index.html      Vue3 + AG Grid 单页（CDN，无需 node 构建）
└── lib/            本地静态依赖
```

## 说明 / 下一步

- 前端现用 CDN 引入 Vue/AG Grid，无构建步骤，便于快速验证。正式 v2 建议改 **Vite + 单文件组件**，Dockerfile 升级为多阶段构建（node 编译 → 静态文件拷入 Python 镜像，FastAPI 同时服务 API 和静态页）。
- 卡片（波形/梅尔/高级频谱）后续按设计文档：后端 `cards/*.py` 计算并返回 **Plotly figure JSON**，前端用通用 Plotly.js 渲染器渲染，保持"丢一个 py 文件即扩展"。
- 双击跳转的 `index` 即设计文档里的 `current_index`，与逐条标注共用。

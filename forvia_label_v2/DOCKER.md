# Forvia 标注 v2 · Docker 运行（Mac）

镜像自包含（内置复用的 v1 子集 data_manager / sample_view / cfg），不依赖宿主仓库，
构建一次即可独立运行。前端走 CDN，无需 node 构建。

## 前置
- 安装 **Docker Desktop for Mac**（Apple Silicon / Intel 均可）并启动。

## 三步运行
```bash
cd forvia_label_v2

# 1) 配置数据目录：编辑 .env（首次运行脚本会自动从 .env.example 生成）
#    把 FORVIA_DATA_DIR 指向你的数据所在目录，它会挂到容器内 /data。
#    例（外接盘）： FORVIA_DATA_DIR=/Volumes/18555440521

# 2) 一键构建并启动（首次较慢，需联网装依赖）
./scripts/run-mac.sh          # 前台运行，Ctrl+C 停止
# 或后台： ./scripts/run-mac.sh -d

# 3) 浏览器打开
open http://localhost:8000
```

## 在界面里怎么填路径
容器里看到的是挂载进来的 **`/data`**，所以任务表单填**容器内路径**：

| 表单项 | 例（宿主 `FORVIA_DATA_DIR=/Volumes/18555440521`） |
|---|---|
| 第一步 TDMS 目录 | `/data/fault/data_root/factory_raw/epump2` |
| 第三步 数据库文件夹 | `/data/fault/data_root/metadata` |
| 第二步 sample_view.csv（可选） | `/data/.../sample_view.csv` |

页面里的“浏览本地文件”浏览的就是容器内的 `/data`，可直接点选。

## 数据持久化
- 标注库、任务配置、缓存都存在 `forvia_label_v2/_data/`（已挂载为容器 `/app/_data`）。
  **重启 / 重建容器后标注都还在。** 不要删除该目录。
- “导出新表格 / 写回 sample_view”会写到容器内的 `/data/...`，即对应宿主的数据目录。

## 常用命令
```bash
docker compose logs -f            # 看日志
docker compose down               # 停止
docker compose up --build -d      # 改了代码后重建
```

## 访问外接盘 `/Volumes/18555440521`（重点）
Docker Desktop 跑在 Linux 虚拟机里，宿主目录必须先“共享”给 Docker，容器才看得到。
外接盘默认不共享，且**新版默认的 VirtioFS 经常不支持外接盘**——所以要按下面三步切到 gRPC FUSE：

**① 切换文件共享实现为 gRPC FUSE**
Docker Desktop → Settings → **General** → 文件共享实现选 **“gRPC FUSE”**（取消 VirtioFS）→ Apply & Restart。
> gRPC FUSE 对 `/Volumes/...` 外接盘的支持比 VirtioFS 稳。

**② 把外接盘加入共享目录**
Settings → **Resources → File sharing** → “+” → 加入 `/Volumes/18555440521`（或直接加 `/Volumes`）→ Apply & Restart。

**③ 指向外接盘并启动**
```bash
# .env 里设：
# FORVIA_DATA_DIR=/Volumes/18555440521
./scripts/run-mac.sh
```
表单里填容器内路径：TDMS 目录 `/data/fault/data_root/factory_raw/epump2`、数据库 `/data/fault/data_root/metadata`。

**验证容器确实看到了盘：**
```bash
docker compose exec forvia-label-v2 ls -la /data
```
能列出盘里的目录就成功了；若为空说明①②没生效，重做并确认 Docker 已重启。

> 提示：外接盘经 gRPC FUSE 读取会比本地慢，但本程序对解压后的信号有本地缓存（容器内 `/tmp`），
> 顺序翻页/二次查看会快很多；首次解压每条仍受外接盘 IO 限制。

**实在挂不上时的退路**：把数据放主目录（Docker 默认共享 `/Users`），
`FORVIA_DATA_DIR=/Users/你的用户名/forvia_data`。盘太大放不下时优先用上面的 gRPC FUSE 方案。

## 排查：点“选择…”报 `libtk8.6.so ...`
这是**正常**的——容器里没有图形界面，系统文件对话框用不了。
**改用页面上方的“文件浏览器”**（已默认从 `/data` 开始），或直接在输入框**手动粘贴容器内路径**（如 `/data/fault/data_root/factory_raw`）。容器模式下后端已不再尝试弹系统对话框，只给出该提示。

## 其它
- 改了后端/前端代码后，重新跑 `./scripts/run-mac.sh`（会重新暂存 vendor 并重建）。
- 首次构建会编译/下载 numba、librosa 等依赖，请耐心等待并保持联网。
- Windows 版见 `DOCKER.win.md`（待补）。

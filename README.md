# Forvia AI 2.0 Console

工业声学数据的数据库、标注、训练和推理工具。

## 环境要求

- macOS、Linux 或 Windows
- 已安装 Conda
- 已创建并激活自己的 Conda 环境
- Python 3.10 或更高版本

项目不会强制使用名为 `fault` 的环境。所有命令都使用当前已激活的 Conda 环境。

如果未激活 Conda 环境，启动脚本会停止并提示，不会打开一个无法连接的网页。

## 安装依赖

在项目根目录执行：

### macOS / Linux

```bash
conda activate <你的环境名>
chmod +x install_forvia_dependencies.sh start_forvia_console.sh
./install_forvia_dependencies.sh
```

### Windows

在 Anaconda Prompt 中执行：

```bat
conda activate <你的环境名>
install_forvia_dependencies.bat
```

安装脚本会从当前环境调用 Python 和 pip，不会创建或切换 Conda 环境。

## 启动网页

### macOS / Linux

```bash
conda activate <你的环境名>
./start_forvia_console.sh
```

启动后访问：

- Console 首页：`http://127.0.0.1:8012/console`
- Label v2：`http://127.0.0.1:8012/`
- Train v2：`http://127.0.0.1:8001/`

脚本会同时启动 Label v2、Train v2 和 Console 首页，并在 macOS 上自动打开首页。首页由 Label 服务提供，避免直接打开本地 HTML 导致模块跳转失效。

### Windows

在 Anaconda Prompt 中激活自己的环境后，双击或执行：

```bat
start_forvia_console.bat
```

脚本会打开两个服务窗口，并自动打开 `http://127.0.0.1:8012/console`。

## 关闭服务

启动脚本保持运行状态。回到启动脚本所在的终端按 `Ctrl+C`，会关闭 Label v2、Train v2 及其占用的端口。

如果直接关闭浏览器网页，后台服务不会自动退出。

## 目录说明

- `web/forvia_label_v2/`：数据库、标注和高级频谱分析
- `web/forvia_train_v2/`：数据集、特征、训练和结果分析
- `web/forvia_console_preview.html`：统一 Console 首页
- `web/forvia_console_modules.js`：Console 模块入口配置
- `cfg/`：全局规则和模型配置
- `install_forvia_dependencies.sh` / `.bat`：本地依赖安装脚本

数据库、TDMS、训练结果和运行日志应放在项目外部或配置的数据目录中，不提交到 Git。

首次使用时进入 **数据库 → 打开数据库**，选择本机的 `data_root`；或者使用 **新建数据库** 创建新的数据目录。数据库路径保存在本机浏览器和本机配置中，不随 Git 同步。

训练页提供四个内置模型配置：`epump2_general`、`epump3_general`、`epump4_general`、`etilt1_general`。它们只包含模型和训练参数，不包含本机数据库路径；打开项目后在“配置项目信息”中选择数据库即可。

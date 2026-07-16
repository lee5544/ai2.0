# 打包成「双击就能用」的 App（Mac）

目标：发给同事一个 App，**对方无需装 Docker、无需装 Python、不用改任何路径**，
双击打开后在浏览器里用，**直接选择本地/外接盘的文件夹**即可（外接盘 `/Volumes/...` 原生可读）。

> 为什么不用 Docker：Docker 一定要对方先装 Docker Desktop 并配置文件共享，外接盘还经常读不到。
> 原生 App 没有这些限制——它就是个本地程序，能直接访问整个磁盘。

## 一、你（开发者）构建一次
在你的 Mac 上、仓库根目录执行：
```bash
cd forvia_label_v2
./scripts/build-app-mac.sh
```
产物：`dist/Forvia标注v2.app`

建议在干净的虚拟环境里构建，避免把多余依赖打进去：
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r web/forvia_label_v2/requirements.txt pyinstaller
pyinstaller --clean --noconfirm web/forvia_label_v2/forvia_label_v2.spec
```

## 二、分发给别人
1. 把 `dist/Forvia标注v2.app` 压成 zip 发给对方。
2. 对方解压后**双击**即可：会自动启动本地服务并打开浏览器页面。
3. 首次打开若提示「来自身份不明的开发者」：右键 → 打开，或运行一次
   `xattr -dr com.apple.quarantine /路径/Forvia标注v2.app`。

> 注意：未做苹果签名/公证的 App，跨机器首次打开需要上面这步“去隔离”。
> 若要免这步，需要 Apple 开发者账号做 codesign + notarize（可后续再加）。

## 三、对方怎么用
- 双击 App → 浏览器自动打开页面。
- 在「任务」里点「选择…」直接浏览本机/外接盘文件夹（TDMS 目录、数据库文件夹、sample_view）。
- 标注结果、任务配置自动存到对方用户目录：
  `~/Library/Application Support/ForviaLabelV2/`（卸载 App 不会丢；重装仍在）。
- 关闭弹出的终端小窗口即退出程序。

## 四、架构 / Win 版
- App 内置了 FastAPI 服务 + 前端 + 复用的 v1 子集（data_manager / sample_view / cfg），自包含。
- 路径都是运行时选择，**代码里没有写死任何数据路径**。
- Windows 版：在 Windows 上用同一个 spec 跑 `pyinstaller web/forvia_label_v2/forvia_label_v2.spec`
  会得到 `dist/Forvia标注v2/Forvia标注v2.exe`（onedir）。需要在 Windows 机器上构建（PyInstaller 不能跨平台）。

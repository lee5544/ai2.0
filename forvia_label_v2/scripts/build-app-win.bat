@echo off
REM 在 Windows 上跑一次，产出可双击的 exe：dist\Forvia标注v2\Forvia标注v2.exe
REM 接收者无需装 Python，双击 exe 即用。注意：必须在 Windows 上构建，无法用 Mac 交叉编译。
setlocal enabledelayedexpansion
chcp 65001 >nul

REM 目录定位：本脚本在 forvia_label_v2\scripts\，HERE=forvia_label_v2，REPO=仓库根 AI-2.0
set "HERE=%~dp0.."
pushd "%HERE%\.." || (echo 找不到仓库根 & exit /b 1)
set "REPO=%CD%"

echo == [1/3] 安装依赖（建议先建干净 venv: python -m venv .venv ^&^& .venv\Scripts\activate）
python -m pip install --upgrade pip
python -m pip install -r "%HERE%\requirements.txt" || (echo 依赖安装失败 & popd & exit /b 1)
python -m pip install pyinstaller || (echo 安装 PyInstaller 失败 & popd & exit /b 1)

echo == [2/3] 打包（首次较慢，会收集 librosa/numba 等二进制）
rmdir /s /q "%REPO%\build\Forvia标注v2" 2>nul
rmdir /s /q "%REPO%\dist\Forvia标注v2" 2>nul
pyinstaller --clean --noconfirm "%HERE%\forvia_label_v2.spec" || (echo 打包失败 & popd & exit /b 1)

echo == [3/3] 完成
echo     程序文件夹: %REPO%\dist\Forvia标注v2\
echo     双击运行:   %REPO%\dist\Forvia标注v2\Forvia标注v2.exe
echo     分发: 把整个 dist\Forvia标注v2 文件夹压成 zip 发送（exe 不能单独拷出，依赖同目录文件）
popd
endlocal

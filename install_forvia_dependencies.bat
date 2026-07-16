@echo off
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

where python >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到当前 Python 环境，请先激活 conda 环境。
    popd
    exit /b 1
)

for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)"') do set "PYTHON_PATH=%%P"
echo 使用当前 Python 环境：%PYTHON_PATH%
call python -m pip install --upgrade pip
if errorlevel 1 goto failed
call python -m pip install -r web\forvia_label_v2\requirements.txt -r web\forvia_train_v2\requirements.txt pyinstaller
if errorlevel 1 goto failed
call python -c "import fastapi, uvicorn, numpy, pandas, scipy, librosa, soundfile, nptdms, zstandard, plotly, pywt, sklearn, xgboost, lightgbm, matplotlib; print('Forvia AI2.0 依赖检查通过')"
if errorlevel 1 goto failed
echo 依赖安装完成。当前环境可直接启动 Label/Train 服务。
popd
exit /b 0

:failed
echo 依赖安装失败，请检查网络、Python 版本和错误信息。
popd
pause
exit /b 1

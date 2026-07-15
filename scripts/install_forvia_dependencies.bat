@echo off
setlocal
set "ROOT=%~dp0.."
set "CONDA_ENV=%FORVIA_CONDA_ENV%"
if "%CONDA_ENV%"=="" set "CONDA_ENV=fault"
pushd "%ROOT%"

where conda >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 conda，请先安装 conda。
    popd
    exit /b 1
)
conda env list | findstr /R /C:"^[ ]*%CONDA_ENV%[ ]" >nul 2>&1
if errorlevel 1 (
    echo 错误：conda 环境不存在：%CONDA_ENV%
    popd
    exit /b 1
)

echo 使用 conda 环境：%CONDA_ENV%
call conda run -n %CONDA_ENV% --no-capture-output python -m pip install --upgrade pip
if errorlevel 1 goto failed
call conda run -n %CONDA_ENV% --no-capture-output python -m pip install -r forvia_label_v2\requirements.txt -r forvia_train_v2\requirements.txt pyinstaller
if errorlevel 1 goto failed
call conda run -n %CONDA_ENV% --no-capture-output python -c "import fastapi, uvicorn, numpy, pandas, scipy, librosa, soundfile, nptdms, zstandard, plotly, pywt, sklearn, xgboost, lightgbm, matplotlib; print('Forvia AI2.0 依赖检查通过')"
if errorlevel 1 goto failed
echo 依赖安装完成。请激活 %CONDA_ENV% 后启动 Label/Train 服务。
popd
exit /b 0

:failed
echo 依赖安装失败，请检查网络、Python 版本和错误信息。
popd
pause
exit /b 1

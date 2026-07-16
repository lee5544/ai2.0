@echo off
setlocal
set "ROOT=%~dp0"
pushd "%ROOT%"

where python >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到当前 Python 环境，请先激活 Conda 环境。
    popd
    pause
    exit /b 1
)

python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo 错误：当前 Python 环境没有安装 uvicorn。
    echo 请先运行 install_forvia_dependencies.bat。
    popd
    pause
    exit /b 1
)

set "PYTHONPATH=%ROOT%web;%PYTHONPATH%"
echo 使用当前 Python 环境：
python -c "import sys; print(sys.executable)"
echo 启动 Label v2 和 Train v2...

start "Forvia Label v2" /D "%ROOT%" cmd /k "set PYTHONPATH=%PYTHONPATH% && python -m uvicorn forvia_label_v2.backend.main:app --reload --port 8012"
start "Forvia Train v2" /D "%ROOT%" cmd /k "set PYTHONPATH=%PYTHONPATH% && python -m uvicorn forvia_train_v2.backend.main:app --reload --port 8001"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8012/console"

echo Forvia AI2.0 Console 已启动。
echo Console: http://127.0.0.1:8012/console
echo Label v2: http://127.0.0.1:8012/
echo Train v2: http://127.0.0.1:8001/
echo 请保留两个服务窗口；关闭服务窗口即可停止对应服务。
popd
endlocal

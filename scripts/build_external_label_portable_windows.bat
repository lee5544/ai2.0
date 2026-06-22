@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set ENV_DIR=%CD%\build\external_label_env
set PACKAGE_DIR=%CD%\dist\ExternalLabelPortable
set RUNTIME_ZIP=%CD%\build\external_label_runtime.zip

where conda >nul 2>nul
if errorlevel 1 (
    echo Conda was not found. Please run this from Anaconda Prompt or Miniforge Prompt.
    pause
    exit /b 1
)

if exist "%ENV_DIR%" rmdir /s /q "%ENV_DIR%"
if exist "%PACKAGE_DIR%" rmdir /s /q "%PACKAGE_DIR%"
if exist "%RUNTIME_ZIP%" del /f /q "%RUNTIME_ZIP%"

echo.
echo Creating slim Python runtime...
conda create -y -p "%ENV_DIR%" -c conda-forge python=3.12 streamlit pandas numpy plotly nptdms pyyaml zstandard librosa conda-pack
if errorlevel 1 goto fail

echo.
echo Packing runtime...
conda run -p "%ENV_DIR%" conda-pack -p "%ENV_DIR%" -o "%RUNTIME_ZIP%" --force
if errorlevel 1 goto fail

mkdir "%PACKAGE_DIR%"
mkdir "%PACKAGE_DIR%\runtime"

echo.
echo Unpacking runtime into portable package...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Expand-Archive -Path '%RUNTIME_ZIP%' -DestinationPath '%PACKAGE_DIR%\runtime' -Force"
if errorlevel 1 goto fail

xcopy /E /I /Y external_label "%PACKAGE_DIR%\external_label" >nul
xcopy /E /I /Y cfg "%PACKAGE_DIR%\cfg" >nul
copy /Y scripts\start_external_label_portable.bat "%PACKAGE_DIR%\启动.bat" >nul

echo.
echo Creating zip package...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path 'dist\ExternalLabelPortable' -DestinationPath 'dist\ExternalLabelPortable_windows.zip' -Force"
if errorlevel 1 goto fail

echo.
echo Portable package complete:
echo %CD%\dist\ExternalLabelPortable_windows.zip
echo.
echo Send this zip to the target Windows computer. The user only needs to unzip and double-click 启动.bat.
pause
exit /b 0

:fail
echo.
echo Build failed.
pause
exit /b 1

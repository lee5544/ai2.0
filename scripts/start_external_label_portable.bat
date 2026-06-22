@echo off
setlocal

cd /d "%~dp0"

if not exist "runtime\python.exe" (
    echo runtime\python.exe not found.
    echo Please rebuild the portable package.
    pause
    exit /b 1
)

if exist "runtime\Scripts\conda-unpack.exe" (
    if not exist "runtime\.conda_unpacked" (
        echo Preparing runtime for this folder...
        runtime\Scripts\conda-unpack.exe
        if errorlevel 1 (
            echo Runtime preparation failed.
            pause
            exit /b 1
        )
        echo ok>runtime\.conda_unpacked
    )
)

start "" http://localhost:8501
runtime\python.exe -m streamlit run external_label\app.py --server.headless=true --browser.gatherUsageStats=false

if errorlevel 1 (
    echo.
    echo Program exited with an error.
    pause
)

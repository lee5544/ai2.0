@echo off
setlocal

cd /d "%~dp0\..\dist\ForviaExternalFolderLabel"

if not exist "ForviaExternalFolderLabel.exe" (
    echo ForviaExternalFolderLabel.exe not found.
    echo Please run scripts\build_forvia_external_folder_label_windows.bat first.
    pause
    exit /b 1
)

ForviaExternalFolderLabel.exe
if errorlevel 1 (
    echo.
    echo Program exited with an error.
    pause
)

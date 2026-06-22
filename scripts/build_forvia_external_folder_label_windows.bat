@echo off
setlocal

cd /d "%~dp0\.."

set PYINSTALLER_CONFIG_DIR=%CD%\.pyinstaller

pyinstaller --clean --noconfirm forvia_external_folder_label.spec
if errorlevel 1 (
    echo.
    echo Build failed.
    exit /b 1
)

echo.
echo Build complete:
echo %CD%\dist\ForviaExternalFolderLabel\ForviaExternalFolderLabel.exe

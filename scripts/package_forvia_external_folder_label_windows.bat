@echo off
setlocal

cd /d "%~dp0\.."

if not exist "dist\ForviaExternalFolderLabel\ForviaExternalFolderLabel.exe" (
    echo ForviaExternalFolderLabel.exe not found.
    echo Please run scripts\build_forvia_external_folder_label_windows.bat first.
    pause
    exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path 'dist\ForviaExternalFolderLabel' -DestinationPath 'dist\ForviaExternalFolderLabel_windows.zip' -Force"

if errorlevel 1 (
    echo.
    echo Package failed.
    pause
    exit /b 1
)

echo.
echo Package complete:
echo %CD%\dist\ForviaExternalFolderLabel_windows.zip
pause

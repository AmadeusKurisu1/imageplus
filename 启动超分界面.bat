@echo off
chcp 65001 >nul
title Real-ESRGAN 图像超分辨率
cd /d "%~dp0"
echo ========================================
echo   Real-ESRGAN 图像超分辨率
echo   启动中...
echo ========================================
echo.

set "PYTHON_EXE=python"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
)

"%PYTHON_EXE%" "%~dp0启动超分界面.py"
if errorlevel 1 (
    echo.
    echo 启动失败。请确认已安装依赖，或先创建并安装 .venv 环境。
    echo.
)
pause

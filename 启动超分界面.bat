@echo off
chcp 65001 >nul
title Real-ESRGAN 图像超分辨率
cd /d "%~dp0Real-ESRGAN"
echo ========================================
echo   Real-ESRGAN 图像超分辨率
echo   启动中...
echo ========================================
echo.
start http://127.0.0.1:7860
python app.py
pause
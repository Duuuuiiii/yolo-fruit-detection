@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
if errorlevel 1 (
    echo.
    echo 启动失败，请先按 README 安装 Python 环境和依赖。上方是具体错误。
    pause
)

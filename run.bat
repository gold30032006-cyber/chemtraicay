@echo off
chcp 65001 >nul
cd /d "%~dp0"
title FRUIT SLASH - Setup

where py >nul 2>&1
if errorlevel 1 (
    echo Khong tim thay Python Launcher. Hay cai Python 3.11 64-bit.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Tao moi truong Python 3.11...
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc virtual environment. Kiem tra Python 3.11.
        pause
        exit /b 1
    )
)

echo [2/3] Cai dat thu vien...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo [3/3] Khoi dong FRUIT SLASH...
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo Game ket thuc voi loi. Hay chup man hinh loi de kiem tra.
    pause
)
exit /b 0

:error
echo.
echo Cai dat that bai. Kiem tra ket noi Internet va Python 3.11 64-bit.
pause
exit /b 1

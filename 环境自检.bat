@echo off
chcp 936 >nul
title 黑流树海识别 - 环境自检
cd /d "%~dp0"
echo ============================================
echo   黑流树海 路线识别 - 环境自检（首次使用先跑这个）
echo ============================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0环境自检.ps1"
echo.
pause

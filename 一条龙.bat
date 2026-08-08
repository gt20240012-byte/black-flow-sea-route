@echo off
chcp 936 >nul
title 黑流树海 路线识别 - 一条龙
cd /d "%~dp0"
setlocal enabledelayedexpansion
set "PYTHONIOENCODING=gbk"
set "OUT=输出"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%OUT%\历史结果" mkdir "%OUT%\历史结果"

echo ============================================
echo   黑流树海 路线识别 一条龙（v40.6）
echo ============================================
echo.

REM ---------- 0. 读取环境配置（环境自检.bat 生成） ----------
set "CFG=%~dp0env.ini"
if exist "%CFG%" goto :have_cfg
echo [错误] 还没跑过环境自检！
echo        请先双击 环境自检.bat 自动检测模拟器和 Python。
echo.
pause
exit /b 1
:have_cfg
for /f "usebackq eol=# tokens=1,* delims==" %%a in ("%CFG%") do set "%%a=%%b"

REM ---------- 1. 定位 adb（MuMu 自带，自动发现） ----------
set "ADB=%MUMU_DIR%\nx_main\adb.exe"
if not exist "%ADB%" set "ADB=adb"

REM ---------- 2. 连接模拟器（自动尝试配置里的端口） ----------
echo [1/5] 连接模拟器 (MuMu 端口: %ADB_PORTS%)...
set "ADB_OK="
for %%p in (%ADB_PORTS%) do (
    if "!ADB_OK!"=="" (
        "%ADB%" connect 127.0.0.1:%%p >nul 2>&1
        "%ADB%" -s 127.0.0.1:%%p get-state >nul 2>&1
        if not errorlevel 1 set "ADB_OK=%%p"
    )
)
if not "!ADB_OK!"=="" goto :adb_ok
echo.
echo [错误] 模拟器没连上！请先打开 MuMu 模拟器并启动明日方舟。
echo        如果还是连不上，重新双击 环境自检.bat 刷新配置。
echo.
pause
exit /b 1
:adb_ok
echo 模拟器在线 √ (端口 %ADB_OK%)
echo.

REM ---------- 3. 就位提示 ----------
echo [2/5] 请在游戏里就位：黑流树海地图 + 徒步模式，站着别动
echo        （录屏 20 秒内烟雾自己飘，你不需要操作）
echo.
pause
echo 开始录制，20 秒内请不要操作游戏...
echo.

REM ---------- 4. 截图 + 录屏 20 秒 ----------
echo [3/5] 截图 + 录屏 20 秒...
"%ADB%" -s 127.0.0.1:%ADB_OK% shell screencap -p /sdcard/shot.png
"%ADB%" -s 127.0.0.1:%ADB_OK% pull /sdcard/shot.png "%OUT%\shot.png" >nul
"%ADB%" -s 127.0.0.1:%ADB_OK% shell "screenrecord --bit-rate 6000000 --time-limit 20 /sdcard/road.mp4"
"%ADB%" -s 127.0.0.1:%ADB_OK% pull /sdcard/road.mp4 "%OUT%\road.mp4" >nul
echo 录制完成 √
echo.

REM ---------- 5. 判路（自动选 Python：Windows 原生 或 WSL） ----------
echo [4/5] v40.6 判路中（约 30 秒）...
if /i "%PY_CMD%"=="python" (
    python 判路_v40终版.py "%OUT%\road.mp4" "%OUT%\shot.png" "%OUT%\result"
) else (
    REM Windows 路径 → WSL 路径（不依赖 wslpath，C:/D: 盘都支持）
    set "WINPATH=%~dp0"
    set "WSLPATH=!WINPATH:~0,1!"
    if /i "!WSLPATH!"=="C" set "WSLPATH=!WINPATH:C:\=/mnt/c/!"
    if /i "!WSLPATH!"=="D" set "WSLPATH=!WINPATH:D:\=/mnt/d/!"
    set "WSLPATH=!WSLPATH:\=/!"
    set "PYD=!PY_CMD:wsl:=!"
    wsl -d !PYD! -- bash -c "export PYTHONIOENCODING=gbk && cd '!WSLPATH!' && python3 判路_v40终版.py 输出/road.mp4 输出/shot.png 输出/result"
)
if not errorlevel 1 goto :ok
echo.
echo [错误] 判路失败！看上方报错信息。
pause
exit /b 1
:ok
echo.

REM ---------- 6. 结果 ----------
echo [5/5] 完成！
set "TS=%date:~0,4%%date:~5,2%%date:~8,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "TS=%TS: =0%"
copy /y "%OUT%\result_roads.png" "%OUT%\历史结果\result_%TS%.png" >nul
copy /y "%OUT%\result_roads.json" "%OUT%\历史结果\result_%TS%.json" >nul
echo.
echo 结果文件（输出 文件夹下）:
echo    输出\result_roads.png   路线图（绿=路 红=断 橙圈=可行动节点）
echo    输出\result_roads.json  路网数据（节点/边/亮节点/你的位置）
echo    输出\历史结果\       按时间戳存档
echo.
start "" "%OUT%\result_roads.png"
start "" "%OUT%\result_roads.json"
pause

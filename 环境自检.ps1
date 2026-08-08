﻿﻿# Black Flow Sea route recognition - Environment checker
# Auto-detect: MuMu install dir / adb port / Python (Windows or WSL)
# Writes 环境配置.ini (env config) for 一条龙.bat (one-click script)
$ErrorActionPreference = 'SilentlyContinue'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "============================================"
Write-Host "  Black Flow Sea Route Recognition"
Write-Host "  Environment Check"
Write-Host "============================================"
Write-Host ""

# ---------- 1. Find MuMu install dir ----------
Write-Host "[1/4] Looking for MuMu emulator..."
$candidates = @()
$regPaths = @(
    'HKLM:\SOFTWARE\Netease\MuMuPlayer',
    'HKLM:\SOFTWARE\WOW6432Node\Netease\MuMuPlayer',
    'HKCU:\SOFTWARE\Netease\MuMuPlayer'
)
foreach ($rp in $regPaths) {
    if (Test-Path $rp) {
        $p = (Get-ItemProperty $rp).InstallPath
        if ($p) { $candidates += $p }
    }
}
$common = @(
    "$env:ProgramFiles\Netease\MuMuPlayer",
    "${env:ProgramFiles(x86)}\Netease\MuMuPlayer",
    "$env:LOCALAPPDATA\Netease\MuMuPlayer",
    'D:\mumu\MuMuPlayer', 'C:\mumu\MuMuPlayer',
    'D:\Program Files\Netease\MuMuPlayer',
    'C:\Program Files\Netease\MuMuPlayer'
)
foreach ($p in $common) { if (Test-Path $p) { $candidates += $p } }

$mumuDir = $null
foreach ($c in ($candidates | Select-Object -Unique)) {
    if (Test-Path "$c\nx_main\adb.exe") { $mumuDir = $c; break }
}
if ($mumuDir) {
    Write-Host "  OK  MuMu: $mumuDir"
} else {
    Write-Host "  !!  MuMu not found (looking for nx_main\adb.exe)"
}

# ---------- 2. Read adb ports ----------
Write-Host "[2/4] Reading MuMu adb port..."
$ports = @()
if ($mumuDir) {
    Get-ChildItem "$mumuDir\vms" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        $cfg = Join-Path $_.FullName "configs\vm_config.json"
        if (Test-Path $cfg) {
            try {
                $j = Get-Content $cfg -Raw -Encoding UTF8 | ConvertFrom-Json
                $p = $j.vm.nat.port_forward.adb.host_port
                if ($p) { $ports += [string]$p }
            } catch {}
        }
    }
}
$ports = $ports | Select-Object -Unique
if ($ports) {
    Write-Host "  OK  adb ports: $($ports -join ', ')"
} else {
    Write-Host "  !!  no adb port found, will try defaults 16416 (MuMu15) / 16384 (MuMu12)"
    $ports = @('16416', '16384')
}

# ---------- 3. Python check (Windows native first, then WSL) ----------
Write-Host "[3/4] Checking Python environment..."
$pyCmd = $null

# 3a. Windows Python
$pyWin = $null
foreach ($cand in @('python', 'py')) {
    $probe = & $cand -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $probe) { $pyWin = $cand; break }
}
if ($pyWin) {
    $cv2ok = & $pyWin -c "import cv2, numpy; print('ok')" 2>$null
    if ($LASTEXITCODE -eq 0 -and $cv2ok -match 'ok') {
        $pyCmd = "python"
        Write-Host "  OK  Windows Python + OpenCV ready"
    } else {
        Write-Host "  !!  Windows Python found but missing OpenCV/numpy"
        Write-Host "      fix: pip install opencv-python numpy"
    }
}

# 3b. WSL Python
if (-not $pyCmd) {
    # wsl.exe 输出 UTF-16LE，PS5.1 默认 ANSI 解码会乱 → 先切编码
    $oldEnc = [Console]::OutputEncoding
    [Console]::OutputEncoding = [System.Text.Encoding]::Unicode
    $wslOut = & wsl -l -q 2>$null | Out-String
    [Console]::OutputEncoding = $oldEnc
    $distros = ($wslOut -split "`r?`n") | ForEach-Object {
        $_.Trim().Replace([string][char]0xFEFF, '')
    } | Where-Object { $_ }
    Write-Host "  WSL distros: $($distros -join ', ')"
    foreach ($d in $distros) {
        $probe = & wsl -d $d -- python3 -c "import cv2, numpy; print('ok')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe -match 'ok') {
            $pyCmd = "wsl:$d"
            Write-Host "  OK  WSL($d) + Python + OpenCV ready"
            break
        }
    }
    if (-not $pyCmd) {
        Write-Host "  !!  no usable python3+cv2+numpy in WSL"
        Write-Host "      fix: sudo apt install python3-opencv python3-numpy"
    }
}

if (-not $pyCmd) {
    Write-Host ""
    Write-Host "  [FATAL] No usable Python environment!"
    Write-Host "    Option A: install Windows Python + pip install opencv-python numpy"
    Write-Host "    Option B: enable WSL + sudo apt install python3-opencv"
    Write-Host ""
}

# ---------- 4. Write config ----------
$cfgPath = Join-Path $here "env.ini"
$lines = @(
    "# Black Flow Sea recognition env config (auto-generated)",
    "MUMU_DIR=$mumuDir",
    "ADB_PORTS=$($ports -join ',')",
    "PY_CMD=$pyCmd"
)
[System.IO.File]::WriteAllLines($cfgPath, $lines, (New-Object System.Text.UTF8Encoding $false))
Write-Host "[4/4] Config written: env.ini"
Write-Host ""
Write-Host "Done! Now run 一条龙.bat (one-click)."
Write-Host ""

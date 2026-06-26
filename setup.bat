@echo off
setlocal EnableDelayedExpansion

REM == DEEPX Setup: Download sample models and videos ==
REM Delegates to scripts\setup_assets.py

pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%"

REM ---------- Find Python ----------
set "PYTHON_EXE="

REM 1) Explicit Python path from environment
if defined PYTHON_PATH (
    set "PYTHON_EXE=%PYTHON_PATH:"=%"
    goto :python_found
)

REM 2) Project venv
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
    goto :python_found
)

REM 3) PATH-available python
where python >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=python"
    goto :python_found
)

REM 4) py launcher
where py >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=py"
    goto :python_found
)

REM 5) Common install locations
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if not defined PYTHON_EXE if exist %%~P set "PYTHON_EXE=%%~P"
)
if defined PYTHON_EXE goto :python_found

echo [ERR] Python not found.
echo       Please install Python: https://www.python.org/downloads/
echo       Or set PYTHON_PATH to a working python.exe.
goto :err

:python_found
echo [OK] Python: %PYTHON_EXE%

REM Verify Python works
"%PYTHON_EXE%" --version >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    if defined PYTHON_PATH (
        echo [ERR] PYTHON_PATH is set but Python failed to execute: %PYTHON_EXE%
        echo       Please set PYTHON_PATH to a working python.exe.
        goto :err
    )
    echo [WARN] Python found at %PYTHON_EXE% but failed to execute. Searching further...
    set "PYTHON_EXE="
    for %%P in (
        "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
        "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
        "%ProgramFiles%\Python312\python.exe"
        "%ProgramFiles%\Python311\python.exe"
        "C:\Python312\python.exe"
        "C:\Python311\python.exe"
        "C:\Python310\python.exe"
    ) do (
        if not defined PYTHON_EXE if exist %%~P (
            "%%~P" --version >nul 2>nul
            if !ERRORLEVEL! EQU 0 set "PYTHON_EXE=%%~P"
        )
    )
    if not defined PYTHON_EXE (
        where py >nul 2>nul
        if !ERRORLEVEL! EQU 0 (
            py --version >nul 2>nul
            if !ERRORLEVEL! EQU 0 set "PYTHON_EXE=py"
        )
    )
    if not defined PYTHON_EXE (
        where python >nul 2>nul
        if !ERRORLEVEL! EQU 0 (
            python --version >nul 2>nul
            if !ERRORLEVEL! EQU 0 set "PYTHON_EXE=python"
        )
    )
    if not defined PYTHON_EXE (
        echo [ERR] Python not found.
        echo       Please install Python: https://www.python.org/downloads/
        echo       Or set PYTHON_PATH to a working python.exe.
        goto :err
    )
)

REM ---------- Ensure requests is available ----------
"%PYTHON_EXE%" -c "import requests" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo [..] Installing required dependency: requests
    "%PYTHON_EXE%" -m pip install --quiet requests
    if !ERRORLEVEL! NEQ 0 (
        echo [WARN] pip install requests failed. Setup may not work.
    )
)

REM ---------- Run setup script ----------
if not exist "%PROJECT_ROOT%\scripts\setup_assets.py" (
    echo [ERR] scripts\setup_assets.py not found.
    goto :err
)

"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\setup_assets.py" %*
set "RC=!ERRORLEVEL!"

popd >nul

if !RC! NEQ 0 (
    echo.
    echo [ERR] Setup failed with exit code !RC!.
)

if not defined DISABLE_BUILD_PAUSE pause
exit /b !RC!

:err
popd >nul 2>nul
if not defined DISABLE_BUILD_PAUSE pause
exit /b 1

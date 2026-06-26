@echo off
setlocal EnableDelayedExpansion

pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%"

REM ---------- Find Python ----------
set "PYTHON_EXE="

REM 1) Explicit Python path from environment
if defined PYTHON_PATH (
    set "PYTHON_EXE=%PYTHON_PATH:"=%"
    goto :verify
)

REM 2) Project venv
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
    goto :verify
)

REM 3) Common install locations first (most reliable)
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
if defined PYTHON_EXE goto :verify

REM 4) py launcher
where py >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=py"
    goto :verify
)

REM 5) PATH python (last - Windows Store stub can fool 'where')
where python >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=python"
    goto :verify
)

echo [ERR] Python not found.
echo       Please install Python: https://www.python.org/downloads/
echo       Or set PYTHON_PATH to a working python.exe.
pause
exit /b 1

:verify
REM Make sure the found Python actually executes
"%PYTHON_EXE%" --version >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    if defined PYTHON_PATH (
        echo [ERR] PYTHON_PATH is set but Python failed to execute: %PYTHON_EXE%
        echo       Please set PYTHON_PATH to a working python.exe.
        pause
        exit /b 1
    )
    echo [WARN] %PYTHON_EXE% found but does not work. Searching further...
    set "PYTHON_EXE="
    REM Try remaining options
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
        pause
        exit /b 1
    )
)

:run
"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\run_demo.py" %*
set "RC=!ERRORLEVEL!"

popd >nul

if !RC! NEQ 0 (
    echo.
    echo Demo exited with code !RC!.
)
if not defined DISABLE_BUILD_PAUSE pause
exit /b !RC!

@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%D in ("%SCRIPT_DIR%..") do set "PROJECT_ROOT=%%~fD"
set "PYTHON_SCRIPT=%SCRIPT_DIR%extract_sln_package.py"

REM ---------- Find Python ----------
set "PYTHON_EXE="

REM 1) Project venv
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"
    goto :verify
)

REM 2) Common install locations first (most reliable)
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
    if not defined PYTHON_EXE if exist "%%~P" set "PYTHON_EXE=%%~P"
)
if defined PYTHON_EXE goto :verify

REM 3) py launcher
where py >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=py"
    goto :verify
)

REM 4) PATH python (last - Windows Store stub can fool 'where')
where python >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=python"
    goto :verify
)

echo [DXAPP] [ERROR] Python not found. Please install Python or run build.bat first.
echo                 https://www.python.org/downloads/
exit /b 1

:verify
"%PYTHON_EXE%" --version >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo [DXAPP] [WARN] %PYTHON_EXE% found but does not work. Searching further...
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
        if not defined PYTHON_EXE if exist "%%~P" (
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
        echo [DXAPP] [ERROR] No working Python found. Please install Python or run build.bat first.
        echo                 https://www.python.org/downloads/
        exit /b 1
    )
)

"%PYTHON_EXE%" "%PYTHON_SCRIPT%" %*
exit /b !ERRORLEVEL!
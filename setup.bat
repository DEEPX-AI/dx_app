@echo off
setlocal EnableDelayedExpansion

REM ── DEEPX Setup: Download sample models and videos ──
REM Delegates to scripts\setup_assets.py (cross-platform Python)

pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%"

REM Resolve Python: prefer venv, fall back to system python
set "PYTHON_EXE=python"
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%\venv\Scripts\python.exe"

REM Ensure requests is available
"%PYTHON_EXE%" -c "import requests" >nul 2>&1
if !ERRORLEVEL! NEQ 0 (
    echo Installing required dependency: requests
    "%PYTHON_EXE%" -m pip install --quiet requests
)

REM Forward all arguments to the Python setup script
"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\setup_assets.py" %*
set "RC=!ERRORLEVEL!"

popd >nul

if !RC! NEQ 0 echo Setup failed with exit code !RC!.

REM Pause for interactive users (CI sets DISABLE_BUILD_PAUSE=1)
if not defined DISABLE_BUILD_PAUSE pause

exit /b !RC!

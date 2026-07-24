@echo off
REM =========================================================================
REM Windows Test Runner for DX-APP
REM
REM Runs C++, Python, and pybinding invalid argument tests.
REM
REM Prerequisites:
REM   1. setup.bat   (downloads models and videos)
REM   2. build.bat   (compiles C++ executables + installs dx_postprocess)
REM
REM Usage:
REM   run_tests.bat              Run all tests
REM   run_tests.bat cpp          Run C++ tests only
REM   run_tests.bat python       Run Python script tests only
REM   run_tests.bat pybinding    Run pybinding tests only
REM   run_tests.bat smoke        Run smoke subset (fast)
REM =========================================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..

REM --- Find Python ---
if defined PYTHON_PATH (
    set "PYTHON=%PYTHON_PATH%"
    goto :found_python
)
if exist "%PROJECT_ROOT%\venv\Scripts\python.exe" (
    set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
    goto :found_python
)
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python"
    goto :found_python
)
echo [ERR] Python not found. Set PYTHON_PATH or create a venv.
exit /b 1

:found_python

REM --- Ensure pytest is available ---
%PYTHON% -m pytest --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing pytest...
    %PYTHON% -m pip install pytest --quiet
)

REM --- Parse arguments ---
set "MARKER="
set "TEST_FILE="

if "%~1"=="" goto :run_all
if /i "%~1"=="cpp" goto :run_cpp
if /i "%~1"=="python" goto :run_python
if /i "%~1"=="pybinding" goto :run_pybinding
if /i "%~1"=="smoke" goto :run_smoke
goto :usage

:run_all
echo.
echo  Running ALL Windows tests...
echo.
goto :execute

:run_cpp
echo.
echo  Running C++ invalid argument tests...
echo.
set "MARKER=-m cpp"
set "TEST_FILE=test_cpp_invalid_args.py"
goto :execute

:run_python
echo.
echo  Running Python script invalid argument tests...
echo.
set "MARKER=-m python_script"
set "TEST_FILE=test_python_invalid_args.py"
goto :execute

:run_pybinding
echo.
echo  Running pybinding invalid argument tests...
echo.
set "MARKER=-m pybinding"
set "TEST_FILE=test_pybinding_invalid_args.py"
goto :execute

:run_smoke
echo.
echo  Running smoke tests (fast subset)...
echo.
set "MARKER=-m smoke"
goto :execute

:usage
echo.
echo  Usage: run_tests.bat [cpp / python / pybinding / smoke]
echo.
exit /b 1

:execute
pushd "%SCRIPT_DIR%"

if defined TEST_FILE (
    %PYTHON% -m pytest %TEST_FILE% %MARKER% --tb=short -v %2 %3 %4 %5
) else (
    %PYTHON% -m pytest . %MARKER% --tb=short -v %2 %3 %4 %5
)

set EXIT_CODE=%errorlevel%
popd

echo.
if %EXIT_CODE% equ 0 (
    echo  [PASS] All tests passed.
) else (
    echo  [FAIL] Some tests failed. Exit code: %EXIT_CODE%
)

exit /b %EXIT_CODE%

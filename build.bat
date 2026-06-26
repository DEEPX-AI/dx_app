@echo off
setlocal EnableDelayedExpansion

REM ============================================================
REM  build.bat - One-click build script with auto-recovery
REM ============================================================

pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%\\"
set "STEP="

echo.
echo ============================================================
echo  DX App Build
echo ============================================================
echo.

REM ---------- STEP 1: Find Python ----------
set "STEP=Python detection"
set "PYTHON_EXE="

REM 1a) Explicit Python path from environment
if defined PYTHON_PATH (
    set "PYTHON_EXE=%PYTHON_PATH:"=%"
    goto :python_verify
)

REM 1b) Project venv
if exist "%PROJECT_ROOT%venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"
    goto :python_verify
)

REM 1c) Common install locations first (most reliable)
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
if defined PYTHON_EXE goto :python_verify

REM 1d) py launcher
where py >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=py"
    goto :python_verify
)

REM 1e) PATH python (last - Windows Store stub can fool 'where')
where python >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "PYTHON_EXE=python"
    goto :python_verify
)

goto :python_missing

:python_verify
REM Verify Python actually works before using it.
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
    if not defined PYTHON_EXE goto :python_missing
)
goto :python_found

:python_missing
echo [ERR] Python not found.
echo       Please install Python: https://www.python.org/downloads/
echo       Or set PYTHON_PATH to a working python.exe.
goto :err

:python_found
echo [OK] Python: %PYTHON_EXE%

REM ---------- STEP 2: Ensure pip is available ----------
set "STEP=pip check"
"%PYTHON_EXE%" -m pip --version >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo [..] pip not found. Installing pip...
    "%PYTHON_EXE%" -m ensurepip --upgrade >nul 2>nul
    if !ERRORLEVEL! NEQ 0 (
        echo [WARN] pip is not available. Python module install step may fail.
    ) else (
        echo [OK] pip installed.
    )
)

REM ---------- STEP 3: Check cmake availability ----------
set "STEP=cmake check"
set "CMAKE_EXE="
where cmake >nul 2>nul
if !ERRORLEVEL! EQU 0 (
    set "CMAKE_EXE=cmake"
    goto :cmake_found
)

REM Try VS 2022 default locations
set "VS_CMAKE_BASE=%ProgramFiles%\Microsoft Visual Studio\2022"
for %%E in (Community Professional Enterprise) do (
    if not defined CMAKE_EXE (
        set "VS_CMAKE_TRY=!VS_CMAKE_BASE!\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        if exist "!VS_CMAKE_TRY!" set "CMAKE_EXE=!VS_CMAKE_TRY!"
    )
)

:cmake_found
if defined CMAKE_EXE (
    echo [OK] CMake: %CMAKE_EXE%
) else (
    echo [WARN] cmake not found. The build may fail.
    echo        Install Visual Studio with C++ workload or CMake standalone.
)

REM ---------- STEP 4: Check git ----------
set "STEP=git check"
where git >nul 2>nul
if !ERRORLEVEL! NEQ 0 (
    echo [WARN] git not found. vcpkg installation requires git.
    echo        https://git-scm.com/download/win
)

REM ---------- STEP 5: VCPKG handling ----------
set "STEP=vcpkg setup"
if defined VCPKG_ROOT (
    echo [..] VCPKG_ROOT: %VCPKG_ROOT%
    if exist "%VCPKG_ROOT%\vcpkg.exe" (
        echo [OK] vcpkg found.
    ) else (
        echo [WARN] VCPKG_ROOT is set but vcpkg.exe not found. Trying project-local...
        set "VCPKG_ROOT="
    )
)

if not defined VCPKG_ROOT (
    if exist "%PROJECT_ROOT%vcpkg\vcpkg.exe" (
        set "VCPKG_ROOT=%PROJECT_ROOT%vcpkg"
        echo [OK] Using existing project-local vcpkg.
    ) else (
        echo [..] Installing project-local vcpkg...
        if not exist "%PROJECT_ROOT%scripts\install_vcpkg.ps1" (
            echo [ERR] scripts\install_vcpkg.ps1 not found.
            goto :err
        )
        powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\install_vcpkg.ps1" -Install
        if !ERRORLEVEL! NEQ 0 (
            echo [ERR] vcpkg installation failed. Check network and git.
            goto :err
        )
        if exist "%PROJECT_ROOT%vcpkg\vcpkg.exe" (
            set "VCPKG_ROOT=%PROJECT_ROOT%vcpkg"
            echo [OK] vcpkg installed.
        ) else (
            echo [ERR] vcpkg install script ran but vcpkg.exe not found.
            goto :err
        )
    )
)

set "CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
if not exist "%CMAKE_TOOLCHAIN_FILE%" (
    echo [ERR] vcpkg toolchain file not found: %CMAKE_TOOLCHAIN_FILE%
    goto :err
)
echo [OK] Toolchain: %CMAKE_TOOLCHAIN_FILE%

REM Build scope selection: all (default), minimal (run_demo C++), category (single category targets)
if not defined BUILD_SCOPE (
    echo Select build scope:
    echo   1. All build
    echo   2. Minimal build (run_demo C++ apps)
    echo   3. Category build (type "list" after selecting to show categories)
    set /p BUILD_SCOPE_CHOICE=Select build scope [1-3]:
    if "!BUILD_SCOPE_CHOICE!"=="" (
        echo [DXAPP] [ERROR] Build scope selection is required.
        goto :err
    )
    if "!BUILD_SCOPE_CHOICE!"=="1" set "BUILD_SCOPE=all"
    if "!BUILD_SCOPE_CHOICE!"=="2" set "BUILD_SCOPE=minimal"
    if "!BUILD_SCOPE_CHOICE!"=="3" set "BUILD_SCOPE=category"
    if not defined BUILD_SCOPE (
        echo [DXAPP] [ERROR] Invalid build scope selection: !BUILD_SCOPE_CHOICE!
        goto :err
    )
)

REM Validate BUILD_SCOPE using whitelist flag
set "VALID_SCOPE=0"
if /I "!BUILD_SCOPE!"=="all" set "VALID_SCOPE=1"
if /I "!BUILD_SCOPE!"=="minimal" set "VALID_SCOPE=1"
if /I "!BUILD_SCOPE!"=="category" set "VALID_SCOPE=1"
if "!VALID_SCOPE!"=="0" (
    echo [DXAPP] [ERROR] Invalid BUILD_SCOPE: !BUILD_SCOPE!
    goto :err
)

if /I "%BUILD_SCOPE%"=="category" if not defined BUILD_CATEGORY (
    set /p BUILD_CATEGORY=Enter category name, or "list" to show categories: 
)

REM Check for empty BUILD_CATEGORY using delayed expansion
if /I "%BUILD_SCOPE%"=="category" if "!BUILD_CATEGORY!"=="" (
    echo [DXAPP] [ERROR] BUILD_CATEGORY is required when BUILD_SCOPE=category.
    goto :err
)

REM Invoke generator directly per scope to properly pass category as quoted delayed expansion
if /I "%BUILD_SCOPE%"=="all" (
    "%PYTHON_EXE%" .\scripts\generate_build_bat.py --run
) else if /I "%BUILD_SCOPE%"=="minimal" (
    "%PYTHON_EXE%" .\scripts\generate_build_bat.py --run --minimal
) else if /I "%BUILD_SCOPE%"=="category" (
    "%PYTHON_EXE%" .\scripts\generate_build_bat.py --run --category "!BUILD_CATEGORY!"
)
if %ERRORLEVEL% NEQ 0 goto :err

REM Install bin copy logic only runs for all build scope
if /I "%BUILD_SCOPE%"=="all" (
    REM Determine install directory: use BUILD_CONFIG env or default x64-Release
    if defined BUILD_CONFIG (
        set "CFG_NAME=%BUILD_CONFIG%"
    ) else (
        set "CFG_NAME=x64-Release"
    )
    set "INSTALL_DIR=%PROJECT_ROOT%out\install\!CFG_NAME!"
    echo Will copy installed binaries from: !INSTALL_DIR!\bin

    REM Derive short config name (Release/Debug) from CFG_NAME if it contains a dash
    set "CFG_SHORT=!CFG_NAME!"
    for /f "tokens=2 delims=-" %%A in ("!CFG_NAME!") do set "CFG_SHORT=%%A"

    REM Prefer per-config subfolder (e.g., bin\Release) then plain bin
    if exist "!INSTALL_DIR!\bin\!CFG_SHORT!" (
        set "SRC_BIN=!INSTALL_DIR!\bin\!CFG_SHORT!"
    ) else if exist "!INSTALL_DIR!\bin" (
        set "SRC_BIN=!INSTALL_DIR!\bin"
    ) else (
        set "SRC_BIN="
    )

    if defined SRC_BIN (
        echo Copying from !SRC_BIN! to %PROJECT_ROOT%bin
        mkdir "%PROJECT_ROOT%bin" >nul 2>nul
        xcopy /E /Y "!SRC_BIN!\*" "%PROJECT_ROOT%bin\" >nul
        echo Copied installed binaries to %PROJECT_ROOT%bin
    ) else (
        echo No bin directory found to copy from (checked !INSTALL_DIR!\bin\!CFG_SHORT! and !INSTALL_DIR!\bin)
    )
) else (
    echo Skipping install bin copy for target-only build scope: %BUILD_SCOPE%
)

REM Build and install dx_postprocess Python bindings when demo modes need it
set "INSTALL_DX_POSTPROCESS=0"
if /I "%BUILD_SCOPE%"=="all" set "INSTALL_DX_POSTPROCESS=1"
if /I "%BUILD_SCOPE%"=="minimal" set "INSTALL_DX_POSTPROCESS=1"
if "%INSTALL_DX_POSTPROCESS%"=="1" (
    if exist "%PROJECT_ROOT%build_env.bat" (
        echo Loading DXRT environment from build_env.bat
        call "%PROJECT_ROOT%build_env.bat"
        echo Loaded DXRT environment from build_env.bat
    ) else (
        echo build_env.bat not found; proceeding without DXRT exports.
    )

    set "MODULE_DIR=%PROJECT_ROOT%src\bindings\python\dx_postprocess"
    echo Checking module directory: !MODULE_DIR!
    if exist "!MODULE_DIR!" (
        echo Installing dx_postprocess Python module...
        pushd "!MODULE_DIR!" >nul
        set "SKBUILD_CMAKE_ARGS=-DCMAKE_BUILD_TYPE=Release"
        set "SKBUILD_INSTALL_STRIP=true"
        echo Using Python executable: %PYTHON_EXE%
        "%PYTHON_EXE%" -m pip install .
        set "PIP_RC=!ERRORLEVEL!"
        popd >nul
        if !PIP_RC! NEQ 0 goto :err
    ) else (
        echo Module directory does not exist, skipping dx_postprocess installation.
    )
) else (
    echo Skipping dx_postprocess installation for build scope: %BUILD_SCOPE%
)

REM ---------- DONE ----------
popd >nul

echo.
echo ============================================================
echo  BUILD SUCCESSFUL
echo ============================================================
echo.

if not defined DISABLE_BUILD_PAUSE (
    echo Press any key to close...
    pause >nul
)
exit /b 0

REM ---------- ERROR HANDLER ----------
:err
echo.
echo ============================================================
echo  BUILD FAILED at step: %STEP%
echo ============================================================
echo.
echo  Troubleshooting:
echo    1. Install Python: https://python.org/downloads
echo    2. Install Visual Studio 2022 with C++ workload
echo    3. Install git: https://git-scm.com
echo    4. Check network connection for vcpkg
echo    5. Try running as Administrator
echo.

popd >nul 2>nul

if not defined DISABLE_BUILD_PAUSE (
    echo Press any key to close...
    pause >nul
)
exit /b 1

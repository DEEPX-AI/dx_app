@echo off
setlocal EnableDelayedExpansion

REM Resolve project root and prefer venv Python when available
pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%\\"
set "PYTHON_EXE=python"
if exist "%PROJECT_ROOT%venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"

REM Remove existing bin folder so we can overwrite cleanly
if exist "%PROJECT_ROOT%bin" (
    echo Removing existing %PROJECT_ROOT%bin ...
    rmdir /s /q "%PROJECT_ROOT%bin"
)

REM VCPKG handling: priority to system/user VCPKG_ROOT. If not defined, attempt project-local installation automatically.
if defined VCPKG_ROOT (
    echo VCPKG_ROOT is defined: %VCPKG_ROOT%
    if exist "%VCPKG_ROOT%\vcpkg.exe" (
        echo vcpkg found at %VCPKG_ROOT%\vcpkg.exe
    ) else (
        echo ERROR: VCPKG_ROOT is defined but vcpkg.exe not found at %VCPKG_ROOT%\vcpkg.exe
        echo Please install vcpkg at that location or unset VCPKG_ROOT and install project-local vcpkg using:
        echo   powershell -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\install_vcpkg.ps1" -Install
        goto :err
    )
) else (
    echo VCPKG_ROOT is not defined. Attempting to install project-local vcpkg into %PROJECT_ROOT%vcpkg...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%PROJECT_ROOT%scripts\install_vcpkg.ps1" -Install
    if %ERRORLEVEL% NEQ 0 (
        echo vcpkg installer failed or was declined.
        goto :err
    )
    if exist "%PROJECT_ROOT%vcpkg\vcpkg.exe" (
        set "VCPKG_ROOT=%PROJECT_ROOT%vcpkg"
        echo Using project-local vcpkg at %VCPKG_ROOT%
    ) else (
        echo Installer did not produce project-local vcpkg; aborting.
        goto :err
    )
)

set "CMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
echo Using CMAKE_TOOLCHAIN_FILE=%CMAKE_TOOLCHAIN_FILE%

"%PYTHON_EXE%" .\scripts\generate_build_bat.py --run
if %ERRORLEVEL% NEQ 0 goto :err

REM Determine install directory: use BUILD_CONFIG env or default x64-Release
if defined BUILD_CONFIG (
    set "CFG_NAME=%BUILD_CONFIG%"
) else (
    set "CFG_NAME=x64-Release"
)
set "INSTALL_DIR=%PROJECT_ROOT%out\install\%CFG_NAME%"
echo Will copy installed binaries from: %INSTALL_DIR%\bin

REM Derive short config name (Release/Debug) from CFG_NAME if it contains a dash
set "CFG_SHORT=%CFG_NAME%"
for /f "tokens=2 delims=-" %%A in ("%CFG_NAME%") do set "CFG_SHORT=%%A"

REM Prefer per-config subfolder (e.g., bin\Release) then plain bin
if exist "%INSTALL_DIR%\bin\%CFG_SHORT%" (
    set "SRC_BIN=%INSTALL_DIR%\bin\%CFG_SHORT%"
) else if exist "%INSTALL_DIR%\bin" (
    set "SRC_BIN=%INSTALL_DIR%\bin"
) else (
    set "SRC_BIN="
)

if defined SRC_BIN (
    echo Copying from %SRC_BIN% to %PROJECT_ROOT%bin
    mkdir "%PROJECT_ROOT%bin" >nul 2>nul
    xcopy /E /Y "%SRC_BIN%\*" "%PROJECT_ROOT%bin\" >nul
    echo Copied installed binaries to %PROJECT_ROOT%bin
) else (
    echo No bin directory found to copy from (checked %INSTALL_DIR%\bin\%CFG_SHORT% and %INSTALL_DIR%\bin)
)

REM Build and install dx_postprocess Python bindings (skbuild)
if exist "%PROJECT_ROOT%build_env.bat" (
    echo Loading DXRT environment from build_env.bat
    call "%PROJECT_ROOT%build_env.bat"
    echo Loaded DXRT environment from build_env.bat
) else (
    echo build_env.bat not found; proceeding without DXRT exports.
)

set "MODULE_DIR=%PROJECT_ROOT%src\bindings\python\dx_postprocess"
echo Checking module directory: %MODULE_DIR%
if exist "%MODULE_DIR%" (
    echo Installing dx_postprocess Python module...
    pushd "%MODULE_DIR%" >nul
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

popd >nul

REM Default behavior: pause at the end to keep window open for debugging. Set DISABLE_BUILD_PAUSE=1 to disable (CI).
if not defined DISABLE_BUILD_PAUSE (
    echo Build finished. Press any key to continue...
    pause >nul
)

exit /b 0

:err
popd >nul

echo Build aborted with exit code %ERRORLEVEL%.
if not defined DISABLE_BUILD_PAUSE (
    echo Press any key to continue (error)...
    pause >nul
)
exit /b %ERRORLEVEL%
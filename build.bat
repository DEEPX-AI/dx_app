@echo off
setlocal EnableDelayedExpansion

REM Resolve project root and prefer venv Python when available
pushd "%~dp0" >nul
set "PROJECT_ROOT=%cd%\"
set "PYTHON_EXE=python"
if exist "%PROJECT_ROOT%venv\Scripts\python.exe" set "PYTHON_EXE=%PROJECT_ROOT%venv\Scripts\python.exe"

"%PYTHON_EXE%" .\scripts\generate_build_bat.py --run
if %ERRORLEVEL% NEQ 0 goto :err

REM Build and install dx_postprocess Python bindings (skbuild)
if exist "%PROJECT_ROOT%build_env.bat" (
	call "%PROJECT_ROOT%build_env.bat"
	echo Loaded DXRT environment from build_env.bat
) else (
	echo build_env.bat not found; proceeding without DXRT exports.
)

call "%PROJECT_ROOT%build_env.bat"
echo Loaded DXRT environment from build_env.bat
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
exit /b 0

:err
popd >nul
exit /b %ERRORLEVEL%
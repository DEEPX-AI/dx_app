@echo off
setlocal
REM Build script for dxapp (Windows, MSVC, Ninja)

REM Check required environment variables
IF NOT DEFINED DXRT_DIR (
    echo ERROR: DXRT_DIR environment variable is not set.
    pause
    exit /b 1
)
IF NOT EXIST "%DXRT_DIR%" (
    echo ERROR: DXRT_DIR directory does not exist: %DXRT_DIR%
    pause
    exit /b 1
)

REM Set build directory
set BUILD_DIR=build_vs2022

REM Create build directory if it doesn't exist
IF NOT EXIST "%BUILD_DIR%" (
    mkdir "%BUILD_DIR%"
)

REM Configure with CMake (Visual Studio 2022 generator, MSVC, C++14)
cmake -S . -B "%BUILD_DIR%" ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_CXX_STANDARD=14 ^
    -DDXRT_DIR="%DXRT_DIR%" ^
    -DOpenCV_DIR="%OpenCV_DIR%"

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: CMake configuration failed.
    pause
    exit /b 1
)

REM Build with MSBuild (Visual Studio solution)
cmake --build "%BUILD_DIR%" --config Release

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed.
    pause
    exit /b 1
)

REM Install the project
cmake --install "%BUILD_DIR%" --config Release

IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: Install failed.
    pause
    exit /b 1
)

echo Visual Studio solution generated at %BUILD_DIR%\dx_app.sln
echo Build and install completed successfully.
pause
endlocal

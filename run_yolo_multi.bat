@echo off
setlocal

set "OPENCV_LIB_PATH=%~dp0\vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=%~dp0\vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_JSON_PATH=%~dp0\example\yolo_multi_demo.json"

start cmd /K "%~dp0\bin\yolo_multi.exe" -c %APP_JSON_PATH% -t

endlocal
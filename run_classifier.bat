@echo off
setlocal

set "OPENCV_LIB_PATH=%~dp0\vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=%~dp0\vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_JSON_PATH=%~dp0\run_classifier\example\imagenet_example.json"

start cmd /K "%~dp0\bin\run_classifier.exe" -c %APP_JSON_PATH%

endlocal
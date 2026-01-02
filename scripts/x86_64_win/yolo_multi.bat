@echo off
setlocal
pushd ..\..\
set "OPENCV_LIB_PATH=vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_JSON_PATH=example\yolo_multi\yolo_multi_demo.json"

start cmd /K "bin\yolo_multi.exe" -c %APP_JSON_PATH%
popd
endlocal
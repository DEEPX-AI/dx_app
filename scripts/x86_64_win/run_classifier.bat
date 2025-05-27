@echo off
setlocal
pushd ..\..\
set "OPENCV_LIB_PATH=vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_JSON_PATH=example\run_classifier\imagenet_example.json"

start cmd /K "bin\run_classifier.exe" -c %APP_JSON_PATH%
popd
endlocal
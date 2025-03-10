@echo off
setlocal

set "OPENCV_LIB_PATH=%~dp0\vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=%~dp0\vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL_PATH=%~dp0\..\models\YoloV7.dxnn"
set "APP_CONFIG_PARAM=4"
set "APP_VIDEO_PATH=%~dp0\..\videos\dogs.mp4"

start cmd /K "%~dp0\bin\yolo.exe" -m %APP_MODEL_PATH% -p %APP_CONFIG_PARAM% -v %APP_VIDEO_PATH%

endlocal
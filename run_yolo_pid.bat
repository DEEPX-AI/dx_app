@echo off
setlocal

set "OPENCV_LIB_PATH=%~dp0\vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=%~dp0\vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL0_PATH=%~dp0\assets\models\YOLOV5S_3.dxnn"
set "APP_MODEL1_PATH=%~dp0\assets\models\DeepLabV3PlusMobileNetV2_2.dxnn"
set "APP_VIDEO_PATH=%~dp0\assets\videos\blackbox-city-road.mp4"

start cmd /K "%~dp0\bin\od_pid.exe" -m0 %APP_MODEL0_PATH% -m1 %APP_MODEL1_PATH% -v %APP_VIDEO_PATH%

endlocal
@echo off
setlocal
pushd ..\..\
set "OPENCV_LIB_PATH=vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL0_PATH=assets\models\YOLOV5S_3.dxnn"
set "APP_MODEL0_PARAM=1"
set "APP_MODEL1_PATH=assets\models\DeepLabV3PlusMobileNetV2_2.dxnn"
set "APP_MODEL1_PARAM=0"
set "APP_VIDEO_PATH=assets\videos\blackbox-city-road.mp4"

start cmd /K "bin\od_segmentation.exe" -m0 %APP_MODEL0_PATH% -p0 %APP_MODEL0_PARAM% -m1 %APP_MODEL1_PATH% -p1 %APP_MODEL1_PARAM% -v %APP_VIDEO_PATH%

endlocal
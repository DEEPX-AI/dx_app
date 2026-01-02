@echo off
setlocal
pushd ..\..\
set "OPENCV_LIB_PATH=vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL_PATH=assets\models\YOLOV5Pose640_1.dxnn"
set "APP_CONFIG_PARAM=0"
set "APP_VIDEO_PATH=assets\videos\dance-group2.mov"

start cmd /K "bin\pose.exe" -m %APP_MODEL_PATH% -p %APP_CONFIG_PARAM% -v %APP_VIDEO_PATH%
popd
endlocal
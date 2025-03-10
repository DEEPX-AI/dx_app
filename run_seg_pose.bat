@echo off
setlocal

set "OPENCV_LIB_PATH=%~dp0\vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=%~dp0\vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL0_PATH=%~dp0\..\models\YOLOV5Pose640_1.dxnn"
set "APP_MODEL1_PATH=%~dp0\..\models\HyundaiDDRNet_1.dxnn"
set "APP_VIDEO_PATH=%~dp0\..\videos\dance-group2.mov"

start cmd /K "%~dp0\bin\pose_ddrnet.exe" -m0 %APP_MODEL0_PATH% -m1 %APP_MODEL1_PATH% -v %APP_VIDEO_PATH% -a

endlocal
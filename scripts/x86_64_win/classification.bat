@echo off
setlocal
pushd ..\..\
set "OPENCV_LIB_PATH=vcpkg_installed\x64-windows\lib\"
set "OPENCV_DLL_PATH=vcpkg_installed\x64-windows\bin\"
set PATH=%OPENCV_LIB_PATH%;%OPENCV_DLL_PATH%;%PATH%

set "APP_MODEL_PATH=assets\models\EfficientNetB0_4.dxnn"
set "APP_IMAGE_PATH=sample\ILSVRC2012\1.jpeg"

start cmd /K "bin\classification.exe" -m %APP_MODEL_PATH% -i %APP_IMAGE_PATH% -l 100
popd
endlocal
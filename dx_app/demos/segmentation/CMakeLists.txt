# DXRT App : Image Segmentation
add_executable(ddrnet ddrnet.cpp)
add_executable(pidnet pidnet.cpp)

add_dxrt_lib()
add_opencv()

add_target(ddrnet)
add_target(pidnet)

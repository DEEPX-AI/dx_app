# DXRT App : Image Segmentation
add_demo_utils()
add_dxrt_lib()
add_opencv()

add_executable(segmentation 
    ${DEMO_UTILS_SRC}
    segmentation.cpp 
)

add_target(segmentation)
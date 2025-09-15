add_demo_utils()
add_dxrt_lib()
add_opencv()

add_executable(od_segmentation 
    ${DEMO_UTILS_SRC}
    yolo_and_segmentation.cpp
)

add_target(od_segmentation)
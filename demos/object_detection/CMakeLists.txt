# DXRT App : Object Detection
add_demo_utils()
add_dxrt_lib()
add_opencv()

add_executable(yolo
    ${DEMO_UTILS_SRC}
    yolo_1channel.cpp
)
add_target(yolo)

add_executable(yolo_multi
    ${DEMO_UTILS_SRC}
    od.cpp
    yolo_multi_channels.cpp
)
add_target(yolo_multi)


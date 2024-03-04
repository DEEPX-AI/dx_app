# DXRT App : Object Detection

set(common_srcs bbox.cpp display.cpp nms.cpp)
set(yolo_srcs ${common_srcs} yolo.cpp yolo_cfg.cpp yolo_demo.cpp)

add_executable(dnyolo ${yolo_srcs})

add_dxrt_lib()
add_opencv()

add_target(dnyolo)
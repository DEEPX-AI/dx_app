add_executable(demo_ddr yolo_ddrnet.cpp yolo.cpp bbox.cpp display.cpp nms.cpp)
add_executable(demo_pid yolo_pidnet.cpp yolo.cpp bbox.cpp display.cpp nms.cpp)

add_dxrt_lib()
add_opencv()

add_target(demo_ddr)
add_target(demo_pid)

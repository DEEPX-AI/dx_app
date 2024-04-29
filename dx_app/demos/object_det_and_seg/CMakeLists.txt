add_executable(od_ddr yolo_ddrnet.cpp yolo.cpp bbox.cpp display.cpp nms.cpp)
add_executable(od_pid yolo_pidnet.cpp yolo.cpp bbox.cpp display.cpp nms.cpp)

add_dxrt_lib()
add_opencv()

add_target(od_ddr)
add_target(od_pid)

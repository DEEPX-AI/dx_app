# DXRT App : Object Detection
if(MSVC)
set(common_srcs bbox.cpp display.cpp nms.cpp)
else()
set(common_srcs bbox.cpp display.cpp nms.cpp v4l2.cpp socket.cpp fb.cpp)
endif()
set(yolo_srcs ${common_srcs} yolo.cpp yolo_cfg.cpp yolo_demo.cpp)
set(yolo_multi_srcs bbox.cpp display.cpp nms.cpp yolo.cpp yolo_cfg.cpp yolo_demo_multi.cpp od.cpp)

add_executable(yolo ${yolo_srcs})
add_executable(yolo_multi ${yolo_multi_srcs})

add_dxrt_lib()
add_opencv()

add_target(yolo)
add_target(yolo_multi)

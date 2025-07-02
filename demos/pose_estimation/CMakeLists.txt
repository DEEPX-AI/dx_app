# DXRT App : Pose Estimation

if(MSVC)
    set(common_srcs bbox.cpp display.cpp nms.cpp)
else()
    set(common_srcs bbox.cpp display.cpp nms.cpp isp.cpp v4l2.cpp osd_eyenix.cpp socket.cpp)
endif()

set(pose_srcs ${common_srcs} yolo.cpp yolo_cfg.cpp pose_demo.cpp)
set(pose_ddrnet_srcs ${common_srcs} yolo.cpp yolo_cfg.cpp pose_ddrnet.cpp)

add_executable(pose ${pose_srcs})
add_executable(pose_ddrnet ${pose_ddrnet_srcs})

add_dxrt_lib()
add_opencv()

add_target(pose)
add_target(pose_ddrnet)

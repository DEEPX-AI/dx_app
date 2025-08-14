# DXRT App : Pose Estimation
add_demo_utils()
add_dxrt_lib()
add_opencv()

add_executable(pose 
    ${DEMO_UTILS_SRC}
    yolo_pose.cpp
)

add_target(pose)
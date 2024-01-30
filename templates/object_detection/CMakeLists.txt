# template Example : Classification

set(run_det_src run_detector.cpp)

add_executable(run_detector ${run_det_src})

add_dxrt_lib()
add_opencv()

add_easylog(run_detector)
add_target(run_detector)

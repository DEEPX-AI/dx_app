# template Example : Classification

set(run_cls_src run_classifier.cpp)

add_executable(run_classifier ${run_cls_src})

add_dxrt_lib()
add_opencv()

add_target(run_classifier)

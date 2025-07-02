# DXRT App : Classification

set(classification_src classification.cpp)
set(imagenet_classification_src imagenet_classification.cpp)

add_executable(classification ${classification_src})
add_executable(imagenet_classification ${imagenet_classification_src})

add_dxrt_lib()
add_opencv()

add_target(classification)
add_target(imagenet_classification)

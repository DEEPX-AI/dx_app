# DXRT App : Classification

set(classification_src classification_sync.cpp)
set(classification_async_src classification_async.cpp)
set(imagenet_classification_src imagenet_classification.cpp)

add_executable(classification ${classification_src})
add_executable(classification_async ${classification_async_src})
add_executable(imagenet_classification ${imagenet_classification_src})

add_dxrt_lib()
add_opencv()

add_target(classification)
add_target(classification_async)
add_target(imagenet_classification)

# DXRT App : Classification
FILE(GLOB_RECURSE datapacket_classification_include "datapacket_include/*.cpp")

set(classification_src classification.cpp)
set(imagenet_classification_src imagenet_classification.cpp)
set(imagenet_classification_grid_version_src imagenet_classification_demo.cpp)
set(datapacket_classification_src datapacket_classification.cpp ${datapacket_classification_include})
set(miware_src miware_classification.cpp)

add_executable(classification ${classification_src})
add_executable(imagenet_classification ${imagenet_classification_src})
add_executable(imagenet_classification_demo ${imagenet_classification_grid_version_src})
add_executable(datapacket_classification ${datapacket_classification_src})
add_executable(miware ${miware_src})

add_dxrt_lib()
add_opencv()

add_target(classification)
add_target(imagenet_classification)
add_target(imagenet_classification_demo)
add_target(datapacket_classification)
add_target(miware)

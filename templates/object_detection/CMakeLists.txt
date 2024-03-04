# template Example : Classification

set(run_det_src run_detector.cpp)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
set(run_det_src ${run_det_src} ${CMAKE_SOURCE_DIR}/lib/utils/v1/isp_eyenix.cpp)
endif()

add_executable(run_detector ${run_det_src})

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
set(ENX_MEDIALINK_LIB ${CMAKE_SOURCE_DIR}/lib/utils/v1/libenxmedialink.a)
target_link_libraries(run_detector ${ENX_MEDIALINK_LIB})
endif()


add_dxrt_lib()
add_opencv()

add_easylog(run_detector)
add_target(run_detector)

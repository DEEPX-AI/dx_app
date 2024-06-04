set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(onnxruntime_LIB_DIRS /usr/local/lib)
set(onnxruntime_INCLUDE_DIRS
    /usr/local/include/onnxruntime
    /usr/local/include/onnxruntime/core/session
)

set(DXRT_INSTALLED_DIR /usr/local)

SET(CMAKE_C_COMPILER      /usr/bin/aarch64-linux-gnu-gcc )
SET(CMAKE_CXX_COMPILER    /usr/bin/aarch64-linux-gnu-g++ )
SET(CMAKE_LINKER          /usr/bin/aarch64-linux-gnu-ld  )
SET(CMAKE_NM              /usr/bin/aarch64-linux-gnu-nm )
SET(CMAKE_OBJCOPY         /usr/bin/aarch64-linux-gnu-objcopy )
SET(CMAKE_OBJDUMP         /usr/bin/aarch64-linux-gnu-objdump )
SET(CMAKE_RANLIB          /usr/bin/aarch64-linux-gnu-ranlib )

set(OpenCV_DIR            ${CMAKE_SOURCE_DIR}/extern/${CMAKE_SYSTEM_PROCESSOR})
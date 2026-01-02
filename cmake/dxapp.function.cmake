macro(add_target name)
  target_include_directories( ${name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/
    ${CMAKE_SOURCE_DIR}/lib/
  )
if(${STD_FS_NO_LIB_NEEDED})
  set(STD_FS_LIB "")
else()
  set(STD_FS_LIB stdc++fs)
endif()
target_link_libraries(${name} ${link_libs} ${STD_FS_LIB})

if(MSVC)
  install(
    TARGETS ${name}
    DESTINATION ${CMAKE_SOURCE_DIR}/bin
    ARCHIVE DESTINATION lib
  )
else()
  install(
    TARGETS ${name}
    DESTINATION bin
    LIBRARY DESTINATION lib
  )
endif()

endmacro(add_target)

macro(add_opencv)
  find_package(OpenCV REQUIRED HINTS ${OpenCV_DIR})
  LIST(APPEND link_libs ${OpenCV_LIBS})
endmacro(add_opencv)

macro(add_demo_utils)
  set(DEMO_UTILS_DIR ${CMAKE_SOURCE_DIR}/demos/demo_utils)
  include_directories(${DEMO_UTILS_DIR}/include)
  set(DEMO_UTILS_SRC 
    ${DEMO_UTILS_DIR}/yolo_cfg.cpp
    ${DEMO_UTILS_DIR}/yolo.cpp
    ${DEMO_UTILS_DIR}/nms.cpp
    ${DEMO_UTILS_DIR}/bbox.cpp
    ${DEMO_UTILS_DIR}/image.cpp
    ${DEMO_UTILS_DIR}/segmentation.cpp
    ${DEMO_UTILS_DIR}/display.cpp
  )
endmacro(add_demo_utils)

macro(add_dxrt_lib)
if(MSVC)
  add_library(dxrt SHARED IMPORTED)
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set_target_properties(dxrt PROPERTIES
      IMPORTED_IMPLIB "${DXRT_DIR}\\lib\\dxrtdbg.lib"
      IMPORTED_LOCATION "${DXRT_DIR}\\lib\\dxrtdbg.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${DXRT_DIR}\\include"
    )
  else()
    set_target_properties(dxrt PROPERTIES
      IMPORTED_IMPLIB "${DXRT_DIR}\\lib\\dxrt.lib"
      IMPORTED_LOCATION "${DXRT_DIR}\\lib\\dxrt.dll"
      INTERFACE_INCLUDE_DIRECTORIES "${DXRT_DIR}\\include"
    )
  endif()
  LIST(APPEND link_libs dxrt)
else()
  if(CROSS_COMPILE)
    if(DXRT_INSTALLED_DIR)
      add_library(dxrt SHARED IMPORTED)
      set_target_properties(dxrt PROPERTIES
        IMPORTED_LOCATION "${DXRT_INSTALLED_DIR}/lib/libdxrt.so"
        INTERFACE_INCLUDE_DIRECTORIES "${DXRT_INSTALLED_DIR}/include"
      )  
    else()
      find_package(dxrt REQUIRED)
    endif()
  else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
  endif()
  LIST(APPEND link_libs dxrt pthread)
endif()  

endmacro(add_dxrt_lib)
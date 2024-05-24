macro(add_target name)
  target_include_directories( ${name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/
    ${CMAKE_SOURCE_DIR}/lib/
  )
  target_link_libraries(${name} ${link_libs})
  install(
    TARGETS ${name}
    DESTINATION bin
    LIBRARY DESTINATION lib
  )
endmacro(add_target)

macro(add_opencv)
	  find_package(OpenCV 4.5.5 REQUIRED HINTS ${OpenCV_DIR})
    LIST(APPEND link_libs ${OpenCV_LIBS})
endmacro(add_opencv)

macro(add_onnxruntime)
	  find_library(onnxruntime HINTS ${onnxruntime_LIB_DIRS})
    LIST(APPEND link_libs onnxruntime)
endmacro(add_onnxruntime)

macro(add_dxrt_lib)
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
  if(USE_ORT)
    add_onnxruntime()
  endif()

endmacro(add_dxrt_lib)
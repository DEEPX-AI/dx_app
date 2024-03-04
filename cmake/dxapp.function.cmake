macro(add_target name)
  target_include_directories( ${name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/
    ${CMAKE_SOURCE_DIR}/lib/
if(USE_EASYLOG)
      ${CMAKE_SOURCE_DIR}/extern/easyloggingpp/
endif()
  )
  target_link_libraries(${name} ${link_libs} )
  install(
    TARGETS ${name}
    DESTINATION bin
    LIBRARY DESTINATION lib
  )
endmacro(add_target)

macro(add_easylog name)
  if(USE_EASYLOG)
    target_sources(${name} PUBLIC ${CMAKE_SOURCE_DIR}/extern/easyloggingpp/easylogging++.cc)
  endif()
endmacro(add_easylog)

macro(add_opencv)
if(USE_OPENCV)
	  find_package(OpenCV 4.5.5 REQUIRED HINTS ${OpenCV_DIR})
    LIST(APPEND link_libs ${OpenCV_LIBS})
endif()
endmacro(add_opencv)

macro(add_onnxruntime)
if(USE_ORT)
  find_package(onnxruntime)
  if(onnxruntime_FOUND)
    link_libraries(onnxruntime::onnxruntime)
  else()
    message("onnxruntime not found.")
  endif()
  add_compile_definitions(USE_ORT=1)
endif()
endmacro(add_onnxruntime)

macro(add_dxrt_lib)
  add_library(dxrt_lib SHARED IMPORTED)
  set_target_properties(dxrt_lib PROPERTIES
    IMPORTED_LOCATION "${DXRT_DIR}/lib/libdxrt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${DXRT_DIR}/include"
  )
  LIST(APPEND link_libs dxrt_lib pthread)
endmacro(add_dxrt_lib)
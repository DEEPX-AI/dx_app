macro(add_target name)
  target_include_directories( ${name} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/
    ${CMAKE_SOURCE_DIR}/lib/
  )
  target_link_libraries(${name} ${link_libs} stdc++fs)

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
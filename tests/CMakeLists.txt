# DX APP Unit Test

set(target dxapp_test)

    macro(add_googletest name)
        # release 1.12.0
        include(FetchContent)
        FetchContent_Declare(
            googletest
            # Specify the commit you depend on and update it regularly.
            URL https://github.com/google/googletest/archive/5376968f6948923e2411081fd9372e71a59d8e77.zip
        )
        # For Windows: Prevent overriding the parent project's compiler/linker settings
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
        target_link_libraries(${name} gtest_main)
    endmacro(add_googletest)

    FILE(GLOB srcs "src/*.cpp" "src/*.cc" "src/*.c" "src/*.hpp")
    add_executable(${target} ${srcs})
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib
        ${CMAKE_SOURCE_DIR}/templates
    )

    add_dxrt_lib()
    add_opencv()
    add_googletest(${target})

    target_link_libraries(${target} ${link_libs})

    install(
        TARGETS ${target}
        DESTINATION bin
        LIBRARY DESTINATION lib
    )
    
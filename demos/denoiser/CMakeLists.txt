add_executable(denoiser dncnn.cpp)

add_dxrt_lib()
add_opencv()

add_target(denoiser)
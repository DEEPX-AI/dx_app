# DXRT App : Face Recognition
set(face_recognition_src face_recognition_demo.cpp ssd.cpp nms.cpp bbox.cpp utils.cpp)

add_executable(face_recognition ${face_recognition_src})

add_dxrt_lib()
add_opencv()

add_target(face_recognition)

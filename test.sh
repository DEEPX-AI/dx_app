#!/bin/bash

# scrfd
./bin/scrfd_sync -m assets/models/SCRFD500M_1.dxnn -v assets/videos/dance-group.mov
./bin/scrfd_sync -m assets/models/SCRFD500M_1.dxnn -c 0
./bin/scrfd_sync -m assets/models/SCRFD500M_1.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/scrfd_async -m assets/models/SCRFD500M_1.dxnn -v assets/videos/dance-group.mov
./bin/scrfd_async -m assets/models/SCRFD500M_1.dxnn -c 0
./bin/scrfd_async -m assets/models/SCRFD500M_1.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov5
./bin/yolov5_sync -m assets/models/YOLOV5S_4.dxnn -v assets/videos/dance-group.mov
./bin/yolov5_sync -m assets/models/YOLOV5S_4.dxnn -c 0
./bin/yolov5_sync -m assets/models/YOLOV5S_4.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov5_async -m assets/models/YOLOV5S_4.dxnn -v assets/videos/dance-group.mov
./bin/yolov5_async -m assets/models/YOLOV5S_4.dxnn -c 0
./bin/yolov5_async -m assets/models/YOLOV5S_4.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov5face
./bin/yolov5face_sync -m assets/models/YOLOV5S_Face-1.dxnn -v assets/videos/dance-group.mov
./bin/yolov5face_sync -m assets/models/YOLOV5S_Face-1.dxnn -c 0
./bin/yolov5face_sync -m assets/models/YOLOV5S_Face-1.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov5face_async -m assets/models/YOLOV5S_Face-1.dxnn -v assets/videos/dance-group.mov
./bin/yolov5face_async -m assets/models/YOLOV5S_Face-1.dxnn -c 0
./bin/yolov5face_async -m assets/models/YOLOV5S_Face-1.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov5pose
./bin/yolov5pose_sync -m assets/models/YOLOV5Pose640_1.dxnn -v assets/videos/dance-group.mov
./bin/yolov5pose_sync -m assets/models/YOLOV5Pose640_1.dxnn -c 0
./bin/yolov5pose_sync -m assets/models/YOLOV5Pose640_1.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov5pose_async -m assets/models/YOLOV5Pose640_1.dxnn -v assets/videos/dance-group.mov
./bin/yolov5pose_async -m assets/models/YOLOV5Pose640_1.dxnn -c 0
./bin/yolov5pose_async -m assets/models/YOLOV5Pose640_1.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov7
./bin/yolov7_sync -m assets/models/YOLOv7_512.dxnn -v assets/videos/dance-group.mov
./bin/yolov7_sync -m assets/models/YOLOv7_512.dxnn -c 0
./bin/yolov7_sync -m assets/models/YOLOv7_512.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov7_async -m assets/models/YOLOv7_512.dxnn -v assets/videos/dance-group.mov
./bin/yolov7_async -m assets/models/YOLOv7_512.dxnn -c 0
./bin/yolov7_async -m assets/models/YOLOv7_512.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov8
./bin/yolov8_sync -m assets/models/YoloV8N.dxnn -v assets/videos/dance-group.mov
./bin/yolov8_sync -m assets/models/YoloV8N.dxnn -c 0
./bin/yolov8_sync -m assets/models/YoloV8N.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov8_async -m assets/models/YoloV8N.dxnn -v assets/videos/dance-group.mov
./bin/yolov8_async -m assets/models/YoloV8N.dxnn -c 0
./bin/yolov8_async -m assets/models/YoloV8N.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolov9
./bin/yolov9_sync -m assets/models/YOLOV9S.dxnn -v assets/videos/dance-group.mov
./bin/yolov9_sync -m assets/models/YOLOV9S.dxnn -c 0
./bin/yolov9_sync -m assets/models/YOLOV9S.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolov9_async -m assets/models/YOLOV9S.dxnn -v assets/videos/dance-group.mov
./bin/yolov9_async -m assets/models/YOLOV9S.dxnn -c 0
./bin/yolov9_async -m assets/models/YOLOV9S.dxnn -r rtsp://192.168.30.100:8554/stream9

# yolox
./bin/yolox_sync -m assets/models/YOLOX-S_1.dxnn -v assets/videos/dance-group.mov
./bin/yolox_sync -m assets/models/YOLOX-S_1.dxnn -c 0
./bin/yolox_sync -m assets/models/YOLOX-S_1.dxnn -r rtsp://192.168.30.100:8554/stream9

./bin/yolox_async -m assets/models/YOLOX-S_1.dxnn -v assets/videos/dance-group.mov
./bin/yolox_async -m assets/models/YOLOX-S_1.dxnn -c 0
./bin/yolox_async -m assets/models/YOLOX-S_1.dxnn -r rtsp://192.168.30.100:8554/stream9
#include "yolo.h"

YoloLayerParam createYoloLayerParam(std::string _name, int _gx, int _gy, int _numB, const std::vector<float>& _vAnchorW, const std::vector<float>& _vAnchorH, const std::vector<int>& _vTensorIdx, float _sx = 0.f, float _sy = 0.f)
{
        YoloLayerParam s;
        s.name = _name;
        if(s.name == ""){
            std::cerr << "YoloLayerParam name is empty" << std::endl;
            throw std::runtime_error("YoloLayerParam name is empty");
        }
        s.numGridX = _gx;
        s.numGridY = _gy;
        s.numBoxes = _numB;
        s.anchorWidth = _vAnchorW;
        s.anchorHeight = _vAnchorH;
        s.tensorIdx = _vTensorIdx;
        s.scaleX = _sx;
        s.scaleY = _sy;
        return s;
}

// YOLOv3 configuration for 512x512 input resolution - classic object detection model (YOLOV3_1.dxnn)
YoloParam yolov3_512 = {
    512,  // height
    512,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    0,   // numBoxes
    80,   // numClasses
    "515", // onnx output name
    {     // if use_ort = off, layers config
        createYoloLayerParam("332", 64, 64, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("393", 32, 32, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("454", 16, 16, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    // postproc_type (0: yolo, 1: yolo pose, 2: yolo face)
    PostProcType::OD
};

// YOLOv4 configuration for 416x416 input resolution - improved accuracy with CSP backbone (YOLOV4_3.dxnn)
YoloParam yolov4_416 = {
    416,  // height
    416,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    0,   // numBoxes
    80,   // numClasses
    "2441", // onnx output name
    {     // if use_ort = off, layers config
        createYoloLayerParam("2014", 13, 13, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }, 1.2f, 1.2f),
        createYoloLayerParam("1565", 26, 26, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }, 1.1f, 1.1f),
        createYoloLayerParam("1116", 52, 52, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 }, 1.05f, 1.05f)
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    // postproc_type (0: yolo, 1: yolo pose, 2: yolo face)
    PostProcType::OD
};

// YOLOv5s configuration for 320x320 input resolution - lightweight model for real-time detection (YOLOV5S_4.dxnn)
YoloParam yolov5s_320 = {
    320,  // height
    320,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    0,   // numBoxes
    80,   // numClasses
    "output", // onnx output name
    {     // if use_ort = off, layers config
        createYoloLayerParam("378", 40, 40, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("439", 20, 20, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("500", 10, 10, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    // postproc_type (0: yolo, 1: yolo pose, 2: yolo face)
    PostProcType::OD
};

// YOLOv5s configuration for 512x512 input resolution - balanced performance and accuracy (YOLOV5S_3.dxnn)
YoloParam yolov5s_512 = {
    512,
    512,
    0.25,
    0.3,
    0.4,
    0,
    80,
    "output", 
    {
        createYoloLayerParam("378", 64, 64, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("439", 32, 32, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("500", 16, 16, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::OD
};

// YOLOv5s configuration for 640x640 input resolution - high accuracy detection model (YOLOV5S_6.dxnn)
YoloParam yolov5s_640 = {
    640,
    640,
    0.25,
    0.3,
    0.4,
    0,
    80,
    "1487",
    {
        createYoloLayerParam("611", 80, 80, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("903", 40, 40, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("1195", 20, 20, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::OD
};

// YOLOX-S configuration for 512x512 input resolution - anchor-free detection with decoupled head (YOLOX-S.dxnn)
YoloParam yolox_s_512 = {
    512,
    512,
    0.25,
    0.3,
    0.4,
    0,
    80,
    "output",
    {
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::OD
};

// YOLOv7 configuration for 640x640 input resolution - improved architecture with auxiliary heads (YoloV7.dxnn)
YoloParam yolov7_640 = {
    640,
    640,
    0.25,
    0.3,
    0.4,
    0,
    80,
    "output",
    {
        createYoloLayerParam("onnx::Reshape_491", 80, 80, 3, { 12.0, 19.0, 40.0 }, { 16.0, 36.0, 28.0 }, { 0 }),
        createYoloLayerParam("onnx::Reshape_525", 40, 40, 3, { 36.0, 76.0, 72.0 }, { 75.0, 55.0, 146.0 }, { 1 }),
        createYoloLayerParam("onnx::Reshape_559", 20, 20, 3, { 142.0, 192.0, 459.0 }, { 110.0, 243.0, 401.0 }, { 2 })
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::OD
};

// YOLOv7 configuration for 512x512 input resolution - optimized for speed and accuracy balance (YOLOv7_512.dxnn)
YoloParam yolov7_512 = {
    512,
    512,
    0.15,
    0.25,
    0.4,
    0,
    80,
    "output",
    {
        createYoloLayerParam("onnx::Reshape_491", 64, 64, 3, { 12.0f, 19.0f, 40.0f }, { 16.0f, 36.0f, 28.0f }, { 0 }),
        createYoloLayerParam("onnx::Reshape_525", 32, 32, 3, { 36.0f, 76.0f, 72.0f }, { 75.0f, 55.0f, 146.0f }, { 1 }),
        createYoloLayerParam("onnx::Reshape_559", 16, 16, 3, { 142.0f, 192.0f, 459.0f }, { 110.0f, 243.0f, 401.0f }, { 2 })
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::OD
};

// YOLOv8 configuration for 640x640 input resolution - anchor-free detection with classification-free head (YoloV8N.dxnn)
YoloParam yolov8_640 = {
    640,
    640,
    0.3,
    0.3,
    0.4,
    8400,
    80,
    "output0",
    {
        createYoloLayerParam("/model.22/Sigmoid_output_0", 80, 8400, 3, {}, {}, { 0 }),
        createYoloLayerParam("/model.22/dfl/conv/Conv_output_0", 4, 8400, 3, {}, {}, { 1 }),
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::YOLOV8
};

// YOLOv9 configuration for 640x640 input resolution - latest YOLO version with improved architecture (YOLOV9S.dxnn)
YoloParam yolov9_640 = {
    640,
    640,
    0.3,
    0.3,
    0.4,
    8400,
    80,
    "output0",
    {
    },
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    PostProcType::YOLOV8
};

// YOLOv5s6 pose estimation configuration for 640x640 input resolution - human pose detection (YOLOV5Pose640_1.dxnn)
YoloParam yolov5s6_pose_640 = {
    640,
    640,
    0.3,
    0.3,
    0.4,
    0,
    1,
    "detections",
    {
    },
    {"person"},
    PostProcType::POSE
};

// YOLOv5s face detection configuration for 640x640 input resolution - specialized for face detection (YOLOV5S_Face-1.dxnn)
YoloParam yolov5s_face_640 = {
    640,
    640,
    0.25,
    0.3,
    0.4,
    0,
    1,
    "704",
    {
        createYoloLayerParam("/model.23/m.0/Conv_output_0", 80, 80, 3, { 4.0, 8.0, 13.0 }, { 5.0, 10.0, 16.0 }, { 0 }),
        createYoloLayerParam("/model.23/m.1/Conv_output_0", 40, 40, 3, { 23.0, 43.0, 73.0 }, { 29.0, 55.0, 105.0 }, { 1 }),
        createYoloLayerParam("/model.23/m.2/Conv_output_0", 20, 20, 3, { 146.0, 231.0, 335.0 }, { 217.0, 300.0, 433.0 }, { 2 })
    },
    {"face"},
    PostProcType::FACE
};
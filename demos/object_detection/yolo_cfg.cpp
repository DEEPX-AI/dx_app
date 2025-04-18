#include "yolo.h"

YoloLayerParam createYoloLayerParam(std::string _name, int _gx, int _gy, int _numB, const std::vector<float>& _vAnchorW, const std::vector<float>& _vAnchorH, const std::vector<int>& _vTensorIdx, float _sx = 0.f, float _sy = 0.f)
{
        YoloLayerParam s;
        s.name = _name;
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

YoloParam yolov5s_320 = {
    320,  // height
    320,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
        createYoloLayerParam("", 40, 40, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("", 20, 20, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("", 10, 10, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};

YoloParam yolov5s_512 = {
    512,  // height
    512,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
        createYoloLayerParam("", 64, 64, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("", 32, 32, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("", 16, 16, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};

YoloParam yolov5s_640 = {
    640,  // height
    640,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
        createYoloLayerParam("", 80, 80, 3, { 10.0f, 16.0f, 33.0f }, { 13.0f, 30.0f, 23.0f }, { 0 }),
        createYoloLayerParam("", 40, 40, 3, { 30.0f, 62.0f, 59.0f }, { 61.0f, 45.0f, 119.0f }, { 1 }),
        createYoloLayerParam("", 20, 20, 3, { 116.0f, 156.0f, 373.0f }, { 90.0f, 198.0f, 326.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};
YoloParam yolox_s_512 = {
    512,  // height
    512,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
        createYoloLayerParam("", 64, 64, 3, {}, {}, {0, 1, 2}),
        createYoloLayerParam("", 32, 32, 3, {}, {}, {3, 4, 5}),
        createYoloLayerParam("", 16, 16, 3, {}, {}, {6, 7, 8})
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};

YoloParam yolov7_640 = {
    640,  // height
    640,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {
            createYoloLayerParam("onnx::Reshape_491", 80, 80, 3, { 12.0, 19.0, 40.0 }, { 16.0, 36.0, 28.0 }, { 0 }),
            createYoloLayerParam("onnx::Reshape_525", 40, 40, 3, { 36.0, 76.0, 72.0 }, { 75.0, 55.0, 146.0 }, { 1 }),
            createYoloLayerParam("onnx::Reshape_559", 20, 20, 3, { 142.0, 192.0, 459.0 }, { 110.0, 243.0, 401.0 }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};

YoloParam yolov7_512 = {
    512,  // height
    512,  // width
    0.15, // confThreshold
    0.25, // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
        createYoloLayerParam("", 64, 64, 3, { 12.0f, 19.0f, 40.0f }, { 16.0f, 36.0f, 28.0f }, { 0 }),
        createYoloLayerParam("", 32, 32, 3, { 36.0f, 76.0f, 72.0f }, { 75.0f, 55.0f, 146.0f }, { 1 }),
        createYoloLayerParam("", 16, 16, 3, { 142.0f, 192.0f, 459.0f }, { 110.0f, 243.0f, 401.0f }, { 2 })
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};

YoloParam yolox_s_640 = {
    640,  // height
    640,  // width
    0.25, // confThreshold
    0.3,  // scoreThreshold
    0.4,  // iouThreshold
    -1,   // numBoxes
    80,   // numClasses
    {     // layers
            createYoloLayerParam("", 80, 80, 3, {}, {}, {0, 1, 2}),
            createYoloLayerParam("", 40, 40, 3, {}, {}, {3, 4, 5}),
            createYoloLayerParam("", 20, 20, 3, {}, {}, {6, 7, 8})
    },
    // classNames
    { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "diningtable", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"}
};
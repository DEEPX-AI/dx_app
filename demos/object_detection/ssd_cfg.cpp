#include "ssd.h"

SsdParam mv1_ssd_300 = {
    300,                // image_size
    false,              // use_softmax
    0.3,                // score_threshold
    0.6,                // iou_threshold
    91,                 // num_classes
    // class_names
    {"BACKGROUND", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant", "_", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "_", "backpack", "umbrella", "_", "_", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", "_", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed", "_", "diningtable", "_", "_", "toilet", "_", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator", "_", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    // priorBoxes
    {
        6,              // num_layers
        0.2,            // min_scale
        0.95,           // max_scale
        0.1,            // center_variance
        0.2,            // size_variance
        // dim
        {
            { 19, 19, 3},
            { 10, 10, 6},
            { 5, 5, 6},
            { 3, 3, 6},
            { 2, 2, 6},
            { 1, 1, 6},
        },
        // data_file : import prior boxes from binary file
        ""
    }
};
SsdParam mv2_ssd_320 = {
    320,                // image_size
    true,               // use_softmax
    0.5,                // score_threshold
    0.5,                // iou_threshold
    5,                  // num_classes
    // class_names
    {"BACKGROUND", "person", "car", "motorcycle", "bicycle"},
    // priorBoxes
    {
        5,              // num_layers
        0.2,            // min_scale
        0.95,           // max_scale
        0.125,          // center_variance
        0.125,          // size_variance
        // dim
        {
            { 20, 20, 12},
            { 10, 10, 6},
            { 5, 5, 6},
            { 3, 3, 6},
            { 1, 1, 4},
        },
        // data_file
        "sample/mv2_ssd_320.prior_boxes.bin"
    }
};
SsdParam mv1_ssd_512 = {
    512,                // image_size
    true,               // use_softmax
    0.3,                // score_threshold
    0.4,                // iou_threshold
    21,                 // num_classes
    // class_names
    { "BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"},
    // priorBoxes
    {
        6,              // num_layers
        0.1171875,      // min_scale
        0.556640625,    // max_scale
        0.1,            // center_variance
        0.2,            // size_variance
        // dim
        {
            { 32, 32, 6},
            { 16, 16, 6},
            { 8, 8, 6},
            { 4, 4, 6},
            { 2, 2, 6},
            { 1, 1, 6},
        },
        // data_file
        "sample/mv1_ssd_512.prior_boxes.bin"
    }
};
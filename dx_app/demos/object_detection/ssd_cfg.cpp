#include "ssd.h"

SsdParam mv1_ssd_300 = {
    .image_size = 300,
    .use_softmax = false,
    .score_threshold = 0.3,
    .iou_threshold = 0.6,
    .num_classes = 91,
    .class_names = {"BACKGROUND", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "trafficlight", "firehydrant" ,  "_", "stopsign", "parkingmeter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe" ,  "_", "backpack", "umbrella" ,  "_" ,  "_", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle" ,  "_", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair", "couch", "pottedplant", "bed" ,  "_", "diningtable" ,  "_" ,  "_", "toilet" ,  "_", "tv", "laptop", "mouse", "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator" ,  "_", "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"},
    .priorBoxes = {
        .num_layers = 6,
        .min_scale = 0.2,
        .max_scale = 0.95,
        .center_variance = 0.1,
        .size_variance = 0.2,
        .dim = {
            { 19, 19, 3},
            { 10, 10, 6},
            { 5, 5, 6},
            { 3, 3, 6},
            { 2, 2, 6},
            { 1, 1, 6},
        },
        // .data_file = "sample/mv1_ssd_300.prior_boxes.bin" // import prior boxes from binary file
        .data_file = "", // generate prior boxes
    },
};
SsdParam mv2_ssd_320 = {
    .image_size = 320,
    .use_softmax = true,
    .score_threshold = 0.5,
    .iou_threshold = 0.5,
    .num_classes = 5,
    .class_names = {"BACKGROUND", "person", "car", "motorcycle", "bicycle"},
    .priorBoxes = {
        .num_layers = 5,
        .min_scale = 0.2,
        .max_scale = 0.95,
        .center_variance = 0.125,
        .size_variance = 0.125,
        .dim = {
            { 20, 20, 12},
            { 10, 10, 6},
            { 5, 5, 6},
            { 3, 3, 6},
            { 1, 1, 4},
        },
        .data_file = "sample/mv2_ssd_320.prior_boxes.bin" // import prior boxes from binary file
    },
};
SsdParam mv1_ssd_512 = {
    .image_size = 512,
    .use_softmax = true,
    .score_threshold = 0.3,
    .iou_threshold = 0.4,
    .num_classes = 21,
    .class_names = {"BACKGROUND", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"},
    .priorBoxes = {
        .num_layers = 6,
        .min_scale = 0.1171875,
        .max_scale = 0.556640625,
        .center_variance = 0.1,
        .size_variance = 0.2,
        .dim = {
            { 32, 32, 6},
            { 16, 16, 6},
            { 8, 8, 6},
            { 4, 4, 6},
            { 2, 2, 6},
            { 1, 1, 6},
        },
        .data_file = "sample/mv1_ssd_512.prior_boxes.bin" // import prior boxes from binary file
    },
};
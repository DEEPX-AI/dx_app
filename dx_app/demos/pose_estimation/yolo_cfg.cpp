#include "yolo.h"

YoloParam yolov5s6_pose_640 = {
    .height = 640,
    .width = 640,
    .confThreshold = 0.3,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 1,
    .layers = {
        {
            .numGridX = 80,
            .numGridY = 80,
            .numBoxes = 3,
            .anchorWidth = { 19.0, 44.0, 38.0 },
            .anchorHeight = { 27.0, 40.0, 94.0 },
            .tensorIdx = { 1, 0 },
        },
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 96.0, 86.0, 180.0 },
            .anchorHeight = { 68.0, 152.0, 137.0 },
            .tensorIdx = { 3, 2 },
        },
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 140.0, 303.0, 238.0 },
            .anchorHeight = { 301.0, 264.0, 542.0 },
            .tensorIdx = { 5, 4 },
        },
        {
            .numGridX = 10,
            .numGridY = 10,
            .numBoxes = 3,
            .anchorWidth = { 436.0, 739.0, 925.0 },
            .anchorHeight = { 615.0, 380.0, 792.0 },
            .tensorIdx = { 7, 6 },
        },
    },
    .classNames = {"person"},
};

YoloParam yolov5s6_pose_1280 = {
    .height = 1280,
    .width = 1280,
    .confThreshold = 0.3,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 1,
    .layers = {
        {
            .numGridX = 160,
            .numGridY = 160,
            .numBoxes = 3,
            .anchorWidth = { 19.0, 44.0, 38.0 },
            .anchorHeight = { 27.0, 40.0, 94.0 },
            .tensorIdx = { 0, 1 },
        },
        {
            .numGridX = 80,
            .numGridY = 80,
            .numBoxes = 3,
            .anchorWidth = { 96.0, 86.0, 180.0 },
            .anchorHeight = { 68.0, 152.0, 137.0 },
            .tensorIdx = { 2, 3 },
        },
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 140.0, 303.0, 238.0 },
            .anchorHeight = { 301.0, 264.0, 542.0 },
            .tensorIdx = { 4, 5 },
        },
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 436.0, 739.0, 925.0 },
            .anchorHeight = { 615.0, 380.0, 792.0 },
            .tensorIdx = { 6, 7 },
        },
    },
    .classNames = {"person"},
};

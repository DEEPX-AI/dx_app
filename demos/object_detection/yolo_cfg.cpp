#include "yolo.h"
YoloParam yolov5s_320 = {
    .height = 320,
    .width = 320,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 10.0, 16.0, 33.0 },
            .anchorHeight = { 13.0, 30.0, 23.0 },
            .tensorIdx = { 0 },
        },
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 30.0, 62.0, 59.0 },
            .anchorHeight = { 61.0, 45.0, 119.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 10,
            .numGridY = 10,
            .numBoxes = 3,
            .anchorWidth = { 116.0, 156.0, 373.0 },
            .anchorHeight = { 90.0, 198.0, 326.0 },
            .tensorIdx = { 2 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};
YoloParam yolov5s_512 = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 64,
            .numGridY = 64,
            .numBoxes = 3,
            .anchorWidth = { 10.0, 16.0, 33.0 },
            .anchorHeight = { 13.0, 30.0, 23.0 },
            .tensorIdx = { 0 },
        },
        {
            .numGridX = 32,
            .numGridY = 32,
            .numBoxes = 3,
            .anchorWidth = { 30.0, 62.0, 59.0 },
            .anchorHeight = { 61.0, 45.0, 119.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 16,
            .numGridY = 16,
            .numBoxes = 3,
            .anchorWidth = { 116.0, 156.0, 373.0 },
            .anchorHeight = { 90.0, 198.0, 326.0 },
            .tensorIdx = { 2 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};
YoloParam yolov5s_640 = {
    .height = 640,
    .width = 640,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 80,
            .numGridY = 80,
            .numBoxes = 3,
            .anchorWidth = { 10.0, 16.0, 33.0 },
            .anchorHeight = { 13.0, 30.0, 23.0 },
            .tensorIdx = { 0 },
        },
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 30.0, 62.0, 59.0 },
            .anchorHeight = { 61.0, 45.0, 119.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 116.0, 156.0, 373.0 },
            .anchorHeight = { 90.0, 198.0, 326.0 },
            .tensorIdx = { 2 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};
YoloParam yolov5s_640_ppu = {
    .height = 640,
    .width = 640,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 116.0, 156.0, 373.0 },
            .anchorHeight = { 90.0, 198.0, 326.0 },
            .tensorIdx = { 2 },
        },
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 30.0, 62.0, 59.0 },
            .anchorHeight = { 61.0, 45.0, 119.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 80,
            .numGridY = 80,
            .numBoxes = 3,
            .anchorWidth = { 10.0, 16.0, 33.0 },
            .anchorHeight = { 13.0, 30.0, 23.0 },
            .tensorIdx = { 0 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};
YoloParam yolov5s_512_concat = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = 16125,
    .numClasses = 80,
    .layers = {},
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};
YoloParam yolox_s_512 = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 64,
            .numGridY = 64,
            .numBoxes = 3,
            .anchorWidth = { }, // no anchor
            .anchorHeight = { },// no anchor
            .tensorIdx = { 0, 1, 2 },//location, boxScore, classScore
        },
        {
            .numGridX = 32,
            .numGridY = 32,
            .numBoxes = 3,
            .anchorWidth = {  },// no anchor
            .anchorHeight = {  },// no anchor
            .tensorIdx = { 3, 4, 5 },//location, boxScore, classScore
        },
        {
            .numGridX = 16,
            .numGridY = 16,
            .numBoxes = 3,
            .anchorWidth = {  },// no anchor
            .anchorHeight = {  },// no anchor
            .tensorIdx = { 6, 7, 8 },//location, boxScore, classScore
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};

YoloParam yolov7_640 = {
    .height = 640,
    .width = 640,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 80,
            .numGridY = 80,
            .numBoxes = 3,
            .anchorWidth = { 12.0, 19.0, 40.0 },
            .anchorHeight = { 16.0, 36.0, 28.0 },
            .tensorIdx = { 0 },
        },
        {
            .numGridX = 40,
            .numGridY = 40,
            .numBoxes = 3,
            .anchorWidth = { 36.0, 76.0, 72.0 },
            .anchorHeight = { 75.0, 55.0, 146.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 20,
            .numGridY = 20,
            .numBoxes = 3,
            .anchorWidth = { 142.0, 192.0, 459.0 },
            .anchorHeight = { 110.0, 243.0, 401.0 },
            .tensorIdx = { 2 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};

YoloParam yolov7_512 = {
    .height = 512,
    .width = 512,
    .confThreshold = 0.15,
    .scoreThreshold = 0.25,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
        {
            .numGridX = 64,
            .numGridY = 64,
            .numBoxes = 3,
            .anchorWidth = { 12.0, 19.0, 40.0 },
            .anchorHeight = { 16.0, 36.0, 28.0 },
            .tensorIdx = { 0 },
        },
        {
            .numGridX = 32,
            .numGridY = 32,
            .numBoxes = 3,
            .anchorWidth = { 36.0, 76.0, 72.0 },
            .anchorHeight = { 75.0, 55.0, 146.0 },
            .tensorIdx = { 1 },
        },
        {
            .numGridX = 16,
            .numGridY = 16,
            .numBoxes = 3,
            .anchorWidth = { 142.0, 192.0, 459.0 },
            .anchorHeight = { 110.0, 243.0, 401.0 },
            .tensorIdx = { 2 },
        },
    },
    .classNames = {"person" ,"bicycle" ,"car" ,"motorcycle" ,"airplane" ,"bus" ,"train" ,"truck" ,"boat" ,"trafficlight" ,"firehydrant" ,"stopsign" ,"parkingmeter" ,"bench" ,"bird" ,"cat" ,"dog" ,"horse" ,"sheep" ,"cow" ,"elephant" ,"bear" ,"zebra" ,"giraffe" ,"backpack" ,"umbrella" ,"handbag" ,"tie" ,"suitcase" ,"frisbee" ,"skis" ,"snowboard" ,"sportsball" ,"kite" ,"baseballbat" ,"baseballglove" ,"skateboard" ,"surfboard" ,"tennisracket" ,"bottle" ,"wineglass" ,"cup" ,"fork" ,"knife" ,"spoon" ,"bowl" ,"banana" ,"apple" ,"sandwich" ,"orange" ,"broccoli" ,"carrot" ,"hotdog" ,"pizza" ,"donut" ,"cake" ,"chair" ,"couch" ,"pottedplant" ,"bed" ,"diningtable" ,"toilet" ,"tv" ,"laptop" ,"mouse" ,"remote" ,"keyboard" ,"cellphone" ,"microwave" ,"oven" ,"toaster" ,"sink" ,"refrigerator" ,"book" ,"clock" ,"vase" ,"scissors" ,"teddybear" ,"hairdrier", "toothbrush"},
};

YoloParam yolov4_608 = {
    .height = 608,
    .width = 608,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 12,
    .layers = {
        {
            .numGridX = 76,
            .numGridY = 76,
            .numBoxes = 3,
            .anchorWidth = {12.0, 19.0, 40.0},
            .anchorHeight = {16.0, 36.0, 28.0},
            .tensorIdx = {0},
            .scaleX = 1.2,
            .scaleY = 1.2,
        },
        {
            .numGridX = 38,
            .numGridY = 38,
            .numBoxes = 3,
            .anchorWidth = {36.0, 76.0, 72.0},
            .anchorHeight = {75.0, 55.0, 146.0},
            .tensorIdx = {1},
            .scaleX = 1.1,
            .scaleY = 1.1,
        },
        {
            .numGridX = 19,
            .numGridY = 19,
            .numBoxes = 3,
            .anchorWidth = {142.0, 192.0, 459.0},
            .anchorHeight = {110.0, 243.0, 401.0},
            .tensorIdx = {2},
            .scaleX = 1.05,
            .scaleY = 1.05,

        },
    },
    .classNames = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"},
};
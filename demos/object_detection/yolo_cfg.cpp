#include "yolo.h"

YoloLayerParam createYoloLayerParam(int _gx, int _gy, int _numB, const std::vector<float> &_vAnchorW, const std::vector<float> &_vAnchorH, const std::vector<int> &_vTensorIdx, float _sx = 0.f, float _sy = 0.f)
{
        YoloLayerParam s;
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
    .height = 320,
    .width = 320,
    .confThreshold = 0.25,
    .scoreThreshold = 0.3,
    .iouThreshold = 0.4,
    .numBoxes = -1, // check from layer info.
    .numClasses = 80,
    .layers = {
            createYoloLayerParam(40, 40, 3, { 10.0, 16.0, 33.0 }, { 13.0, 30.0, 23.0 }, { 0 }),
            createYoloLayerParam(20, 20, 3, { 30.0, 62.0, 59.0 }, { 61.0, 45.0, 119.0 }, { 1 }),
            createYoloLayerParam(10, 10, 3, { 116.0, 156.0, 373.0 }, { 90.0, 198.0, 326.0 }, { 2 })
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
            createYoloLayerParam(64, 64, 3, { 10.0, 16.0, 33.0 }, { 13.0, 30.0, 23.0 }, { 0 }),
            createYoloLayerParam(32, 32, 3, { 30.0, 62.0, 59.0 }, { 61.0, 45.0, 119.0 }, { 1 }),
            createYoloLayerParam(16, 16, 3, { 116.0, 156.0, 373.0 }, { 90.0, 198.0, 326.0 }, { 2 })
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
            createYoloLayerParam(80, 80, 3, { 10.0, 16.0, 33.0 }, { 13.0, 30.0, 23.0 }, { 0 }),
            createYoloLayerParam(40, 40, 3, { 30.0, 62.0, 59.0 }, { 61.0, 45.0, 119.0 }, { 1 }),
            createYoloLayerParam(20, 20, 3, { 116.0, 156.0, 373.0 }, { 90.0, 198.0, 326.0 }, { 2 })
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
            createYoloLayerParam(20, 20, 3, { 116.0, 156.0, 373.0 }, { 90.0, 198.0, 326.0 }, { 0 }),
            createYoloLayerParam(40, 40, 3, { 30.0, 62.0, 59.0 }, { 61.0, 45.0, 119.0 }, { 1 }),
            createYoloLayerParam(80, 80, 3, { 10.0, 16.0, 33.0 }, { 13.0, 30.0, 23.0 }, { 2 })
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
            createYoloLayerParam(64, 64, 3, {}, {}, {0, 1, 2}),
            createYoloLayerParam(32, 32, 3, {}, {}, {3, 4, 5}),
            createYoloLayerParam(16, 16, 3, {}, {}, {6, 7, 8})
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
            createYoloLayerParam(20, 20, 3, { 142.0, 192.0, 459.0 }, { 110.0, 243.0, 401.0 }, { 0 }),
            createYoloLayerParam(40, 40, 3, { 36.0, 76.0, 72.0 }, { 75.0, 55.0, 146.0 }, { 1 }),
            createYoloLayerParam(80, 80, 3, { 12.0, 19.0, 40.0 }, { 16.0, 36.0, 28.0 }, { 2 })
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
            createYoloLayerParam(64, 64, 3, { 12.0, 19.0, 40.0 }, { 16.0, 36.0, 28.0 }, { 0 }),
            createYoloLayerParam(32, 32, 3, { 36.0, 76.0, 72.0 }, { 75.0, 55.0, 146.0 }, { 1 }),
            createYoloLayerParam(16, 16, 3, { 142.0, 192.0, 459.0 }, { 110.0, 243.0, 401.0 }, { 2 })
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
            createYoloLayerParam(76, 76, 3, { 12.0, 19.0, 40.0 }, { 16.0, 36.0, 28.0 }, { 0 }, 1.2, 1.2),
            createYoloLayerParam(38, 38, 3, { 36.0, 76.0, 72.0 }, { 75.0, 55.0, 146.0 }, { 1 }, 1.1, 1.1),
            createYoloLayerParam(19, 19, 3, { 142.0, 192.0, 459.0 }, { 110.0, 243.0, 401.0 }, { 2 }, 1.05, 1.05)
    },
    .classNames = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"},
};
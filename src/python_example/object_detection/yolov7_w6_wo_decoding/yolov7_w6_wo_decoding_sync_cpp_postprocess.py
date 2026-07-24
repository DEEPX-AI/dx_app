#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov7_w6_wo_decoding Synchronous Inference Example

Usage:
    python yolov7_w6_wo_decoding_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


import os
if os.name == 'nt':
    _dxrt_dir = os.environ.get('DEEPX_SDK_DIR')
    if _dxrt_dir:
        os.add_dll_directory(os.path.join(_dxrt_dir, 'bin'))

from dx_postprocess import YOLOv5PostProcess
from dx_engine import InferenceOption
from common.utility import convert_cpp_detections
from factory import Yolov7_w6_wo_decodingFactory
from factory.yolov7_w6_wo_decoding_factory import YOLOV7_W6_ANCHORS
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv7-W6 Sync Inference (C++ Postprocess)")
def main():
    args = parse_args()
    factory = Yolov7_w6_wo_decodingFactory()

    def on_engine_init(runner):
        use_ort = InferenceOption().get_use_ort()
        config = runner.factory.config
        obj_thr = config.get("obj_threshold", 0.25)
        conf_thr = config.get("conf_threshold", config.get("score_threshold", 0.3))
        nms_thr = config.get("nms_threshold", 0.45)
        post = YOLOv5PostProcess(
            runner.input_width, runner.input_height, obj_thr, conf_thr, nms_thr, use_ort)
        post.set_anchors({int(s): [(int(a[0]), int(a[1])) for a in al]
                          for s, al in YOLOV7_W6_ANCHORS.items()})
        runner._cpp_postprocessor = post
        runner._cpp_convert_fn = convert_cpp_detections

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()

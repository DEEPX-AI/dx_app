#!/usr/bin/env python3
# Copyright (C) 2018- DEEPX Ltd. All rights reserved.
"""
Yolov8Seg Synchronous Inference Example

Usage:
    python yolo26n_seg_sync_cpp_postprocess.py --model model.dxnn --image input.jpg
"""

import sys
from pathlib import Path

_module_dir = Path(__file__).parent
_v3_dir = _module_dir.parent.parent
for _path in [str(_v3_dir), str(_module_dir)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dx_postprocess import YOLOv8SegPostProcess
from dx_engine import InferenceOption
from common.utility.visualization import yolov8seg_cpp_visualize
from factory import Yolo26n_segFactory
from common.runner import SyncRunner, parse_common_args

def parse_args():
    return parse_common_args("YOLOv8n-SEG Sync Inference")
def main():
    args = parse_args()
    factory = Yolo26n_segFactory()

    def on_engine_init(runner):
        input_w = runner.input_width
        input_h = runner.input_height
        use_ort = InferenceOption().get_use_ort()
        config = runner.factory.config
        score_thr = config.get("score_threshold", 0.3)
        nms_thr = config.get("nms_threshold", 0.45)
        runner._cpp_postprocessor = YOLOv8SegPostProcess(input_w, input_h, score_thr, nms_thr, use_ort)
        runner._cpp_visualize_fn = yolov8seg_cpp_visualize

    runner = SyncRunner(factory, on_engine_init=on_engine_init)
    runner.run(args)

if __name__ == "__main__":
    main()

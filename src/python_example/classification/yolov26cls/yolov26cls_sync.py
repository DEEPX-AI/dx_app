import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.labels import get_labels
from utils.performance_summary import print_image_processing_summary


class YOLOv26Cls:

    def __init__(self, model_path: str):

        self.ie = InferenceEngine(model_path)

        if version.parse(self.ie.get_model_version()) < version.parse(
            "7"
        ):  # pragma: no cover
            print(
                "[ERROR] .dxnn format version 7 or higher is required. Please update DX-COM to the latest version and re-compile the ONNX model."
            )
            exit(1)

        input_tensors_info = self.ie.get_input_tensors_info()

        self.input_height = input_tensors_info[0]["shape"][1]
        self.input_width = input_tensors_info[0]["shape"][2]

        print(f"\n[INFO] Model loaded: {model_path}")
        print(f"[INFO] Model input size (WxH): {self.input_width}x{self.input_height}")

        self.classes = get_labels("imagenet1000")

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor = cv2.resize(
            img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )

        return input_tensor

    def postprocess(self, output_tensors: List[np.ndarray]) -> np.ndarray:

        output = output_tensors[0]

        probabilities = output.flatten() if output.ndim > 1 else output

        class_id = np.argmax(probabilities).item()
        confidence_score = probabilities[class_id].item()

        return class_id, confidence_score

    def image_inference(self, image_path: str):

        t_start = time.perf_counter()
        img = cv2.imread(image_path)

        if img is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            exit(1)

        print(f"[INFO] Input image: {image_path}")
        print(f"[INFO] Image resolution (WxH): {img.shape[1]}x{img.shape[0]}")

        t0 = time.perf_counter()
        input_tensor = self.preprocess(img)

        t1 = time.perf_counter()
        output_tensors = self.ie.run([input_tensor])

        t2 = time.perf_counter()
        class_id, confidence_score = self.postprocess(output_tensors)

        t3 = time.perf_counter()

        print_image_processing_summary(t_start, t0, t1, t2, t3)

        if confidence_score is not None:
            print(
                f"[SUCCESS] Predicted Class: '{self.classes[class_id]}' (ID: {class_id}, Confidence: {confidence_score:.2f})"
            )
        else:
            print(
                f"[SUCCESS] Predicted Class: '{self.classes[class_id]}' (ID: {class_id})"
            )


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Input your DXNN model."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover

    config = Configuration()
    if version.parse(config.get_version()) < version.parse("3.0.0"):
        print(
            "[ERROR] DX-RT v3.0.0 or higher is required. Please update DX-RT to the latest version."
        )
        exit(1)

    args = parse_arguments()

    if not os.path.exists(args.model):
        print(
            "[ERROR] .dxnn model file does not exist. Please input correct model path."
        )
        exit(1)

    if not os.path.exists(args.image):
        print("[ERROR] image file does not exist. Please input correct image path.")
        exit(1)

    model = YOLOv26Cls(args.model)

    model.image_inference(args.image)

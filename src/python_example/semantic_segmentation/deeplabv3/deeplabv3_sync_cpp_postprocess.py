import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine
from dx_postprocess import DeepLabv3PostProcess
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.performance_summary import (
    print_image_processing_summary,
    print_sync_performance_summary,
)


class DeepLabv3:

    def __init__(self, model_path: str):

        self.model_path = model_path

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

        self.postprocessor = DeepLabv3PostProcess(self.input_width, self.input_height)

        # Cityscapes 19-class color palette
        self.color_palette = np.array(
            [
                [128, 64, 128],  # 0: road
                [244, 35, 232],  # 1: sidewalk
                [70, 70, 70],  # 2: building
                [102, 102, 156],  # 3: wall
                [190, 153, 153],  # 4: fence
                [153, 153, 153],  # 5: pole
                [250, 170, 30],  # 6: traffic light
                [220, 220, 0],  # 7: traffic sign
                [107, 142, 35],  # 8: vegetation
                [152, 251, 152],  # 9: terrain
                [70, 130, 180],  # 10: sky
                [220, 20, 60],  # 11: person
                [255, 0, 0],  # 12: rider
                [0, 0, 142],  # 13: car
                [0, 0, 70],  # 14: truck
                [0, 60, 100],  # 15: bus
                [0, 80, 100],  # 16: train
                [0, 0, 230],  # 17: motorcycle
                [119, 11, 32],  # 18: bicycle
            ],
            dtype=np.uint8,
        )

    def draw_segmentation(self, img: np.ndarray, class_map: np.ndarray) -> np.ndarray:
        """Draw segmentation mask overlay on image"""

        h, w = img.shape[:2]
        class_map_resized = cv2.resize(
            class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )

        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(len(self.color_palette)):
            if class_id == 0:
                continue
            mask = class_map_resized == class_id
            if np.any(mask):
                colored_mask[mask] = self.color_palette[class_id]

        alpha = 0.6
        cv2.addWeighted(img, 1 - alpha, colored_mask, alpha, 0, dst=img)

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor = cv2.resize(
            img, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR
        )

        return input_tensor

    def image_inference(self, image_path: str, display: bool = True):

        t_start = time.perf_counter()
        img = cv2.imread(image_path)

        if img is None:
            print(f"[ERROR] Failed to load image: {image_path}")
            exit(1)

        print(f"\n[INFO] Input image: {image_path}")
        print(f"[INFO] Image resolution (WxH): {img.shape[1]}x{img.shape[0]}")

        t0 = time.perf_counter()
        input_tensor = self.preprocess(img)

        t1 = time.perf_counter()
        output_tensors = self.ie.run([input_tensor])

        t2 = time.perf_counter()
        class_map = self.postprocessor.postprocess(output_tensors)

        t3 = time.perf_counter()
        self.draw_segmentation(img, class_map)
        t4 = time.perf_counter()

        print_image_processing_summary(t_start, t0, t1, t2, t3, t4)

        if display:
            cv2.imshow("Output", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            root = Path(__file__).absolute().parents[4]
            out_dir = root / "artifacts" / "python_example" / "semantic_segmentation"
            out_dir.mkdir(parents=True, exist_ok=True)

            script = Path(__file__).stem
            model_name = Path(self.model_path).stem
            input_name = Path(image_path).stem
            save_path = out_dir / f"{script}-{model_name}-{input_name}.jpg"

            cv2.imwrite(save_path, img)

            print(f"[SUCCESS] Output saved: {save_path}")

    def stream_inference(self, source, display: bool = True):

        metrics = {
            "sum_read": 0.0,
            "sum_preprocess": 0.0,
            "sum_inference": 0.0,
            "sum_postprocess": 0.0,
            "sum_render": 0.0,
        }

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open input source: {source}")
            exit(1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"\n[INFO] Camera index: {source}")
        elif isinstance(source, str) and source.startswith("rtsp://"):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"\n[INFO] RTSP URL: {source}")
        else:
            print(f"\n[INFO] Video file: {source}")

        print(f"[INFO] Input source resolution (WxH): {width}x{height}")

        if total_frames > 0:
            print(f"[INFO] Total frames: {total_frames}")
        if fps > 0:
            print(f"[INFO] Input source FPS: {fps:.2f}")

        print("\n[INFO] Starting inference...")

        try:
            cnt = 0
            start_time = time.perf_counter()
            while True:

                t_start = time.perf_counter()
                ok, frame_bgr = cap.read()

                if not ok:
                    break

                cnt += 1

                t0 = time.perf_counter()
                input_tensor = self.preprocess(frame_bgr)

                t1 = time.perf_counter()
                output_tensors = self.ie.run([input_tensor])

                t2 = time.perf_counter()
                class_map = self.postprocessor.postprocess(output_tensors)

                t3 = time.perf_counter()

                metrics["sum_read"] += t0 - t_start
                metrics["sum_preprocess"] += t1 - t0
                metrics["sum_inference"] += t2 - t1
                metrics["sum_postprocess"] += t3 - t2

                if display:
                    self.draw_segmentation(frame_bgr, class_map)

                    cv2.imshow("Output", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF

                    t4 = time.perf_counter()
                    metrics["sum_render"] += t4 - t3

                    if key == ord("q") or key == 27:
                        print("\n[INFO] User requested to quit")
                        break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
        finally:
            if cnt == 0:
                print("[WARNING] No frames processed")
            else:
                elapsed = time.perf_counter() - start_time
                print_sync_performance_summary(metrics, cnt, elapsed, display)

            cap.release()
            cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Input your DXNN model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to input image.")
    group.add_argument("--video", type=str, help="Path to input video.")
    group.add_argument(
        "--camera", type=int, help="Camera device index (e.g., 0 for default camera)."
    )
    group.add_argument(
        "--rtsp", type=str, help="RTSP stream URL (e.g., rtsp://ip:port/stream)."
    )
    parser.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Do not display output window.",
    )
    parser.set_defaults(display=True)
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

    if args.image:
        if not os.path.exists(args.image):
            print("[ERROR] image file does not exist. Please input correct image path.")
            exit(1)

    elif args.video:
        if not os.path.exists(args.video):
            print("[ERROR] video file does not exist. Please input correct video path.")
            exit(1)

    model = DeepLabv3(args.model)

    if args.image:
        model.image_inference(args.image, display=args.display)
    elif args.video:
        model.stream_inference(args.video, display=args.display)
    elif args.camera is not None:
        model.stream_inference(args.camera, display=args.display)
    elif args.rtsp:
        model.stream_inference(args.rtsp, display=args.display)

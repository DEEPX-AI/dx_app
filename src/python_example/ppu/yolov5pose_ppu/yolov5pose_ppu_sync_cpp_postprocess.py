import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine, InferenceOption
from dx_postprocess import YOLOv5PosePPUPostProcess
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.performance_summary import (
    print_image_processing_summary,
    print_sync_performance_summary,
)

SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6],   [5, 7],   [6, 8],   [7, 9],   [8, 10],  [1, 2],  [0, 1],
    [0, 2],   [1, 3],   [2, 4],   [3, 5],   [4, 6]
]

POSE_LIMB_COLOR = [
    (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
    (255, 51, 255), (255, 51, 255), (255, 51, 255), (255, 128, 0),
    (255, 128, 0),  (255, 128, 0),  (255, 128, 0),  (255, 128, 0),
    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),
    (0, 255, 0),    (0, 255, 0),    (0, 255, 0)
]

POSE_KPT_COLOR = [
    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),    (0, 255, 0),
    (0, 255, 0),    (255, 128, 0),  (255, 128, 0),  (255, 128, 0),
    (255, 128, 0),  (255, 128, 0),  (255, 128, 0),  (51, 153, 255),
    (51, 153, 255), (51, 153, 255), (51, 153, 255), (51, 153, 255),
    (51, 153, 255)
]

class YOLOv5Pose_PPU:

    def __init__(self, model_path: str):

        self.model_path = model_path

        option = InferenceOption()
        if not option.get_use_ort():  # pragma: no cover
            print(
                "[ERROR] USE_ORT=OFF is not supported in this example. Please build DX-RT with USE_ORT=ON option."
            )
            exit(1)

        self.ie = InferenceEngine(model_path)

        # if version.parse(self.ie.get_model_version()) < version.parse(
        #     "7"
        # ):  # pragma: no cover
        #     print(
        #         "[ERROR] .dxnn format version 7 or higher is required. Please update DX-COM to the latest version and re-compile the ONNX model."
        #     )
        #     exit(1)

        input_tensors_info = self.ie.get_input_tensors_info()

        self.input_height = input_tensors_info[0]["shape"][1]
        self.input_width = input_tensors_info[0]["shape"][2]

        print(f"\n[INFO] Model loaded: {model_path}")
        print(f"[INFO] Model input size (WxH): {self.input_width}x{self.input_height}")

        self.score_threshold = 0.3
        self.nms_threshold = 0.45
        self.classes = ["person"]
        self.num_keypoints = 17

        self.postprocessor = YOLOv5PosePPUPostProcess(
            self.input_width,
            self.input_height,
            self.score_threshold,
            self.nms_threshold,
        )

        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def letterbox(
        self, img: np.ndarray, new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:

        shape = img.shape[:2]

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (top, left)

    def convert_to_original_coordinates(self, detections: np.ndarray) -> np.ndarray:

        if len(detections) == 0:
            return detections

        detections[:, 0] = np.clip(
            (detections[:, 0] - self.pad[1]) / self.gain, 0, self.img_width - 1
        )
        detections[:, 1] = np.clip(
            (detections[:, 1] - self.pad[0]) / self.gain, 0, self.img_height - 1
        )
        detections[:, 2] = np.clip(
            (detections[:, 2] - self.pad[1]) / self.gain, 0, self.img_width - 1
        )
        detections[:, 3] = np.clip(
            (detections[:, 3] - self.pad[0]) / self.gain, 0, self.img_height - 1
        )

        for i in range(self.num_keypoints):
            detections[:, 6 + i * 3] = np.clip(
                (detections[:, 6 + i * 3] - self.pad[1]) / self.gain,
                0,
                self.img_width - 1,
            )
            detections[:, 6 + i * 3 + 1] = np.clip(
                (detections[:, 6 + i * 3 + 1] - self.pad[0]) / self.gain,
                0,
                self.img_height - 1,
            )

        return detections

    def draw_detections(self, img: np.ndarray, detections: np.ndarray) -> None:

        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection[:6]
            keypoints = detection[6:]

            color = self.color_palette[int(class_id)]

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            label = f"{self.classes[int(class_id)]}: {score:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )

            label_x = int(x1)
            label_y = int(y1) - 10 if int(y1) - 10 > label_height else int(y1) + 10

            cv2.rectangle(
                img,
                (label_x, label_y - label_height),
                (label_x + label_width, label_y + label_height),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                img,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

            for j, sk in enumerate(SKELETON):
                idx1, idx2 = sk[0], sk[1]
                pt1_x = int(keypoints[idx1 * 3])
                pt1_y = int(keypoints[idx1 * 3 + 1])
                pt2_x = int(keypoints[idx2 * 3])
                pt2_y = int(keypoints[idx2 * 3 + 1])
                cv2.line(img, (pt1_x, pt1_y), (pt2_x, pt2_y), POSE_LIMB_COLOR[j], 2, cv2.LINE_AA)

            for k in range(self.num_keypoints):
                kx = int(keypoints[k * 3])
                ky = int(keypoints[k * 3 + 1])
                cv2.circle(img, (kx, ky), 3, (0, 0, 255), -1)

    def preprocess(self, img: np.ndarray) -> np.ndarray:

        self.img_height, self.img_width = img.shape[:2]
        self.gain = min(
            self.input_height / self.img_height, self.input_width / self.img_width
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_tensor, self.pad = self.letterbox(
            img, (self.input_width, self.input_height)
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
        detections = self.postprocessor.postprocess(output_tensors)

        t3 = time.perf_counter()
        detections_scaled = self.convert_to_original_coordinates(detections)

        self.draw_detections(img, detections_scaled)
        t4 = time.perf_counter()

        print_image_processing_summary(t_start, t0, t1, t2, t3, t4)

        if display:
            cv2.imshow("Output", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            root = Path(__file__).absolute().parents[4]
            out_dir = root / "artifacts" / "python_example" / "object_detection"
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
                detections = self.postprocessor.postprocess(output_tensors)

                t3 = time.perf_counter()

                metrics["sum_read"] += t0 - t_start
                metrics["sum_preprocess"] += t1 - t0
                metrics["sum_inference"] += t2 - t1
                metrics["sum_postprocess"] += t3 - t2

                if display:
                    detections_scaled = self.convert_to_original_coordinates(detections)
                    self.draw_detections(frame_bgr, detections_scaled)

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

    model = YOLOv5Pose_PPU(args.model)

    if args.image:
        model.image_inference(args.image, display=args.display)
    elif args.video:
        model.stream_inference(args.video, display=args.display)
    elif args.camera is not None:
        model.stream_inference(args.camera, display=args.display)
    elif args.rtsp:
        model.stream_inference(args.rtsp, display=args.display)

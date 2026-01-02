import argparse
import os
import queue
import sys
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine, InferenceOption
from dx_postprocess import YOLOv7PPUPostProcess
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.labels import get_labels
from utils.performance_summary import print_async_performance_summary


class YOLOv7_PPU:

    def __init__(self, model_path: str):

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

        self.obj_threshold = 0.25
        self.score_threshold = 0.3
        self.nms_threshold = 0.45
        self.classes = get_labels("coco80")

        self.postprocessor = YOLOv7PPUPostProcess(
            self.input_width,
            self.input_height,
            self.obj_threshold,
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

        return detections

    def draw_detections(self, img: np.ndarray, detections: np.ndarray) -> None:

        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection

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

    def stream_inference(self, source, display: bool = True):

        metrics = {
            "sum_read": 0.0,
            "sum_preprocess": 0.0,
            "sum_inference": 0.0,
            "sum_postprocess": 0.0,
            "sum_render": 0.0,
            "infer_completed": 0,
            "infer_first_ts": None,
            "infer_last_ts": None,
            "inflight_last_ts": None,
            "inflight_current": 0,
            "inflight_max": 0,
            "inflight_time_sum": 0.0,
        }
        metrics_lock = threading.Lock()

        input_image_queue: "queue.Queue[tuple]" = queue.Queue()
        req_id_queue: "queue.Queue[tuple]" = queue.Queue()
        output_tensor_queue: "queue.Queue[tuple]" = queue.Queue()
        if display:
            detections_queue: "queue.Queue[tuple]" = queue.Queue()

        stop_event = threading.Event()
        SENTINEL = object()

        def set_stop_event():
            stop_event.set()
            queues = [input_image_queue, req_id_queue, output_tensor_queue]
            if display:
                queues.append(detections_queue)

            for q_ in queues:
                try:
                    while True:
                        _ = q_.get_nowait()
                except queue.Empty:
                    pass

            input_image_queue.put(SENTINEL)
            req_id_queue.put(SENTINEL)
            output_tensor_queue.put(SENTINEL)
            if display:
                detections_queue.put(SENTINEL)

        def preprocess_worker():
            while True:
                item = input_image_queue.get()
                try:
                    if item is SENTINEL or stop_event.is_set():
                        req_id_queue.put(SENTINEL)
                        break

                    frame_bgr, meta = item

                    t0 = time.perf_counter()
                    input_tensor = self.preprocess(frame_bgr)
                    t1 = time.perf_counter()

                    meta["t_preprocess"] = t1 - t0
                    meta["t_run_async_start"] = t1

                    req_id = self.ie.run_async([input_tensor])
                    t2 = time.perf_counter()

                    # Pass input_tensor along to maintain memory reference until async inference completes
                    req_id_queue.put((frame_bgr, input_tensor, req_id, meta))

                    with metrics_lock:

                        if metrics["infer_first_ts"] is None:
                            metrics["infer_first_ts"] = t1

                        if metrics["inflight_last_ts"] is None:
                            metrics["inflight_last_ts"] = t2
                        else:
                            dt = t2 - metrics["inflight_last_ts"]
                            metrics["inflight_time_sum"] += (
                                metrics["inflight_current"] * dt
                            )
                            metrics["inflight_last_ts"] = t2

                        metrics["inflight_current"] += 1
                        if metrics["inflight_current"] > metrics["inflight_max"]:
                            metrics["inflight_max"] = metrics["inflight_current"]

                finally:
                    pass

        def wait_worker():
            while True:
                item = req_id_queue.get()
                try:
                    if item is SENTINEL or stop_event.is_set():
                        output_tensor_queue.put(SENTINEL)
                        break

                    frame_bgr, input_tensor, req_id, meta = item

                    output_tensors = self.ie.wait(req_id)

                    t0 = time.perf_counter()
                    meta["t_inference"] = t0 - meta["t_run_async_start"]

                    output_tensor_queue.put((frame_bgr, output_tensors, meta))

                    with metrics_lock:

                        metrics["infer_last_ts"] = t0
                        metrics["infer_completed"] += 1

                        dt = t0 - metrics["inflight_last_ts"]
                        metrics["inflight_time_sum"] += metrics["inflight_current"] * dt
                        metrics["inflight_last_ts"] = t0

                        metrics["inflight_current"] = metrics["inflight_current"] - 1

                finally:
                    pass

        def postprocess_worker():
            while True:
                item = output_tensor_queue.get()
                try:
                    if item is SENTINEL or stop_event.is_set():
                        if display:
                            detections_queue.put(SENTINEL)
                        break

                    frame_bgr, output_tensors, meta = item

                    t0 = time.perf_counter()
                    detections = self.postprocessor.postprocess(output_tensors)
                    meta["t_postprocess"] = time.perf_counter() - t0

                    if display:
                        detections_queue.put((frame_bgr, detections))

                    with metrics_lock:
                        metrics["sum_read"] += meta["t_read"]
                        metrics["sum_preprocess"] += meta["t_preprocess"]
                        metrics["sum_inference"] += meta["t_inference"]
                        metrics["sum_postprocess"] += meta["t_postprocess"]

                finally:
                    pass

        def render_worker():
            while True:
                output_item = detections_queue.get()
                try:
                    if output_item is SENTINEL or stop_event.is_set():
                        break

                    frame_bgr, detections = output_item

                    t0 = time.perf_counter()
                    detections_scaled = self.convert_to_original_coordinates(detections)
                    self.draw_detections(frame_bgr, detections_scaled)

                    cv2.imshow("Output", frame_bgr)
                    key = cv2.waitKey(1) & 0xFF

                    with metrics_lock:
                        metrics["sum_render"] += time.perf_counter() - t0

                    if key == ord("q") or key == 27:
                        set_stop_event()
                        break
                finally:
                    pass

        threads = [
            threading.Thread(target=preprocess_worker, daemon=True),
            threading.Thread(target=wait_worker, daemon=True),
            threading.Thread(target=postprocess_worker, daemon=True),
        ]
        if display:
            threads.append(threading.Thread(target=render_worker, daemon=True))

        for t in threads:
            t.start()

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
            while not stop_event.is_set():

                t0 = time.perf_counter()
                ok, frame_bgr = cap.read()

                if not ok:
                    break

                cnt += 1

                t1 = time.perf_counter()
                meta = {"t_read": t1 - t0}

                input_image_queue.put((frame_bgr, meta))

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user (Ctrl+C)")
            set_stop_event()
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            set_stop_event()
        finally:

            if not stop_event.is_set():
                input_image_queue.put(SENTINEL)

            for t in threads:
                t.join()

            if metrics["infer_completed"] == 0:
                print("[WARNING] No frames were processed.")
            else:
                elapsed = time.perf_counter() - start_time
                print_async_performance_summary(metrics, cnt, elapsed, display)

            cap.release()
            cv2.destroyAllWindows()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, help="Input your DXNN model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
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

    if args.video:
        if not os.path.exists(args.video):
            print("[ERROR] video file does not exist. Please input correct video path.")
            exit(1)

    model = YOLOv7_PPU(args.model)

    if args.video:
        model.stream_inference(args.video, display=args.display)
    elif args.camera is not None:
        model.stream_inference(args.camera, display=args.display)
    elif args.rtsp:
        model.stream_inference(args.rtsp, display=args.display)

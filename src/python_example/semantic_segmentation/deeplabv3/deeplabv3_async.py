import argparse
import os
import queue
import sys
import threading
import time
from typing import List, Tuple

import cv2
import numpy as np
from dx_engine import Configuration, InferenceEngine
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.performance_summary import print_async_performance_summary


class DeepLabv3:

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

    def postprocess(self, output_tensors: List[np.ndarray]) -> np.ndarray:

        class_map = np.argmax(output_tensors[0][0], axis=0)

        return class_map

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
            class_map_queue: "queue.Queue[tuple]" = queue.Queue()

        stop_event = threading.Event()
        SENTINEL = object()

        def set_stop_event():
            stop_event.set()
            queues = [input_image_queue, req_id_queue, output_tensor_queue]
            if display:
                queues.append(class_map_queue)

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
                class_map_queue.put(SENTINEL)

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
                            class_map_queue.put(SENTINEL)
                        break

                    frame_bgr, output_tensors, meta = item

                    t0 = time.perf_counter()
                    class_map = self.postprocess(output_tensors)
                    meta["t_postprocess"] = time.perf_counter() - t0

                    if display:
                        class_map_queue.put((frame_bgr, class_map))

                    with metrics_lock:
                        metrics["sum_read"] += meta["t_read"]
                        metrics["sum_preprocess"] += meta["t_preprocess"]
                        metrics["sum_inference"] += meta["t_inference"]
                        metrics["sum_postprocess"] += meta["t_postprocess"]

                finally:
                    pass

        def render_worker():
            while True:
                output_item = class_map_queue.get()
                try:
                    if output_item is SENTINEL or stop_event.is_set():
                        break

                    frame_bgr, class_map = output_item

                    t0 = time.perf_counter()
                    self.draw_segmentation(frame_bgr, class_map)

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

    model = DeepLabv3(args.model)

    if args.video:
        model.stream_inference(args.video, display=args.display)
    elif args.camera is not None:
        model.stream_inference(args.camera, display=args.display)
    elif args.rtsp:
        model.stream_inference(args.rtsp, display=args.display)

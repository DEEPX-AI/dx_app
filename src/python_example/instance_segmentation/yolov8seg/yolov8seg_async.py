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
from packaging import version

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.labels import get_labels
from utils.performance_summary import print_async_performance_summary


class YOLOv8Seg:

    def __init__(self, model_path: str):

        option = InferenceOption()
        if not option.get_use_ort():  # pragma: no cover
            print(
                "[ERROR] USE_ORT=OFF is not supported in this example. Please build DX-RT with USE_ORT=ON option."
            )
            exit(1)

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

        self.score_threshold = 0.3
        self.nms_threshold = 0.45
        self.classes = get_labels("coco80")

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

    def convert_to_original_coordinates(
        self, detections: np.ndarray, masks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if len(detections) == 0:
            return detections, masks

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

        if len(masks) > 0:
            # Calculate the dimensions of the resized image inside the letterbox
            unpad_h = int(self.img_height * self.gain)
            unpad_w = int(self.img_width * self.gain)

            top, left = int(self.pad[0]), int(self.pad[1])

            # Slice the valid area
            masks = masks[:, top : top + unpad_h, left : left + unpad_w]

            new_masks = np.zeros(
                (len(masks), self.img_height, self.img_width), dtype=np.float32
            )
            for i, mask in enumerate(masks):
                new_masks[i] = cv2.resize(
                    mask,
                    (self.img_width, self.img_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            return detections, new_masks

        return detections, masks

    def draw_detections(
        self, img: np.ndarray, detections: np.ndarray, masks: np.ndarray
    ) -> None:

        for i, mask in enumerate(masks):
            class_id = int(detections[i, 5])
            color = self.color_palette[class_id]

            m = mask > 0.5
            img[m] = (img[m] * 0.6 + np.array(color) * 0.4).astype(np.uint8)

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

    def postprocess(
        self, output_tensors: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:

        outputs = np.transpose(np.squeeze(output_tensors[0]))

        cls_scores = outputs[:, 4:84]
        cls_max_scores = np.max(cls_scores, axis=1)
        cls_ids = np.argmax(cls_scores, axis=1)
        mask_coefs = outputs[:, 84:]

        # (center_x, center_y, width, height)
        boxes_cxcywh = outputs[:, :4]

        # (left, top, right, bottom)
        boxes_x1y1x2y2 = np.column_stack(
            [
                boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] * 0.5,
                boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] * 0.5,
                boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] * 0.5,
                boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] * 0.5,
            ]
        )

        # (left, top, width, height)
        boxes_x1y1wh = np.column_stack(
            [
                boxes_x1y1x2y2[:, 0],
                boxes_x1y1x2y2[:, 1],
                boxes_x1y1x2y2[:, 2] - boxes_x1y1x2y2[:, 0],
                boxes_x1y1x2y2[:, 3] - boxes_x1y1x2y2[:, 1],
            ]
        )

        indices = cv2.dnn.NMSBoxes(
            boxes_x1y1wh.tolist(),
            cls_max_scores.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )

        if len(indices) > 0:
            keep = np.array(indices).reshape(-1)
            detections = np.column_stack(
                [boxes_x1y1x2y2[keep], cls_max_scores[keep], cls_ids[keep]]
            ).astype(np.float32)

            masks_coef = mask_coefs[keep]
            proto = np.squeeze(output_tensors[1])
            c, mh, mw = proto.shape

            masks = masks_coef @ proto.reshape(c, -1)
            masks = 1 / (1 + np.exp(-masks))
            masks = masks.reshape(-1, mh, mw)

            scaled_masks = np.zeros(
                (len(masks), self.input_height, self.input_width), dtype=np.float32
            )
            for i, mask in enumerate(masks):
                scaled_masks[i] = cv2.resize(
                    mask,
                    (self.input_width, self.input_height),
                    interpolation=cv2.INTER_LINEAR,
                )

            for i, box in enumerate(detections[:, :4]):
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self.input_width, x2), min(self.input_height, y2)

                scaled_masks[i, :y1, :] = 0
                scaled_masks[i, y2:, :] = 0
                scaled_masks[i, :, :x1] = 0
                scaled_masks[i, :, x2:] = 0

            return detections, scaled_masks

        return np.empty((0, 6), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)

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
                    detections, masks = self.postprocess(output_tensors)
                    meta["t_postprocess"] = time.perf_counter() - t0

                    if display:
                        detections_queue.put((frame_bgr, detections, masks))

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

                    frame_bgr, detections, masks = output_item

                    t0 = time.perf_counter()
                    detections_scaled, masks_scaled = (
                        self.convert_to_original_coordinates(detections, masks)
                    )
                    self.draw_detections(frame_bgr, detections_scaled, masks_scaled)

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

    model = YOLOv8Seg(args.model)

    if args.video:
        model.stream_inference(args.video, display=args.display)
    elif args.camera is not None:
        model.stream_inference(args.camera, display=args.display)
    elif args.rtsp:
        model.stream_inference(args.rtsp, display=args.display)

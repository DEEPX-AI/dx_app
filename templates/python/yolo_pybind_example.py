import os, cv2, json, time, argparse, queue, threading, sys
import numpy as np
from packaging import version
from dx_engine import InferenceEngine, Configuration
from dx_postprocess import YoloPostProcess


def draw_detections(frame_v, pp_output, colors):
    for i in range(pp_output.shape[0]):
        pt1 = pp_output[i, :2].astype(int)
        pt2 = pp_output[i, 2:4].astype(int)
        class_id = int(pp_output[i, 5]) if pp_output.shape[1] >= 6 else 0
        cv2.rectangle(frame_v, pt1, pt2, colors[class_id % len(colors)], 2)
        if pp_output.shape[1] > 6:
            kpts = pp_output[i, 6:]
            kpts_reshape = kpts.reshape(-1, 3)
            for k in range(kpts_reshape.shape[0]):
                kpt = kpts_reshape[k, :2].astype(int)
                cv2.circle(frame_v, kpt, 1, (0, 0, 255), -1)
    return frame_v


class UserArgs:
    def __init__(self, frame, ratio, pad, target_output_tensor_idx, output_queue:queue.Queue):
        self.frame = frame
        self.ratio = ratio
        self.pad = pad
        self.target_output_tensor_idx = target_output_tensor_idx
        self.output_queue = output_queue


def post_process_worker(ypp:YoloPostProcess, output_queue:queue.Queue, pp_output_queue:queue.Queue):
    while True:
        item = output_queue.get()
        try:
            if item is None:
                # sentinel for shutdown
                break
            frame, ie_outputs, ratio, pad = item
            pp_output = ypp.Run(ie_outputs, ratio, pad)
            pp_output_queue.put((frame, pp_output))
        finally:
            output_queue.task_done()


def wait_worker(ie:InferenceEngine, req_id_queue:queue.Queue, output_queue:queue.Queue):
    while True:
        item = req_id_queue.get()
        try:
            if item is None:
                break
            req_id, frame, ratio, pad, target_idx = item
            ie_output = ie.wait(req_id)
            if target_idx is not None:
                ie_output = [ie_output[target_idx]]
            output_queue.put((frame, ie_output, ratio, pad))
        finally:
            req_id_queue.task_done()


def letter_box(image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=None):
    src_shape = image_src.shape[:2] # height, width
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / src_shape[0], new_shape[1] / src_shape[1])

    ratio = r, r  
    new_unpad = int(round(src_shape[1] * r)), int(round(src_shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  

    dw /= 2 
    dh /= 2

    if src_shape[::-1] != new_unpad:  
        image_src = cv2.resize(image_src, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image_new = cv2.copyMakeBorder(image_src, top, bottom, left, right, cv2.BORDER_CONSTANT, value=fill_color)  # add border
    if format is not None:
        image_new = cv2.cvtColor(image_new, format)
    
    return image_new, ratio, (dw, dh)


def run_example(args):

    config = Configuration()
    if version.parse(config.get_version()) < version.parse('3.0.0'):
        print("DX-RT version 3.0.0 or higher is required. Please update DX-RT to the latest version.")
        exit()
    
    # Load json config
    with open(args.config_path, "r") as f:
        json_config = json.load(f)
    
    # Check Input Size
    input_size = json_config['model']["param"]['input_size']
    
    # Initialize InferenceEngine
    ie = InferenceEngine(json_config["model"]["path"])

    if version.parse(ie.get_model_version()) < version.parse('7'):
        print("DXNN file format version 7 or higher is required. Please update/re-export the model.")
        exit()

    output_tensors_info = ie.get_output_tensors_info()
    target_output_tensor_idx = None
    if 'target_output_tensor_name' in json_config:
        for i, tensor_info in enumerate(output_tensors_info):
            if tensor_info['name'] == json_config['target_output_tensor_name']:
                target_output_tensor_idx = i
                break
    
    if target_output_tensor_idx is None:
        layer_info = json_config['model']['param']['layer']
        # Reorder layers to follow output tensors' name order
        output_name_order = [ti.get('name') for ti in output_tensors_info if 'name' in ti]
        name_to_layer = {layer.get('name'): layer for layer in layer_info if 'name' in layer}
        # Validate that all output tensor names exist in layer_info
        missing = [name for name in output_name_order if name not in name_to_layer]
        if missing:
            print(f"[Error] Output tensor name(s) not found in model.param.layer: {missing}. Please update the 'name' fields in model.param.layer to match the output tensor names.")
            sys.exit(1)
        reordered_layers = [name_to_layer[name] for name in output_name_order if name in name_to_layer]
        json_config['model']['param']['layer'] = reordered_layers
    
    # Initialize Post-Processor
    ypp = YoloPostProcess(json_config)

    # Queues for pipeline
    req_id_queue = queue.Queue()
    output_queue = queue.Queue()
    pp_output_queue = queue.Queue()

    # Worker threads
    pp_thread = threading.Thread(target=post_process_worker, args=(ypp, output_queue, pp_output_queue), daemon=True)
    pp_thread.start()
    wait_thread = threading.Thread(target=wait_worker, args=(ie, req_id_queue, output_queue), daemon=True)
    wait_thread.start()

    # Visualization color map (main thread only)
    colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()
    
    # Initialize VideoCapture
    cap = cv2.VideoCapture(args.video_path)
    
    # Process video frames
    frame_idx = 0
    start = time.time()
    stop = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # PreProcessing
        input_tensor, ratio, pad = letter_box(frame, (input_size, input_size), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        
        if args.run_async: 
            req_id = ie.run_async([input_tensor])
            # pass a copy of frame to decouple lifetime
            req_id_queue.put((req_id, frame.copy(), ratio, pad, target_output_tensor_idx))
        else:
            # Synchronous: enqueue directly
            ie_output = ie.run([input_tensor])
            if target_output_tensor_idx is not None:
                ie_output = [ie_output[target_output_tensor_idx]]
            output_queue.put((frame.copy(), ie_output, ratio, pad))

        # Consume available post-processed outputs and visualize
        try:
            while True:
                frame_v, pp_output = pp_output_queue.get_nowait()
                if args.visualize:
                    frame_v = draw_detections(frame_v, pp_output, colors)
                    cv2.imshow('result', frame_v)
                    if cv2.waitKey(1) == ord('q'):
                        stop = True
                pp_output_queue.task_done()
        except queue.Empty:
            pass

        if stop:
            break

    # Ensure async waits complete
    if args.run_async:
        try:
            req_id_queue.join()
        except Exception:
            pass

    # Ensure all queued outputs are processed by worker
    try:
        output_queue.join()
    except Exception:
        pass

    # Flush and visualize any remaining post-processed outputs
    try:
        while True:
            frame_v, pp_output = pp_output_queue.get_nowait()
            if args.visualize:
                frame_v = draw_detections(frame_v, pp_output, colors)
                cv2.imshow('result', frame_v)
                cv2.waitKey(1)
            pp_output_queue.task_done()
    except queue.Empty:
        pass

    # Stop worker threads
    req_id_queue.put(None)
    output_queue.put(None)
    wait_thread.join(timeout=2.0)
    pp_thread.join(timeout=2.0)

    cv2.destroyAllWindows()

    end = time.time()
    print("FPS : ", frame_idx / (end - start))
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./example/run_detector/yolov7_example.json', type=str, help='yolo object detection json config path')
    parser.add_argument('--video_path', required=True, type=str, help='input video path')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize post-process results')
    parser.add_argument('--run_async', action='store_true', dest='run_async', help='run the inference engine asynchronously')
    args = parser.parse_args()

    if not os.path.exists(args.config_path):
        parser.print_help()
        exit()
        
    run_example(args)
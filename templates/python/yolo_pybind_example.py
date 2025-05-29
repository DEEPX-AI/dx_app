import os, cv2, json, time, argparse, queue, threading
import numpy as np
from dx_engine import InferenceEngine
from dx_engine import version as dx_version
from dx_postprocess import YoloPostProcess

class UserArgs:
    def __init__(self, ypp:YoloPostProcess, frame, ratio, pad):
        self.ypp_ = ypp
        self.frame_ = frame
        self.ratio_ = ratio
        self.pad_ = pad

        
def visualize(args):
    
    colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()

    while True:
        if q.qsize() > 0:

            q_item = q.get()
            q.task_done()
            
            if q_item is None:
                cv2.destroyAllWindows()
                break
            
            if args.visualize:
                
                frame = q_item[0]
                pp_output = q_item[1]
                
                # visualize
                for i in range(pp_output.shape[0]):
                    pt1, pt2, score, class_id = pp_output[i, :2].astype(int), pp_output[i, 2:4].astype(int), pp_output[i, 4], pp_output[i, 5].astype(int)
                    frame = cv2.rectangle(frame, pt1, pt2, colors[class_id], 2)
                    
                    if pp_output.shape[1] > 6:
                        kpts = pp_output[i, 6:]
                        kpts_reshape = kpts.reshape(-1, 3)
                        for k in range(kpts_reshape.shape[0]):
                            kpt = kpts_reshape[k, :2].astype(int)
                            kpt_score = kpts_reshape[k, -1]
                            cv2.circle(frame, kpt, 1, (0, 0, 255), -1)
                            
                cv2.imshow('result', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
        else: 
            time.sleep(0.0001)
                
def pp_callback(ie_outputs, user_args):
    if dx_version.__version__ == '1.1.0':
        value:UserArgs = user_args
    else:
        value:UserArgs = user_args.value
    pp_output_ = value.ypp_.Run(ie_outputs, value.ratio_, value.pad_)
    q.put([value.frame_, pp_output_])

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
    
    # Load json config
    with open(args.config_path, "r") as f:
        json_config = json.load(f)
    
    # Check Input Size
    input_size = json_config['model']["param"]['input_size']
    
    # Initialize InferenceEngine
    ie = InferenceEngine(json_config["model"]["path"])
    
    if args.run_async: 
        # Register callback function 
        ie.register_callback(pp_callback)
        
        # Thread for Visualization
        vis_thread = threading.Thread(target=visualize, args=(args,))
        vis_thread.start()
    else:
        # Make color map
        colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()
    
    # Initialize YoloPostProcess
    ypp = YoloPostProcess(json_config)
    
    # Initialize VideoCapture
    cap = cv2.VideoCapture(args.video_path)
    
    # Process video frames
    frame_idx = 0
    start = time.time()
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_idx += 1

        # if frame_idx > 200:
        #     json_config['model']["param"]['conf_threshold'] = 0.9
        #     json_config['model']["param"]['score_threshold'] = 0.9
        #     ypp.SetConfig(json_config)
        
        # PreProcessing
        input_tensor, ratio, pad = letter_box(frame, (input_size, input_size), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        
        if args.run_async: 
            # UserArgs for callback function
            user_args = UserArgs(ypp, frame, ratio, pad)
        
            # Run the inference engine asynchronously
            req_id = ie.run_async([input_tensor], user_args)
        else:
            # Run the inference engine synchronously
            ie_output = ie.run([input_tensor])
            
            # PostProcessing
            pp_output = ypp.Run(ie_output, ratio, pad)
            
            # Visualizing
            if args.visualize:
                
                for i in range(pp_output.shape[0]):
                    
                    pt1, pt2, score, class_id = pp_output[i, :2].astype(int), pp_output[i, 2:4].astype(int), pp_output[i, 4], pp_output[i, 5].astype(int)
                    input_tensor = cv2.rectangle(frame, pt1, pt2, colors[class_id], 2)
                    
                    if pp_output.shape[1] > 6:
                        kpts = pp_output[i, 6:]
                        kpts_reshape = kpts.reshape(-1, 3)
                        for k in range(kpts_reshape.shape[0]):
                            kpt = kpts_reshape[k, :2].astype(int)
                            kpt_score = kpts_reshape[k, -1]
                            cv2.circle(frame, kpt, 1, (0, 0, 255), -1)
                cv2.imshow('result', frame)
                if cv2.waitKey(1) == ord('q'):
                    cv2.destroyAllWindows()
                    break
        
        if args.run_async:
            if not vis_thread.is_alive():
                break

    if args.run_async:
        # Wait until the final inference
        final_ie_output = ie.wait(req_id)
    
        # Finish
        if vis_thread.is_alive():
            q.join()
            q.put(None)
            vis_thread.join()
    
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
        
    if args.run_async:
        q = queue.Queue()
        
    run_example(args)
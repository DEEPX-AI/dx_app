import os
import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine

import torch
import torchvision
from ultralytics.utils import ops

import threading
import queue
import time 
import copy

q = queue.Queue()   # queue of the inference jobs (just runAsync and callback function)
callback_lock = threading.Lock()   # for inference copy lock

callback_cnt = 0
g_shutdown = False

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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


def intersection_filter(x:torch.Tensor):
    for i in range(x.shape[0] - 1):
        for j in range(x.shape[0] - 1):
            a = x[i]
            b = x[j+1]
            if a[5] != b[5]:
                continue
            x1_inter = max(a[0], b[0])
            y1_inter = max(a[1], b[1])
            x2_inter = min(a[2], b[2])
            y2_inter = min(a[3], b[3])
            if b[0] == x1_inter and b[1] == y1_inter and b[2] == x2_inter and b[3] == y2_inter:
                if a[4] > b[4]:
                    x[j+1][4] = 0
                elif a[4] < b[4]:
                    x[i][4] = 0
    return x


def onnx_decode(ie_outputs, cpu_byte):
    import onnxruntime as ort
    sess = ort.InferenceSession(cpu_byte)
    input_names = [input.name for input in sess.get_inputs()]
    input_dict = {}
    
    for i, input_name in enumerate(input_names):
        input_dict.update(input_name, ie_outputs[i])
    
    ort_output = sess.run(None, input_dict)
    return ort_output[0][0]


def all_decode(ie_outputs, layer_config, num_classes):
    ''' slice outputs'''
    outputs = []
    
    outputs.append(ie_outputs[0][...,:255])
    outputs.append(ie_outputs[1][...,:255])
    outputs.append(ie_outputs[2][...,:255])
    
    decoded_tensor = []
    
    for i, output in enumerate(outputs):
        output[...,4:] = sigmoid(output[...,4:]) # obj confidence
        output[...,5:] *= output[...,4:5]
        for l in range(len(layer_config[i]["anchor_width"])):
            layer = layer_config[i]
            stride = layer["stride"]
            grid_size = output.shape[2]
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
            cxcy = output[...,(l*(num_classes + 5))+0:(l*(num_classes + 5))+2]
            wh = output[...,(l*(num_classes + 5))+2:(l*(num_classes + 5))+4]
            cxcy[...,0] = (sigmoid(cxcy[...,0]) * 2 - 0.5 + grid[0]) * stride
            cxcy[...,1] = (sigmoid(cxcy[...,1]) * 2 - 0.5 + grid[1]) * stride
            wh[...,0] = ((sigmoid(wh[...,0]) * 2) ** 2) * layer["anchor_width"][l]
            wh[...,1] = ((sigmoid(wh[...,1]) * 2) ** 2) * layer["anchor_height"][l]
            decoded_tensor.append(output[...,(l*(num_classes + 5))+0:(l*(num_classes + 5))+(num_classes + 5)].reshape(-1, (num_classes + 5)))
            
    decoded_output = np.concatenate(decoded_tensor, axis=0)
    
    return decoded_output


def ppu_decode(ie_outputs, layer_config, num_classes):
    num_det = ie_outputs[0].shape[0]
    ie_output = ie_outputs[0]
    decoded_tensor = []
    for detected_idx in range(num_det):
        tensor = np.zeros((num_classes + 5), dtype=float)
        data = ie_output[detected_idx].tobytes()
        box = np.frombuffer(data[0:16], np.float32)
        gy, gx, anchor, layer = np.frombuffer(data[16:20], np.uint8)
        score = np.frombuffer(data[20:24], np.float32)
        label = np.frombuffer(data[24:28], np.uint32)
        if layer > len(layer_config):
            break
        w = layer_config[layer]["anchor_width"][anchor]
        h = layer_config[layer]["anchor_height"][anchor]
        s = layer_config[layer]["stride"]
        
        grid = np.array([gx, gy], np.float32)
        anchor_wh = np.array([w, h], np.float32)
        xc = (grid - 0.5 + (box[0:2] * 2)) * s
        wh = box[2:4] ** 2 * 4 * anchor_wh
        box = np.concatenate([xc, wh], axis=0)
        tensor[:4] = box
        tensor[4] = score
        tensor[4+1+label] = score
        decoded_tensor.append(tensor)
    if len(decoded_tensor) == 0:
        decoded_tensor = np.zeros((num_classes + 5), dtype=float)
    
    decoded_output = np.stack(decoded_tensor)
    
    return decoded_output


class YoloConfig:
    def __init__(self, model_path, classes, score_threshold, iou_threshold, layers, input_size, output_type, decode_type):
        self.model_path = model_path
        self.classes = classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.layers = layers
        self.input_size = (input_size, input_size)
        self.output_type = output_type
        self.decode_type = decode_type
        self.colors = np.random.randint(0, 256, [len(self.classes), 3], np.uint8).tolist()


class AsyncYolo:
    def __init__(self, ie:InferenceEngine, yolo_config:YoloConfig, classes, score_threshold, iou_threshold, layers, callback_mode):
        self.ie = ie
        self.config = yolo_config
        self.classes = classes
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.layers = layers
        input_resolution = np.sqrt(self.ie.input_size() / 3)
        self.input_size = (input_resolution, input_resolution)
        self.videomode = False
        self.callback_mode = callback_mode
            
        if self.callback_mode:
            self.ie.RegisterCallBack(self.pp_callback)
            
    def run(self, image):
        self.image = copy.deepcopy(image)
        self.input_image, _, _ = letter_box(self.image, self.config.input_size, fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        self.req = self.ie.RunAsync(self.input_image, self)
        if not self.callback_mode :
            self.result_output = self.ie.Wait(self.req)
            q.put(self.req)
        return self.req
    
    def set_videomode(self, video_mode:bool):
        self.videomode = video_mode
    
    def deepcopy(self, outputs):
        self.result_output = copy.deepcopy(outputs)
    
    @staticmethod
    def pp_callback(ie_outputs, user_args):
        with callback_lock:
            value:AsyncYolo = user_args.value
            value.result_output = ie_outputs
            # value.deepcopy(ie_outputs)
            q.put(value.req)
        
class PostProcessingRun:
    def __init__(self, config:YoloConfig, output_task_order):
        self.video_mode = False
        self.config = config
        self.inputsize_w = int(self.config.input_size[0])
        self.inputsize_h = int(self.config.input_size[1])
        self._queue = queue.Queue()
        self.task_order = output_task_order
        self._cpu_model = self.cpu_onnx_extractor()
        
    
    def cpu_onnx_extractor(self):
        try:
            with open(self.config.model_path, 'rb') as f:
                bs = f.read()
                header_dict = json.loads(bs[8:8129].decode().rstrip("\x00"))
                content = bs[8129:]
                data = header_dict["data"]
                offset, size = data["cpu_models"]["cpu_0"]["offset"], data["cpu_models"]["cpu_0"]["size"]
                compile_config = content[offset:offset + size]
                return compile_config
        except:
            print("cpu model handler not found")
            pass
        return 0


    def run(self, result_output):
        self.result_bbox = self.postprocessing(result_output)
        self._queue.put(self.result_bbox)
    
    def replace_layers(self, original_outputs, task_order_list):
        outputs = []
        ''' iformation of slice t'''
        output_lists = [
            "cv2.0.2 : 80, 80, 64 ",
            "cv3.0.2 : 80, 80, 128 -> 80, 80, 80 ",
            "cv2.1.2 : 40, 40, 64 ",
            "cv3.1.2 : 40, 40, 128 -> 40, 40, 80",
            "cv2.2.2 : 20, 20, 64 ",
            "cv3.2.2 : 20, 20, 128 -> 20, 20, 80 ",
                        ]
        order = [1, 0, 3, 2, 5, 4]
        for o in order:
            if o % 2 == 0:
                outputs.append(original_outputs[o][...,:80])
            else:
                outputs.append(original_outputs[o][...,:64])
        return outputs
        
    
    def postprocessing(self, outputs):
        if self.config.output_type == "BBOX":
            decoded_tensor = ppu_decode(outputs, self.config.layers, len(self.config.classes))
        elif len(outputs) > 1:
            if self._cpu_model != 0:
                outputs = self.replace_layers(outputs, self.task_order)
                decoded_tensor = onnx_decode(ie_outputs=outputs, cpu_byte=self._cpu_model)
            else:
                decoded_tensor = all_decode(outputs, self.config.layers, len(self.config.classes))
        else:
            decoded_tensor = outputs[0]
        
        ''' post Processing '''
        if self.config.decode_type in ["yolov8", "yolov9"]:
            x = torch.Tensor(decoded_tensor[0])
            x = x.squeeze(0)
            x = x.T
            # 1, 8400, (num_classes + localization)
            box = ops.xywh2xyxy(x[..., :4])
            conf, j = x[..., 4:].max(-1, keepdim=True)
        else:
            x = torch.Tensor(decoded_tensor)
            x = x[x[...,4] > self.config.score_threshold]
            box = ops.xywh2xyxy(x[:,:4])
            x[:,5:] *= x[:,4:5]
            conf, j = x[:, 5:].max(1, keepdim=True)
        
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > self.config.score_threshold]
        x = x[x[:, 4].argsort(descending=True)]
        # x = intersection_filter(x)
        x = x[torchvision.ops.nms(x[:,:4], x[:, 4], self.config.iou_threshold)]
        x = x[x[:,4] > 0]
        print("[Result] Detected {} Boxes.".format(len(x)))
        return x
        
    def save_result(self, input_image):
        global callback_cnt
        image = input_image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for idx, r in enumerate(self._queue.get().numpy()):
            pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
            if not self.video_mode:
                print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
                      .format(idx, conf, self.config.classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
            image = cv2.rectangle(image, pt1, pt2, self.config.colors[label], 2)
        
        cv2.imwrite(str(callback_cnt)+ "-result.jpg", image)
        callback_cnt += 1
    
    def get_result_frame(self, input_image):
        image = input_image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for idx, r in enumerate(self._queue.get().numpy()):
            pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
            image = cv2.rectangle(image, pt1, pt2, self.config.colors[label], 2)
        self._queue.task_done()
        return image
        

def run_example(config):
    global g_shutdown
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    score_threshold = config["model"]["param"]["score_threshold"]
    iou_threshold = config["model"]["param"]["iou_threshold"]
    layers = config["model"]["param"]["layer"]
    decode_type = config["model"]["param"]["decoding_method"]
    input_list = []
    video_mode = False
    callback_mode = config["callback_mode"]
    print(callback_mode)
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
        if source["type"] == "video" or source["type"] == "camera":
            input_list = [source["path"]]
            video_mode = True
    
    ''' make inference engine (dxrt)'''
    ie = InferenceEngine(model_path)
    task_order = ie.task_order()
    
    yolo_config = YoloConfig(model_path, classes, score_threshold, iou_threshold, layers, np.sqrt(ie.input_size() / 3), ie.output_dtype()[0], decode_type)
    
    async_thread = AsyncYolo(ie, yolo_config, classes, score_threshold, iou_threshold, layers, callback_mode)
    
    pp_thread = PostProcessingRun(yolo_config, task_order)
    
    if video_mode == True:
        async_thread.set_videomode(True)
        cap = cv2.VideoCapture()
        cap.open(input_list[0])
        cap.set(cv2.CAP_PROP_FPS, 30)
        print("video frames length : ", cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while not g_shutdown:
            start_time = time.time()
            ret, image = cap.read()
            if not ret: 
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    print("End of Frames")
                    break
                print("Fail to read frame")
            
            req = async_thread.run(image)
            if q.qsize() > 0:
                pp_thread.run(async_thread.result_output)
                q.get()
                q.task_done()
            
            if pp_thread._queue.qsize() > 0 :
                result_frame = pp_thread.get_result_frame(async_thread.input_image)
                cv2.imshow("test", result_frame)
                end_time = time.time()
                total_time = (end_time - start_time) * 1000.0
                wait_time = 0 if 30 - total_time < 0 else 30 - int(total_time)
                if cv2.waitKey(wait_time + 1) == ord('q'):
                    g_shutdown = True
                    continue
        
        if not q.empty() or not pp_thread._queue.empty():
            while q.qsize() > 0:
                q.get()
                q.task_done()
            while pp_thread._queue.qsize() > 0:
                pp_thread._queue.get()
                pp_thread._queue.task_done()
        q.join()
        pp_thread._queue.join()
    else:
        for input_path in input_list:
            image = cv2.imread(input_path, cv2.IMREAD_COLOR)
            with callback_lock:
                req = async_thread.run(image)
                if q.qsize() > 0:
                    pp_thread.run(async_thread.result_output)
                    q.get()
                    q.task_done()
                
                if pp_thread._queue.qsize() > 0 :
                    pp_thread.save_result(async_thread.input_image)
    
    return 0
                    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./example/yolov8_example.json', type=str, help='yolo object detection json config path')
    parser.add_argument('--callback', action='store_true', dest='callback_mode', help='application runasync type for callback function')
    parser.add_argument('--wait', action='store_false', dest='callback_mode', help='application runasync type for wait function')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()
    
    with open(args.config, "r") as f:
        json_config = json.load(f)
        if args.callback_mode:
            json_config["callback_mode"] = True
        else:
            json_config["callback_mode"] = False
        
    run_example(json_config)
    

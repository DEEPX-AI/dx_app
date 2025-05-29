import os
import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine

import torch
import torchvision
from ultralytics.utils import ops

PPU_TYPES = ["BBOX", "POSE"]

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

def ppu_decode_pose(ie_outputs, layer_config, n_classes):
    ie_output = ie_outputs[0][0]
    num_det = ie_output.shape[0]
    decoded_tensor = []
    for detected_idx in range(num_det):
        tensor = np.zeros((n_classes + 5), dtype=float)
        
        data = ie_output[detected_idx].tobytes()

        box = np.frombuffer(data[0:16], np.float32)
        gy, gx, anchor, layer = np.frombuffer(data[16:20], np.uint8)
        score = np.frombuffer(data[20:24], np.float32)
        label = 0 # np.frombuffer(data[24:28], np.uint32)
        kpts = np.frombuffer(data[28:232], np.float32)

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

        # for i in range(17):
        #     start = (n_classes + 5) + (i * 3) 
        #     tensor[start:start+2] = (grid - 0.5 + (kpts[i*3:i*3+2] * 2)) * s
        #     tensor[start+2:start+3] = kpts[i*3+2:i*3+3]

        decoded_tensor.append(tensor)
    if len(decoded_tensor) == 0:
        decoded_tensor = np.zeros((n_classes + 5), dtype=float)
    
    decoded_output = np.stack(decoded_tensor)
    
    return decoded_output

def ppu_decode(ie_outputs, layer_config, n_classes):
    ie_output = ie_outputs[0][0]
    num_det = ie_output.shape[0]
    decoded_tensor = []
    for detected_idx in range(num_det):
        tensor = np.zeros((n_classes + 5), dtype=float)
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
        decoded_tensor = np.zeros((n_classes + 5), dtype=float)
    
    decoded_output = np.stack(decoded_tensor)
    
    return decoded_output
    

def all_decode(ie_outputs, layer_config, n_classes):
    ''' slice outputs'''
    outputs = []
    outputs.append(ie_outputs[0][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])
    outputs.append(ie_outputs[1][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])
    outputs.append(ie_outputs[2][...,:(n_classes + 5)* len(layer_config[0]["anchor_width"])])
    
    decoded_tensor = []
    
    for i, output in enumerate(outputs):
        for l in range(len(layer_config[i]["anchor_width"])):
            start = l*(n_classes + 5)
            end = start + n_classes + 5
            
            layer = layer_config[i]
            stride = layer["stride"]
            grid_size = output.shape[2]
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
            output[...,start+4:end] = sigmoid(output[...,start+4:end])
            cxcy = output[...,start+0:start+2]
            wh = output[...,start+2:start+4]
            cxcy[...,0] = (sigmoid(cxcy[...,0]) * 2 - 0.5 + grid[0]) * stride
            cxcy[...,1] = (sigmoid(cxcy[...,1]) * 2 - 0.5 + grid[1]) * stride
            wh[...,0] = ((sigmoid(wh[...,0]) * 2) ** 2) * layer["anchor_width"][l]
            wh[...,1] = ((sigmoid(wh[...,1]) * 2) ** 2) * layer["anchor_height"][l]
            decoded_tensor.append(output[...,start+0:end].reshape(-1, n_classes + 5))
            
    decoded_output = np.concatenate(decoded_tensor, axis=0)
    
    return decoded_output


def onnx_decode(ie_outputs, cpu_onnx_path):
    import onnxruntime as ort
    sess = ort.InferenceSession(cpu_onnx_path)
    input_names = [input.name for input in sess.get_inputs()]
    input_dict = {input_names[0]:ie_outputs[0], input_names[1]:ie_outputs[1], input_names[2]:ie_outputs[2]}
    ort_output = sess.run(None, input_dict)
    return ort_output[0][0]


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
    

def run_example(config):
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    n_classes = len(classes)
    score_threshold = config["model"]["param"]["score_threshold"]
    iou_threshold = config["model"]["param"]["iou_threshold"]
    layers = config["model"]["param"]["layer"]
    pp_type = config["model"]["param"]["decoding_method"]
    input_list = []
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
    if len(input_list) == 0 :
        input_list.append("./sample/2.jpg")
    
    ''' make inference engine (dxrt)'''
    ie = InferenceEngine(model_path)
    input_size = np.sqrt(ie.get_input_size() / 3)
    for input_path in input_list:
        image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
        image_input, _, _ = letter_box(image_src, new_shape=(int(input_size), int(input_size)), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        
        ''' detect image (1) run dxrt inference engine, (2) post processing'''
        ie_output = ie.run([image_input])
        print("dxrt inference Done! ")
        decoded_tensor = []
        if ie.get_output_tensors_info()[0]['dtype'] in PPU_TYPES:
            if pp_type in ["yolo_pose"]:
                decoded_tensor = ppu_decode_pose(ie_output, layers, n_classes)
            else:
                decoded_tensor = ppu_decode(ie_output, layers, n_classes)
        elif len(ie_output) > 1:
            cpu_model_path = os.path.join(os.path.split(model_path)[0], "cpu_0.onnx")
            if os.path.exists(cpu_model_path):
                decoded_tensor = onnx_decode(ie_output, cpu_model_path)
            else:
                decoded_tensor = all_decode(ie_output, layers, n_classes)
        elif len(ie_output) == 1:
            decoded_tensor = ie_output[0]
        else:
            raise ValueError(f"[Error] Output Size {len(ie_output)} is not supported !!")
        print("decoding output Done! ")

        ''' post Processing '''
        x = np.squeeze(decoded_tensor)
        x = x[x[..., 4]>score_threshold]
        box = ops.xywh2xyxy(x[..., :4])
        x[:,5:] *= x[:,4:5]
        conf = np.max(x[..., 5:], axis=-1, keepdims=True)
        j = np.argmax(x[..., 5:], axis=-1, keepdims=True)
        mask = conf.flatten() > score_threshold
        filtered = np.concatenate((box, conf, j.astype(np.float32)), axis=1)[mask]
        sorted_indices = np.argsort(-filtered[:, 4])
        x = filtered[sorted_indices]
        x = torch.Tensor(x)
        x = x[torchvision.ops.nms(x[:,:4], x[:, 4], score_threshold)]
        
        ''' save result and print detected info '''
        print("[Result] Detected {} Boxes.".format(len(x)))
        image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        colors = np.random.randint(0, 256, [n_classes, 3], np.uint8).tolist()
        for idx, r in enumerate(x.numpy()):
            pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
            print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
                  .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
            image = cv2.rectangle(image, pt1, pt2, colors[label], 2)
        cv2.imwrite("yolov5s.jpg", image)
        print("save file : yolov5s.jpg ")
        
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./example/run_detector/yolov5s3_example.json', type=str, help='yolo object detection json config path')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()
    
    with open(args.config, "r") as f:
        json_config = json.load(f)
        
    run_example(json_config)

import os
import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine
from dx_engine import Configuration
from packaging import version

import torch
import torchvision
from ultralytics.utils import ops

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


def all_decode(outputs, layer_config, n_classes):
    ''' slice outputs'''
    decoded_tensor = []
    for i, output in enumerate(outputs):
        output = np.squeeze(output)
        for l in range(len(layer_config[i+1]["anchor_width"])):
            start = l*(n_classes + 5)
            end = start + n_classes + 5
            
            layer = layer_config[i+1]
            stride = layer["stride"]
            grid_size = output.shape[2]
            meshgrid_x = np.arange(0, grid_size)
            meshgrid_y = np.arange(0, grid_size)
            grid = np.stack([np.meshgrid(meshgrid_y, meshgrid_x)], axis=-1)[...,0]
            output[start+4:end,...] = sigmoid(output[start+4:end,...])
            cxcy = output[start+0:start+2,...]
            wh = output[start+2:start+4,...]
            cxcy[0,...] = (sigmoid(cxcy[0,...]) * 2 - 0.5 + grid[0]) * stride
            cxcy[1,...] = (sigmoid(cxcy[1,...]) * 2 - 0.5 + grid[1]) * stride
            wh[0,...] = ((sigmoid(wh[0,...]) * 2) ** 2) * layer["anchor_width"][l]
            wh[1,...] = ((sigmoid(wh[1,...]) * 2) ** 2) * layer["anchor_height"][l]
            decoded_tensor.append(output[start+0:end,...].reshape(n_classes + 5, -1))
            
    decoded_output = np.concatenate(decoded_tensor, axis=1)
    decoded_output = decoded_output.transpose(1, 0)
    
    return decoded_output


def transform_box(pt1, pt2, ratio, offset, original_shape):
    dw, dh = offset
    pt1[0] = (pt1[0] - dw) / ratio[0]
    pt1[1] = (pt1[1] - dh) / ratio[1]
    pt2[0] = (pt2[0] - dw) / ratio[0]
    pt2[1] = (pt2[1] - dh) / ratio[1]

    pt1[0] = max(0, min(pt1[0], original_shape[1]))
    pt1[1] = max(0, min(pt1[1], original_shape[0]))
    pt2[0] = max(0, min(pt2[0], original_shape[1]))
    pt2[1] = max(0, min(pt2[1], original_shape[0]))

    return pt1, pt2

def run_example(config):
    if version.parse(Configuration().get_version()) < version.parse("3.0.0"):
        print("DX-RT version 3.0.0 or higher is required. Please update DX-RT to the latest version.")
        exit()
    
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    n_classes = len(classes)
    score_threshold = config["model"]["param"]["score_threshold"]
    layers = config["model"]["param"]["layer"]
    final_output = config["model"]["param"]["final_outputs"][0]
    input_list = []
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
        else:
            raise ValueError(f"[Error] Input Type {source['type']} is not supported !!")
    
    ''' make inference engine (dxrt)'''
    ie = InferenceEngine(model_path)
    if version.parse(ie.get_model_version()) < version.parse('7'):
        print("dxnn files format version 7 or higher is required. Please update/re-export the model.")
        exit()

    tensor_names = ie.get_output_tensor_names()
    layer_idx = []
    for i in range(len(layers)):
        for j in range(len(tensor_names)):
            if layers[i]["name"] == tensor_names[j]:
                layer_idx.append(j)
                break
    if len(layer_idx) == 0:
        raise ValueError(f"[Error] Layer {layers} is not supported !!") 

    input_size = np.sqrt(ie.get_input_size() / 3)
    count = 1
    for input_path in input_list:
        image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
        image_input, ratio, offset = letter_box(image_src, new_shape=(int(input_size), int(input_size)), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        
        ''' detect image (1) run dxrt inference engine, (2) post processing'''
        ie_output = ie.run([image_input])
        print("dxrt inference Done! ")
        decoded_tensor = []
        if not final_output in ie.get_output_tensor_names():
            ie_output = [ie_output[i] for i in layer_idx]
            decoded_tensor = all_decode(ie_output, layers, n_classes)
        else:
            decoded_tensor = ie_output[layer_idx[0]]

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
        # image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        colors = np.random.randint(0, 256, [n_classes, 3], np.uint8).tolist()
        for idx, r in enumerate(x.numpy()):
            pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
            pt1, pt2 = transform_box(pt1, pt2, ratio, offset, image_src.shape)
            print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
                  .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
            image_src = cv2.rectangle(image_src, pt1, pt2, colors[label], 2)
        cv2.imwrite(f"yolov5s_{count}.jpg", image_src)
        print(f"save file : yolov5s_{count}.jpg ")
        count += 1
    

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

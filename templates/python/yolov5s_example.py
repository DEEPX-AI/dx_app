import os
import cv2
import numpy as np
import json
import argparse
import onnxruntime as ort
from dx_engine import InferenceEngine

import torch
import torchvision
from ultralytics.utils import ops

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

def run_example(config):
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    input_list = []
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
    if len(input_list) == 0 :
        input_list.append("./sample/2.jpg")
    ''' define onnx model path '''
    cpu_model_path = os.path.join(os.path.split(model_path)[0], "cpu_0.onnx")
    ''' make inference engine (dxrt & onnx session)'''
    ie = InferenceEngine(model_path)
    sess = ort.InferenceSession(cpu_model_path)
    
    ''' get input nodes name '''
    input_names = [input.name for input in sess.get_inputs()]
    
    for input_path in input_list:
        image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
        image_input, _, _ = letter_box(image_src, new_shape=(512, 512), fill_color=(114, 114, 114), format=cv2.COLOR_BGR2RGB)
        
        ''' detect image (1) run dxrt inference engine, (2) run onnx session for decoding'''
        ie_output = ie.run(image_input)
        print("dxrt inference Done! ")
        input_dict = {input_names[0]:ie_output[0], input_names[1]:ie_output[1], input_names[2]:ie_output[2]}
        ort_output = sess.run(None, input_dict)
        print("cpu node inference Done! ")
        
        ''' post Processing '''
        conf_thres, iou_thres = 0.3, 0.4
        x = torch.Tensor(ort_output[0][0])
        x = x[x[..., 4] > conf_thres]
        box = ops.xywh2xyxy(x[:, :4])
        x[:, 5:] *= x[:, 4:5]
        conf, j = x[:, 5:].max(1, keepdims=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        x = x[x[:, 4].argsort(descending=True)]
        x = x[torchvision.ops.nms(x[:,:4], x[:, 4], iou_threshold=iou_thres)]
        
        print("[Result] Detected {} Boxes.".format(len(x)))
        ''' result view '''
        image = cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        colors = np.random.randint(0, 256, [80, 3], np.uint8).tolist()
        for idx, r in enumerate(x.numpy()):
            pt1, pt2, conf, label = r[0:2].astype(int), r[2:4].astype(int), r[4], r[5].astype(int)
            print("[{}] conf, classID, x1, y1, x2, y2, : {:.4f}, {}({}), {}, {}, {}, {}"
                  .format(idx, conf, classes[label], label, pt1[0], pt1[1], pt2[0], pt2[1]))
            image = cv2.rectangle(image, pt1, pt2, colors[label], 2)
        cv2.imwrite("yolov5s.jpg", image)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./example/yolov5s3_example.json', type=str, help='yolo object detection json config path')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()
    
    with open(args.config, "r") as f:
        json_config = json.load(f)
        
    run_example(json_config)

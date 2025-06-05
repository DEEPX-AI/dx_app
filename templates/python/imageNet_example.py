import os
import cv2
import numpy as np
import json
import argparse
from dx_engine import InferenceEngine

def preprocessing(image, new_shape=(224, 224), align=64, format=None):
    image = cv2.resize(image, new_shape)
    h, w, c = image.shape
    if format is not None:
        image = cv2.cvtColor(image, format)
    if align == 0 :
        return image
    length = w * c
    align_factor = align - (length - (length & (-align)))
    image = np.reshape(image, (h, w * c))
    dummy = np.full([h, align_factor], 0, dtype=np.uint8)
    image_input = np.concatenate([image, dummy], axis=-1)
        
    return image_input

def postprocessing(outputs, n_classes):
    outputs = outputs[...,:n_classes]
    exp_result = np.exp(outputs - np.max(outputs))
    exp_result = exp_result / exp_result.sum()
    top1 = np.argmax(exp_result)
    return top1

def run_example(config):
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    input_list = []
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
    if len(input_list) == 0:
        input_list.append("sample/ILSVRC2012/0.jpeg")
    
    ie = InferenceEngine(model_path)

    image_input_list = []
    output_buffer_list = []
    
    output_data_type = ie.get_output_tensors_info()[0]["dtype"]
    output_size = ie.get_output_size()/np.dtype(output_data_type).itemsize

    for input_path in input_list:
        image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if ie.get_input_size() == 224 * 224 * 3:
            align = 0
        else:
            align = 64
        image_input = preprocessing(image_src, new_shape=(224, 224), align=align, format=cv2.COLOR_BGR2RGB)
        image_input_list.append([image_input])
        output_buffer = np.zeros((int(output_size)), dtype=output_data_type)
        output_buffer_list.append([output_buffer])
    
    # This example demonstrates batch inference using run_batch,
    # which processes multiple input images together for efficient inference

    result = ie.run(image_input_list, output_buffer_list)

    for ie_output in output_buffer_list:
        if(ie_output[0].shape[0] > 1):
            output = postprocessing(ie_output[0], len(classes))
        else:
            output = ie_output[0][0]
        print("[{}] Top1 Result : class {} ({})".format(input_path, output, classes[output]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./example/run_classifer/imagenet_example.json', type=str, help='yolo object detection json config path')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()
    
    with open(args.config, "r") as f:
        json_config = json.load(f)
        
    run_example(json_config)

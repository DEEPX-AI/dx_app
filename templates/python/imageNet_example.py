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


def run_example(config):
    model_path = config["model"]["path"]
    classes = config["output"]["classes"]
    input_list = []
    
    for source in config["input"]["sources"]:
        if source["type"] == "image":
            input_list.append(source["path"])
    if len(input_list) == 0:
        input_list.append("example/ILSVRC2012/0.jpeg")
    
    ie = InferenceEngine(model_path)
    
    for input_path in input_list:
        image_src = cv2.imread(input_path, cv2.IMREAD_COLOR)
        image_input = preprocessing(image_src, new_shape=(224, 224), align=64, format=cv2.COLOR_BGR2RGB)
        ie_output = ie.run(image_input)
        
        print("[{}] Top1 Result : class {} ({})".format(input_path, ie_output[0][0], classes[ie_output[0][0]]))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./example/imagenet_example.json', type=str, help='yolo object detection json config path')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        parser.print_help()
        exit()
    
    with open(args.config, "r") as f:
        json_config = json.load(f)
        
    run_example(json_config)

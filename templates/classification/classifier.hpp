#pragma once
#include <future>
#include <thread>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "post_process/classification_post_processing.hpp"
#include "pre_process/classification_pre_processing.hpp"

#include "dxapp_api.hpp"
#include "app_parser.hpp"

class Classifier
{
public:

    Classifier(dxapp::AppConfig &_config):config(_config)
    {
        bool is_valid = dxapp::validationJsonSchema(config.modelInfo.c_str(), modelInfoSchema);
        if(!is_valid)
        {
            std::cout << config.modelInfo << std::endl;
            std::cout << "model params is invalid parsing" << std::endl;
            std::terminate();
        }
        rapidjson::Document doc;
        doc.Parse(config.modelInfo.c_str());
        std::string modelPath = doc["path"].GetString();
        inferenceEngine = std::make_shared<dxrt::InferenceEngine>(modelPath);
        
        inputShape = inferenceEngine->inputs().front().shape();
        outputShape = inferenceEngine->outputs().front().shape();
        inputSize = inferenceEngine->input_size();
        if(inputShape[0]*inputShape[1]*inputShape[2]*inputShape[3] != inputSize)
            alignFactor = dxapp::common::get_align_factor(inputShape[1] * inputShape[3], 64);
        else
            alignFactor = false;
                
        preConfig = {
            ._dstShape = inputShape,
            ._inputFormat = config.inputFormat,
            ._alignFactor = alignFactor,
            ._needIm2Col = false,
        };
        postConfig = {
            ._outputShape = outputShape,
            ._outputType = config.outputType,
            ._classes = config.classes,
        };

        if(outputShape.size() == 0){
            is_argmax = true;
        }
    };
    ~Classifier(){};
    
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    int inputSize;
    int alignFactor;
    bool is_argmax = false;
    
    std::shared_ptr<dxrt::InferenceEngine> inferenceEngine;
    dxapp::classification::PreConfig preConfig;
    dxapp::classification::PostConfig postConfig;
    dxapp::AppConfig &config;

private:
    const char* modelInfoSchema = R"""(
            {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": [
                    "path"
                ]
            }
        )""";

};
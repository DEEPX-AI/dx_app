#pragma once
#include <future>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include "post_process/classification_post_processing.hpp"
#include "pre_process/classification_pre_processing.hpp"

#include "app_parser.hpp"

class Classifier
{
public:

    Classifier(const dxapp::AppConfig &_config):config(_config)
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
        
        inputShape = std::vector<int64_t>(inferenceEngine->inputs().front().shape());
        outputShape = inferenceEngine->outputs().front().shape();
        inputSize = inferenceEngine->input_size();
        auto factorialShape = inputShape[1]*inputShape[1]*3;
        if(!dxapp::common::checkOrtLinking() && static_cast<uint64_t>(factorialShape) != inputSize)
            alignFactor = dxapp::common::get_align_factor(inputShape[1] * 3, 64);
        else
            alignFactor = false;

        if(outputShape.size() == 1){
            is_argmax = true;
        }

        preConfig._dstShape = inputShape;
        preConfig._dstSize = inputSize;
        preConfig._inputFormat = config.inputFormat;
        preConfig._alignFactor = alignFactor;

        postConfig._outputShape = outputShape;
        postConfig._outputType = is_argmax? OUTPUT_ARGMAX : OUTPUT_NONE_ARGMAX;
        postConfig._classes = config.classes;
    };
    ~Classifier()=default;
    
    dxapp::AppConfig config;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    uint64_t inputSize;
    int alignFactor;
    bool is_argmax = false;
    
    std::shared_ptr<dxrt::InferenceEngine> inferenceEngine;
    dxapp::classification::PreConfig preConfig;
    dxapp::classification::PostConfig postConfig;

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
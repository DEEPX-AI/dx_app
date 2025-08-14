#pragma once
#include <future>
#include <thread>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#include <post_process/classification_post_processing.hpp>
#include <pre_process/classification_pre_processing.hpp>

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
        inputWidth = doc["input_width"].GetInt();
        inputHeight = doc["input_height"].GetInt();
        for(auto& output : doc["final_outputs"].GetArray())
        {
            finalOutputs.push_back(output.GetString());
        }
        inferenceEngine = std::make_shared<dxrt::InferenceEngine>(modelPath);
        if(!dxapp::common::minversionforRTandCompiler(inferenceEngine.get()))
        {
            std::cerr << "[DXAPP] [ER] The version of the compiled model is not compatible with the version of the runtime. Please compile the model again." << std::endl;
            std::terminate();
        }
        
        inputShape = std::vector<int64_t>(inferenceEngine->GetInputs().front().shape());
        outputShape = inferenceEngine->GetOutputs().front().shape();
        inputSize = inferenceEngine->GetInputSize();

        if(outputShape.size() == 1){
            is_argmax = true;
        }

        preConfig._dstShape = inputShape;
        preConfig._dstSize = inputSize;
        preConfig._inputFormat = config.inputFormat;

        postConfig._outputShape = outputShape;
        postConfig._outputType = is_argmax? OUTPUT_ARGMAX : OUTPUT_NONE_ARGMAX;
        postConfig._classes = config.classes;
    };
    ~Classifier()=default;
    
    dxapp::AppConfig config;
    int inputWidth;
    int inputHeight;
    std::vector<std::string> finalOutputs;
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
                    },
                    "input_width": {
                        "type": "number"
                    },
                    "input_height": {
                        "type": "number"
                    },
                    "final_outputs": {
                        "type": "array",
                        "items": "string"
                    }
                },
                "required": [
                    "path", "input_width", "input_height", "final_outputs"
                ]
            }
        )""";

};
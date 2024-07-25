#pragma once
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <map>

#include "rapidjson/error/en.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"

#include "dxapp_api.hpp"

namespace dxapp
{
class AppConfig
{
public:

    AppConfig(const std::string &json_source)
    {
        std::string json_path = "";
        std::string json = "";
        if(dxapp::common::pathValidation(json_source))
        {
            json_path = json_source;
            std::ifstream ifs(json_path);
            if(!ifs.is_open())
            {
                std::cout << "can't open " << json_path << std::endl;
                std::terminate();
            }
            json = std::string((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
        }
        else
        {
            json = json_source;
        }
        bool is_valid = dxapp::validationJsonSchema(json.c_str(), application_json_schema);
        if(!is_valid)
        {
            std::cout << json_path << " file is not a valid." << std::endl;
            std::terminate();
        }
        rapidjson::Document doc;
        doc.Parse(json.c_str());
        std::string read = "";
        
        modelInfo = dxapp::JsonToString(doc["model"]);
        
        read = doc["input"]["format"].GetString();
        if(read == "BIN")
            inputFormat = AppInputFormat::INPUT_BINARY;
        else if(read=="BGR")
            inputFormat = AppInputFormat::IMAGE_BGR;
        else if(read=="RGB")
            inputFormat = AppInputFormat::IMAGE_RGB;
        
        for (auto &d:doc["input"]["sources"].GetArray())
        {
            AppSourceInfo source;
            source.inputPath = d.GetObject()["path"].GetString();
            if(d.GetObject().HasMember("frames"))
            {
                source.numOfFrames = d.GetObject()["frames"].GetInt();
            }
            else
            {
                source.numOfFrames = -1;
            }
            read = d.GetObject()["type"].GetString();
            if(read=="image")
                source.inputType = AppInputType::IMAGE;
            else if(read=="bin")
                source.inputType = AppInputType::BINARY;
            else if(read=="CSV")
                source.inputType = AppInputType::CSV;
            else if(read=="video")
                source.inputType = AppInputType::VIDEO;
            else if(read=="camera")
                source.inputType = AppInputType::CAMERA;
            else if(read=="isp")
                source.inputType = AppInputType::ISP;
            
            sourcesInfo.emplace_back(source);
        }

        read = doc["usage"].GetString();
        if(read=="classification")
            appUsage = AppUsage::CLASSIFICATION;
        else if(read=="detection")
            appUsage = AppUsage::DETECTION;
        else if(read=="faceid")
            appUsage = AppUsage::FACEID;
        else if(read=="segmentation")
            appUsage = AppUsage::SEGMENTATION;
        else if(read=="yolo_pose")
            appUsage = AppUsage::YOLOPOSE;
        else
            appUsage = AppUsage::DETECTION;
        
        uint16_t i = 0;
        for(auto &d:doc["output"]["classes"].GetArray()){
            classes.emplace(i, d.GetString());
            i++;
        }
        numOfClasses = i;

        read = doc["application"]["type"].GetString();
        if(read=="log" || read=="none")
            appType = AppType::NONE;
        else if(read=="save")
            appType = AppType::OFFLINE;
        else if(read=="realtime")
            appType = AppType::REALTIME;
        if(doc["application"].HasMember("resolution"))
        {
            videoOutResolution._width = doc["application"]["resolution"].GetArray()[0].GetInt();
            videoOutResolution._height = doc["application"]["resolution"].GetArray()[1].GetInt();
        }  
        else
        {
            videoOutResolution._width = 0;
            videoOutResolution._height = 0;
        }
        outputType = AppOutputType::OUTPUT_NONE_ARGMAX;
    };
    ~AppConfig()=default;
    
    std::string modelInfo;

    std::vector<AppSourceInfo> sourcesInfo;
    
    AppUsage appUsage;
    AppType appType;
    AppInputFormat inputFormat;
    AppOutputType outputType;
    std::map<uint16_t, std::string> classes;
    int numOfClasses;
    dxapp::common::Size videoOutResolution;
    
private:

    const char *application_json_schema = R"""(
        {
            "type": "object",
            "properties": {
                "usage": {
                    "type": "string"
                },
                "model": {
                    "type": "object"
                },
                "input": {
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string"
                        },
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties":{
                                    "type": {
                                        "type": "string"
                                    },
                                    "path":{
                                        "type": "string"
                                    },
                                    "frames":{
                                        "type": "number"
                                    }
                                },
                                "required": [
                                    "type", "path"
                                ]
                            }
                        }
                    },
                    "required": [
                        "format", "sources"
                    ]
                },
                "output": {
                    "type": "object",
                    "properties": {
                        "classes": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "type": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "classes"
                    ]
                },
                "application": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string"
                        },
                        "resolution": {
                            "type": "array",
                            "item": "number",
                            "minItems": 2
                        }
                    },
                    "required": [
                        "type"
                    ]
                }
            },
            "required": [
                "usage", "model", "input", "output", "application"
            ]
        }
        )""";

};
} // namespace dxapp


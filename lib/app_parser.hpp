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

    AppConfig(std::string json_path)
    {
        std::ifstream ifs(json_path);
        if(!ifs.is_open())
        {
            std::cout << "can't open " << json_path << std::endl;
            std::terminate();
        }
        std::string json((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
        bool is_valid = dxapp::validationJsonSchema(json.c_str(), application_json_schema);
        if(!is_valid)
        {
            std::cout << json_path << " file is not a valid." << std::endl;
            std::terminate();
        }

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
        else if(read=="yolopose")
            appUsage = AppUsage::YOLOPOSE;
        
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

        // TODO : get below informations automatically
        need_im2col = doc["input"]["need_im2col"].GetBool();
        read = doc["output"]["type"].GetString();
        if(read=="argmax")
            outputType = AppOutputType::OUTPUT_ARGMAX;
        else if(read=="raw")
            outputType = AppOutputType::OUTPUT_NONE_ARGMAX;


    };
    ~AppConfig(){};
    
    std::string modelInfo;

    std::vector<AppSourceInfo> sourcesInfo;
    
    AppUsage appUsage;
    AppType appType;
    AppInputFormat inputFormat;
    AppOutputType outputType;
    std::map<uint16_t, std::string> classes;
    int numOfClasses;
    dxapp::common::Size videoOutResolution;

    // TODO : get below informations automatically
    bool need_align;
    bool need_im2col;
    
private:
    rapidjson::Document doc;

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
                        "classes", "type"
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


#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "dxrt/dxrt_api.h"
#include "common/objects.hpp"
#include "utils/box_decode.hpp"
#include "utils/nms.hpp"

namespace dxapp
{
namespace yolo
{
    enum Activate
    {
        NONE=0,
        SIGMOID,
        EXP,
    };
    enum Decode
    {
        YOLO_BASIC=0,
        YOLOX,
        YOLOSCALE,
        YOLO_POSE,
        SCRFD,
        YOLOV8,
        YOLOV9,
        CUSTOM_DECODE,
    };
    enum PPUFormat
    {
        NONEPPU=0,
        BBOX=32,
        FACE=64,
        POSE=256
    };
    enum BBoxFormat
    {
        CXCYWH=0,
        XYX2Y2,
    };
    enum KeyPointOrder
    {
        KPT_FRONT=0,
        BBOX_FRONT,
    };
    struct Layers
    {
        std::string name;
        int stride;
        float scale = 0;
        std::vector<int> anchor_width;
        std::vector<int> anchor_height;
    };
    struct Params 
    {
        std::vector<int64_t> _input_shape;
        std::vector<std::vector<int64_t>> _outputShape;
        float _score_threshold;
        float _iou_threshold;
        std::function<float(float)> _last_activation;
        Decode _decode_method;
        PPUFormat _ppu_format;
        BBoxFormat _box_format;
        KeyPointOrder _kpt_order;
        int _kpt_count;
        std::vector<Layers> _layers;
        std::map<uint16_t, std::string> _classes;
        int _numOfClasses;
    };

    float exp(float x)
    {
        return std::exp(x);
    };
    float sigmoid(float x)
    {
        return 1 / (1 + std::exp(-x));
    };
    float inversSigmoid(float x)
    {
        return std::log(x / (1.0f - x));
    };

    class PostProcessing
    {
    private:
        dxapp::yolo::Params _params;
        
        dxapp::common::Size _inputSize;
        dxapp::common::Size _srcSize;
        dxapp::common::Size _dstSize;
        
        dxapp::common::DetectObject _result;
        std::vector<std::vector<std::pair<float, int>>> _scoreIndices;
        std::vector<dxapp::common::BBox> _rawBoxes;

        dxapp::common::Size_f _postprocScaleRatio;
        dxapp::common::Size_f _postprocPaddedSize;

        int _yoloxLocationIdx = 0;
        int _yoloxBoxScoreIdx = 0;
        int _yoloxClassScoreIdx = 0;
        
        int _scrfdClassScoreIdx = 0;
        int _scrfdLocationIdx = 0;
        int _scrfdKptIdx = 0;

        std::vector<float*> _rawVector;

        std::function<dxapp::common::BBox(std::function<float(float)>, std::vector<float*>, dxapp::common::Point, dxapp::common::Size, int, float)> _decode;

    public:
        PostProcessing(dxapp::yolo::Params& params, dxapp::common::Size inputSize, dxapp::common::Size srcSize, dxapp::common::Size dstSize):_params(params),_inputSize(inputSize),_srcSize(srcSize),_dstSize(dstSize)
        {
            for(int i=0;i<_params._numOfClasses;i++){
                std::vector<std::pair<float, int>> v;
                _scoreIndices.emplace_back(v);
            }
            dxapp::common::Size_f _postprocRatio;
            _postprocRatio._width = (float)_dstSize._width/srcSize._width;
            _postprocRatio._height = (float)_dstSize._height/srcSize._height;

            float _preprocRatio;            
            if(_srcSize == _inputSize)
            {
                _postprocPaddedSize._width = 0.f;
                _postprocPaddedSize._height = 0.f;
                _postprocScaleRatio = dxapp::common::Size_f(_postprocRatio._width, _postprocRatio._height);                
            }
            else
            {
                _preprocRatio = std::min((float)_inputSize._width/_srcSize._width, (float)_inputSize._height/_srcSize._height);
                dxapp::common::Size _resizeSize((int)(_srcSize._width * _preprocRatio), (int)(_srcSize._height * _preprocRatio));
                _postprocPaddedSize._width = (_inputSize._width - _resizeSize._width) / 2.f;
                _postprocPaddedSize._height = (_inputSize._height - _resizeSize._height) / 2.f;
                _postprocScaleRatio = dxapp::common::Size_f(_postprocRatio._width/_preprocRatio, _postprocRatio._height/_preprocRatio);
            }

            switch (_params._decode_method)
            {
            case Decode::YOLO_BASIC:
                _decode = dxapp::decode::yoloBasicDecode;
                break;
            case Decode::YOLOX:
                _decode = dxapp::decode::yoloXDecode;
                _yoloxLocationIdx = 0;
                _yoloxBoxScoreIdx = 1;
                _yoloxClassScoreIdx = 2;
                break;
            case Decode::YOLOSCALE:
                _decode = dxapp::decode::yoloScaledDecode;
                break;    
            case Decode::CUSTOM_DECODE:
                _decode = dxapp::decode::yoloCustomDecode;
                break;
            case Decode::YOLO_POSE:
                _decode = dxapp::decode::yoloPoseDecode;
                break;
            case Decode::SCRFD:
                _decode = dxapp::decode::SCRFDDecode;
                _scrfdClassScoreIdx = 0;
                _scrfdLocationIdx = 1;
                _scrfdKptIdx = 2;
                break;
            case Decode::YOLOV8:
                _decode = dxapp::decode::yoloCustomDecode;
                break;
            default:
                _decode = dxapp::decode::yoloBasicDecode;
                break;
            }
        };
        PostProcessing(){};
        ~PostProcessing(){};
        dxapp::common::DetectObject getResult(){return _result;};
        void run(uint8_t* output_data, int64_t data_length)
        // void run(std::vector<std::shared_ptr<dxrt::Tensor>> outputs)
        {
            for(auto &indices:_scoreIndices){
                indices.clear();
            }
            _rawBoxes.clear();
            _result._detections.clear();
            _result._num_of_detections = 0;

            if(_params._ppu_format > 0) // shape : [124, ], numBoxes = 124
                getBoxesFromPPUOutputs(output_data, data_length, _params._ppu_format);
            else if(_params._outputShape.size() == 1) // ONNX outputs shape : [1, 16384, 85], numBoxes = 16384
                getBoxesFromONNXOutputs(output_data, data_length); 
            else if(_params._decode_method==Decode::YOLOX) // anchor free detector
                getBoxesFromYoloXFormat(output_data, _yoloxLocationIdx, _yoloxBoxScoreIdx, _yoloxClassScoreIdx);
            else if(_params._decode_method==Decode::YOLO_POSE) // yolo pose 
                getBoxesFromYoloPoseFormat(output_data, _params._kpt_order, _params._kpt_count);
            else if(_params._decode_method==Decode::SCRFD) // face detection
                getBoxesFromSCRFDFormat(output_data, _scrfdClassScoreIdx, _scrfdLocationIdx, _scrfdKptIdx, _params._kpt_count);
            else if(_params._decode_method==Decode::YOLOV8)
                getBoxesFromYoloV8Format(output_data);
            else if(_params._decode_method==Decode::YOLOV9)
                getBoxesFromONNXOutputs(output_data, data_length);
            else if(_params._decode_method==Decode::CUSTOM_DECODE)
                getBoxesFromCustomPostProcessing(output_data);
            else
                getBoxesFromYoloFormat(output_data);
            
            dxapp::common::nms(_rawBoxes, _scoreIndices, _params._iou_threshold, _params._classes, _postprocPaddedSize, _postprocScaleRatio, _result);
        };

        static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
        {
            if(a.first > b.first)
                return true;
            else
                return false;
        };

        void getBoxesFromPPUOutputs(uint8_t* outputs, int64_t numDetected, dxapp::yolo::PPUFormat ppuFormat)
        {
            int boxIdx = 0;
            /**
             * FORMAT : BBOX = 32  (512 limit)
             * FORMAT : FACE = 64  (256 limit)
             * FORMAT : POSE = 256 (64 limit)
            */
            size_t ppuDataSize = (size_t)ppuFormat;
            for(int64_t i=0;i<numDetected;i++)
            {
                uint8_t *raw_data = outputs + (i * ppuDataSize);
                dxrt::DeviceBoundingBox_t *data = static_cast<dxrt::DeviceBoundingBox_t*>((void*)raw_data);
                auto layer = _params._layers[data->layer_idx];
                int stride = layer.stride;
                int numGridX = data->grid_x;
                int numGridY = data->grid_y;
                
                if(data->score < _params._score_threshold)
                    continue;
                
                if(ppuFormat > dxapp::yolo::PPUFormat::BBOX)
                    _scoreIndices[0].emplace_back(data->score, boxIdx);
                else
                    _scoreIndices[data->label].emplace_back(data->score, boxIdx);

                dxapp::common::BBox temp;
                switch (_params._decode_method)
                {
                case dxapp::yolo::YOLO_BASIC:
                case dxapp::yolo::YOLO_POSE:
                    temp = {
                        (data->x * 2 - 0.5f + numGridX) * stride,
                        (data->y * 2 - 0.5f + numGridY) * stride,
                        0,
                        0,
                        (data->w * data->w * 4) * layer.anchor_width[data->box_idx],
                        (data->h * data->h * 4) * layer.anchor_height[data->box_idx],
                        {dxapp::common::Point_f(-1, -1, -1)}
                    };
                    break;
                case dxapp::yolo::YOLOSCALE:
                    temp = {
                        (data->x * layer.scale - 0.5f * (layer.scale - 1) + numGridX) * stride,
                        (data->y * layer.scale - 0.5f * (layer.scale - 1) + numGridY) * stride,
                        0,
                        0,
                        (data->w * data->w * 4) * layer.anchor_width[data->box_idx],
                        (data->h * data->h * 4) * layer.anchor_height[data->box_idx],
                        {dxapp::common::Point_f(-1, -1, -1)}
                    };
                    break;
                case dxapp::yolo::YOLOX:
                    temp = {
                        (numGridX + data->x) * stride,
                        (numGridY + data->y) * stride,
                        0,
                        0,
                        exp(data->w) * stride,
                        exp(data->h) * stride,
                        {dxapp::common::Point_f(-1, -1, -1)}
                    };
                    break;
                case dxapp::yolo::SCRFD:
                    temp = {
                        (numGridX - data->x) * stride,
                        (numGridY - data->y) * stride,
                        (numGridX + data->w) * stride,
                        (numGridY + data->h) * stride,
                        2* data->w * stride,
                        2* data->h * stride,
                        {dxapp::common::Point_f(-1, -1, -1)}
                    };
                    break;
                case dxapp::yolo::CUSTOM_DECODE:
                    break;
                };
                if(_params._box_format == dxapp::yolo::BBoxFormat::CXCYWH)
                {
                    temp._xmin = temp._xmin - (temp._width/2);
                    temp._ymin = temp._ymin - (temp._height/2);
                    temp._xmax = temp._xmin + (temp._width/2);
                    temp._ymax = temp._ymin + (temp._height/2);
                }
                else
                {
                    temp._width = temp._xmax - temp._xmin;
                    temp._height = temp._ymax - temp._ymin;
                }
                boxIdx++;
                if(ppuFormat == dxapp::yolo::PPUFormat::POSE)
                {
                    temp._kpts.clear();
                    for(int idx = 0; idx < _params._kpt_count; idx++)
                    {
                        dxrt::DevicePose_t *kpt_data = static_cast<dxrt::DevicePose_t*>((void*)data);
                        if((kpt_data->kpts[idx][2])<0.5)
                        {
                            temp._kpts.emplace_back(dxapp::common::Point_f(-1, -1));
                        }
                        else
                        {
                            temp._kpts.emplace_back(dxapp::common::Point_f(
                                            (kpt_data->kpts[idx][0] * 2 - 0.5 + numGridX) * stride,
                                            (kpt_data->kpts[idx][1] * 2 - 0.5 + numGridY) * stride,
                                            0.5f
                                            ));
                        }
                    }
                }
                else if(ppuFormat == dxapp::yolo::PPUFormat::FACE)
                {
                    temp._kpts.clear();
                    for(int idx = 0; idx < _params._kpt_count; idx++)
                    {
                        dxrt::DeviceFace_t *kpt_data = static_cast<dxrt::DeviceFace_t*>((void*)data);
                        temp._kpts.emplace_back(dxapp::common::Point_f(
                                        (numGridX + kpt_data->kpts[idx][0]) * stride,
                                        (numGridY + kpt_data->kpts[idx][1]) * stride,
                                        0.5f
                                        ));
                    }
                }
                _rawBoxes.emplace_back(temp);
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };

        void getBoxesFromONNXOutputs(uint8_t* outputs, int64_t numDetected)
        {
            int boxIdx = 0;
            float score, score1;
            auto* raw_data = (float*)outputs;
            if (_params._decode_method == dxapp::yolo::Decode::YOLOV8 ||
                _params._decode_method == dxapp::yolo::Decode::YOLOV9)
            {
                /**
                 * @note Ultralytics models, which make yolov8/v5/v7 has same post processing methods
                 *      https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-CPP/inference.cpp
                 */
                auto strideNum = _params._outputShape.front()[2]; // 8400
                auto signalResultNum = _params._outputShape.front()[1]; // 84
                cv::Mat rawData = cv::Mat(signalResultNum, strideNum, CV_32F, raw_data);
                rawData = rawData.t();
                for(int64_t i=0;i<strideNum;++i)
                {
                    float *data = (float*)rawData.data + (signalResultNum * i);
                    float* classesScores = data + 4;
                    cv::Mat scores(1, _params._numOfClasses, CV_32FC1, classesScores);
                    cv::Point class_id;
                    double maxClassScore;
                    cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
                    if(maxClassScore > _params._score_threshold)
                    {
                        _scoreIndices[class_id.x].emplace_back((float)maxClassScore, boxIdx);
                    }
                    dxapp::common::BBox temp {
                                data[0] - data[2]/2.f,
                                data[1] - data[3]/2.f,
                                data[0] + data[2]/2.f,
                                data[1] + data[3]/2.f,
                                data[2],
                                data[3],
                                {dxapp::common::Point_f(-1, -1, -1)}};
                    _rawBoxes.emplace_back(temp);
                    boxIdx++;
                }
            }
            else
            {
                for(int64_t i=0;i<_params._outputShape.front()[1];i++)
                {
                    float* data = raw_data + (4 + 1 + _params._numOfClasses + (_params._kpt_count * 3)) * i;
                    score1 = data[4];
                    if(score1 > _params._score_threshold)
                    {
                        for(int cls=0;cls<_params._numOfClasses;cls++)
                        {
                            bool boxDecoded = false;
                            score = score1 * data[5+cls];
                            if(score > _params._score_threshold)
                            {
                                _scoreIndices[cls].emplace_back(score, boxIdx);
                                if(!boxDecoded)
                                {
                                    dxapp::common::BBox temp {
                                                data[0] - data[2]/2.f,
                                                data[1] - data[3]/2.f,
                                                data[0] + data[2]/2.f,
                                                data[1] + data[3]/2.f,
                                                data[2],
                                                data[3],
                                                {dxapp::common::Point_f(-1, -1, -1)}
                                    };
                                    if(_params._decode_method == dxapp::yolo::Decode::YOLO_POSE)
                                    {
                                        temp._kpts.clear();
                                        for(int idx = 0; idx < _params._kpt_count; idx++)
                                        {
                                            int kptIdx = (idx * 3) + (4 + 1 + _params._numOfClasses);
                                            
                                            if((data[kptIdx + 2])<0.5)
                                            {
                                                temp._kpts.emplace_back(dxapp::common::Point_f(-1, -1));
                                            }
                                            else
                                            {
                                                temp._kpts.emplace_back(dxapp::common::Point_f(
                                                                data[kptIdx + 0],
                                                                data[kptIdx + 1],
                                                                0.5f
                                                                ));
                                            }
                                        }
                                        
                                    }
                                    _rawBoxes.emplace_back(temp);
                                    boxDecoded = true;
                                    boxIdx++;
                                }
                            }
                        }
                    }
                }
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };
        
        void getBoxesFromYoloPoseFormat(uint8_t* outputs, KeyPointOrder kpt_order, int kpt_count)
        {
            int boxIdx = 0;
            float rawThreshold = inversSigmoid(_params._score_threshold);
            float score, score1;
            int first = -2, second = -1;
            int first_block = 0, second_block = 0;
            int first_layer_pitch = 1, second_layer_pitch = 1;
            float* raw_data = (float*)outputs;
            if(kpt_order == KeyPointOrder::KPT_FRONT)
            {
                first = -1, second = -2;
            }

            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                first += 2;
                second += 2;
                int first_pitch = _params._outputShape[first].back();
                int second_pitch = _params._outputShape[second].back();
                for(const auto &s: _params._outputShape[first])
                    first_layer_pitch *= s;
                for(const auto &s: _params._outputShape[second])
                    second_layer_pitch *= s;
                if(kpt_order == KeyPointOrder::KPT_FRONT)
                    first_block = second_block + second_layer_pitch;
                else
                    second_block = first_block + first_layer_pitch;

                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int box=0; box<(int)layer.anchor_width.size(); box++) // num : 3
                        { 
                            bool boxDecoded = false;  
                            float *rawBuffer1, *rawBuffer2, *rawBuffer3;
                            if (box == 0)
                            {   
                                rawBuffer1 = raw_data + first_layer_pitch + (gY * numGridX * first_pitch) 
                                                          + (gX * first_pitch)
                                                          + 0;
                                rawBuffer2 = raw_data + first_layer_pitch + (gY * numGridX * first_pitch) 
                                                          + (gX * first_pitch)
                                                          + 6;
                                rawBuffer3 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + 0;
                            }
                            else if (box == 1)
                            {
                                rawBuffer1 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (13 * 3);
                                rawBuffer2 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (13 * 3) + 6;
                                rawBuffer3 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (17 * 3) + 6;
                            }
                            else if (box == 2)
                            {
                                rawBuffer1 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (30 * 3) + 6;
                                rawBuffer2 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (30 * 3) + 12;
                                rawBuffer3 = raw_data + second_layer_pitch + (gY * numGridX * second_pitch) 
                                                          + (gX * second_pitch)
                                                          + (34 * 3) + 12;
                            }

                            if(rawBuffer1[4]>rawThreshold)
                            {
                                score1 = _params._last_activation(rawBuffer1[4]);
                                if(score1 > _params._score_threshold)
                                {
                                    for(int cls=0; cls<_params._numOfClasses;cls++)
                                    {
                                        score = score1 * _params._last_activation(rawBuffer1[5+cls]); 
                                        if (score > _params._score_threshold)
                                        {
                                            _scoreIndices[cls].emplace_back(score, boxIdx);
                                            if(!boxDecoded)
                                            {
                                                _rawVector.clear();
                                                _rawVector.emplace_back(rawBuffer1);
                                                _rawVector.emplace_back(rawBuffer2);
                                                _rawVector.emplace_back(rawBuffer3);
                                                dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), dxapp::common::Size(layer.anchor_width[box], layer.anchor_height[box]), stride, (float)kpt_count);
                                                _rawBoxes.emplace_back(temp);
                                                boxDecoded = true;
                                                boxIdx++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                first_block += second_block + second_layer_pitch;
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };

        void getBoxesFromSCRFDFormat(uint8_t* outputs,  int classScoreIdx, int locationIdx, int kptIdx, int kpt_count)
        {
            int boxIdx = 0;
            float score, score1;
            float *data, *kpt;
            /***
             * The output order of the SCRFD model is: class 0, location 1, keypoint 2.
             */
            int classScore_pitch = _params._outputShape[classScoreIdx].back();
            int location_pitch = _params._outputShape[locationIdx].back();
            int keypoint_pitch = _params._outputShape[kptIdx].back();
            int classScoreDataSzie = 1, locationDataSize = 1;
            for(const auto &s:_params._outputShape[classScoreIdx])
                classScoreDataSzie *= s;
            for(const auto &s:_params._outputShape[locationIdx])
                locationDataSize *= s;
            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                int stride = layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                float* classScore_data = (float*)outputs;
                float* location_data = classScore_data + classScoreDataSzie;
                float* kpt_data = location_data + locationDataSize;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        float* score_data = classScore_data + (gY * numGridX * classScore_pitch) + (gX * classScore_pitch);
                        for(int idx=0;idx<2;idx++)
                        {
                            score1 = score_data[idx];
                            score = dxapp::yolo::sigmoid(score1);
                            if(score > _params._score_threshold)
                            {
                                _scoreIndices[0].emplace_back(score, boxIdx);
                                data = location_data + (gY * numGridX * location_pitch) + (gX * location_pitch) + (idx * 4);
                                kpt = kpt_data + (gY * numGridX * keypoint_pitch) + (gX * keypoint_pitch) + (idx * kpt_count * 2);
                                _rawVector.clear();
                                _rawVector.emplace_back(data);
                                _rawVector.emplace_back(kpt);
                                dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), 
                                                                   dxapp::common::Size(_params._input_shape[2], _params._input_shape[1]), stride, (float)kpt_count);
                                _rawBoxes.emplace_back(temp);
                                boxIdx++;
                            }
                        }
                    }
                }
                classScoreIdx += 3;
                locationIdx += 3;
                kptIdx += 3;
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };
    
        void getBoxesFromYoloXFormat(uint8_t* outputs, int locationIndex, int boxScoreIndex, int classScoreIndex)
        {
            int boxIdx = 0;
            float score, score1;
            float *data, *classScore;
            float* output_per_layers = (float*)outputs;
            for(int i=0;i<(int)_params._layers.size();i++)
            {
                /***
                 * The output order of the yolox model is: location 0, boxes score 1, class score 2.
                 */
                int location_pitch = _params._outputShape[locationIndex].back();
                int boxes_pitch = _params._outputShape[boxScoreIndex].back();
                int classes_pitch = _params._outputShape[classScoreIndex].back();
                int locationDataSize = 1, boxesDataSize = 1, classesDataSize = 1;
                for(const auto &s:_params._outputShape[locationIndex])
                    locationDataSize *= s;
                for(const auto &s:_params._outputShape[boxScoreIndex])
                    boxesDataSize *= s;
                for(const auto &s:_params._outputShape[classScoreIndex])
                    classesDataSize *= s;
                auto layer = _params._layers[i];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                float* location_data = output_per_layers;
                float* boxScore_data = location_data + locationDataSize;
                float* classScore_data = boxScore_data + boxesDataSize;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        bool boxDecoded = false;  
                        score1 = boxScore_data[(gY * numGridX * boxes_pitch) + (gX * boxes_pitch)];
                        if(score1 > _params._score_threshold)
                        {
                            data = location_data + (gY * numGridX * location_pitch) + (gX * location_pitch);
                            classScore = classScore_data + (gY * numGridX * classes_pitch) + (gX * classes_pitch);
                            for(int cls=0;cls<_params._numOfClasses;cls++)
                            {
                                score = score1 * classScore[cls];
                                if(score > _params._score_threshold)
                                {
                                    _scoreIndices[cls].emplace_back(score, boxIdx);
                                    if(!boxDecoded)
                                    {
                                        _rawVector.clear();
                                        _rawVector.emplace_back(data);
                                        dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), dxapp::common::Size(0, 0), stride, scale);
                                        _rawBoxes.emplace_back(temp);
                                        boxDecoded = true;
                                        boxIdx++;
                                    }
                                }
                            }
                        }
                    }
                }
                boxScoreIndex += 3;
                locationIndex += 3;
                classScoreIndex += 3;
                output_per_layers += locationDataSize + boxesDataSize + classesDataSize;
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };
        
        void getBoxesFromYoloFormat(uint8_t* outputs)
        {
            int boxIdx = 0;
            float rawThreshold = inversSigmoid(_params._score_threshold);
            float score, score1;
            float* output_per_layers = (float*)outputs;

            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                auto output_shape = _params._outputShape[i];
                int layer_pitch = 1;
                if(i > 0)
                {
                    for(const auto &s:_params._outputShape[i-1])
                    {
                        layer_pitch *= s;
                    }
                    output_per_layers += layer_pitch;
                }
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int box=0; box<(int)layer.anchor_width.size(); box++)
                        { 
                            bool boxDecoded = false;  
                            float* data = output_per_layers + (gY * numGridX * output_shape.back())
                                                                + (gX * output_shape.back())
                                                                + (box * (_params._numOfClasses + 5));
                            if(data[4]>rawThreshold)
                            {
                                score1 = _params._last_activation(data[4]);
                                if(score1 > _params._score_threshold)
                                {
                                    for(int cls=0; cls<_params._numOfClasses;cls++)
                                    {
                                        score = score1 * _params._last_activation(data[5+cls]); 
                                        if (score > _params._score_threshold)
                                        {
                                            _scoreIndices[cls].emplace_back(score, boxIdx);
                                            if(!boxDecoded)
                                            {
                                                _rawVector.clear();
                                                _rawVector.emplace_back(data);
                                                dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), dxapp::common::Size(layer.anchor_width[box], layer.anchor_height[box]), stride, scale);
                                                _rawBoxes.emplace_back(temp);
                                                boxDecoded = true;
                                                boxIdx++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };

        void getBoxesFromYoloV8Format(uint8_t* outputs)
        {
            
            std::cout << "[ERR:dx-app] not supported yolov8 all decode function." << std::endl;
            std::cout << "[ERR:dx-app] please using \"USE_ORT=ON\" option in dx_rt." << std::endl;

        };
        
        void getBoxesFromCustomPostProcessing(uint8_t* outputs /* Users can add necessary parameters manually. */)
        {
            /**
             * @brief adding your post processing code
             * 
             * example code ..
             * 
             * int boxIdx = 0;
             * std::shared_ptr<dxrt::Tensor>> node_a;
             * std::shared_ptr<dxrt::Tensor>> node_b;
             * std::shared_ptr<dxrt::Tensor>> node_c;
             * for(int i=0; i<outputs.size(); i++)
             * {
             *      if (outputs[i]->name() == "node_a")
             *          node_a = outputs[i];
             *      else if (outputs[i]->name() == "node_b")
             *          node_b = outputs[i];
             *      else if (outputs[i]->name() == "node_c")
             *          node_c = outputs[i];
             * }
             * 
             * 
             */
        };
    };

} // namespace yolo
} // namespace dxapp

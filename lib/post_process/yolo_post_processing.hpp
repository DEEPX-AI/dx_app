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
            default:
                _decode = dxapp::decode::yoloBasicDecode;
                break;
            }
            if(_params._ppu_format > 0)
            {
                /* layer re-ordering */
                std::sort(_params._layers.begin(), _params._layers.end(), 
                            [&](const Layers &a, const Layers &b)
                            {
                                return a.stride > b.stride;
                            });             
            }
        };
        PostProcessing(){};
        ~PostProcessing(){};
        dxapp::common::DetectObject getResult(){return _result;};
        void run(std::vector<std::shared_ptr<dxrt::Tensor>> outputs)
        {
            for(auto &indices:_scoreIndices){
                indices.clear();
            }
            _rawBoxes.clear();
            _result._detections.clear();
            _result._num_of_detections = 0;

            if(_params._ppu_format > 0) // shape : [124, ], numBoxes = 124
                getBoxesFromPPUOutputs(outputs, outputs.front()->shape()[0], _params._ppu_format);
            else if(_params._outputShape.size() == 1) // ONNX outputs shape : [1, 16384, 85], numBoxes = 16384
                getBoxesFromONNXOutputs(outputs, outputs.front()->shape()[1]); 
            else if(_params._decode_method==Decode::YOLOX) // anchor free detector
                getBoxesFromYoloXFormat(outputs, _yoloxLocationIdx, _yoloxBoxScoreIdx, _yoloxClassScoreIdx);
            else if(_params._decode_method==Decode::YOLO_POSE) // yolo pose 
                getBoxesFromYoloPoseFormat(outputs, _params._kpt_order, _params._kpt_count);
            else if(_params._decode_method==Decode::SCRFD) // face detection
                getBoxesFromSCRFDFormat(outputs, _scrfdClassScoreIdx, _scrfdLocationIdx, _scrfdKptIdx, _params._kpt_count);
            else if(_params._decode_method==Decode::CUSTOM_DECODE)
                getBoxesFromCustomPostProcessing(outputs);
            else
                getBoxesFromYoloFormat(outputs);
            
            dxapp::common::nms(_rawBoxes, _scoreIndices, _params._iou_threshold, _params._classes, _postprocPaddedSize, _postprocScaleRatio, _result);
        };

        static bool scoreComapre(const std::pair<float, int> &a, const std::pair<float, int> &b)
        {
            if(a.first > b.first)
                return true;
            else
                return false;
        };

        void getBoxesFromPPUOutputs(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, int64_t numDetected, dxapp::yolo::PPUFormat ppuFormat)
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
                uint8_t *raw_data = (uint8_t*)outputs.front()->data() + (i * ppuDataSize);
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

        void getBoxesFromONNXOutputs(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, int64_t numDetected)
        {
            int boxIdx = 0;
            float score, score1;
            float* raw_data = (float*)(outputs.front()->data());
            for(int64_t i=0;i<numDetected;i++)
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
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };
        
        void getBoxesFromYoloPoseFormat(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, KeyPointOrder kpt_order, int kpt_count)
        {
            int boxIdx = 0;
            float rawThreshold = inversSigmoid(_params._score_threshold);
            float score, score1;
            int first = -2, second = -1;
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
                                rawBuffer1 = (float*)(outputs[first]->data(gY, gX, 0));
                                rawBuffer2 = (float*)(outputs[first]->data(gY, gX, 6));
                                rawBuffer3 = (float*)(outputs[second]->data(gY, gX, 0));
                            }
                            else if (box == 1)
                            {
                                rawBuffer1 = (float*)(outputs[second]->data(gY, gX, (13 * 3)));
                                rawBuffer2 = (float*)(outputs[second]->data(gY, gX, (13 * 3) + 6));
                                rawBuffer3 = (float*)(outputs[second]->data(gY, gX, (17 * 3) + 6));
                            }
                            else if (box == 2)
                            {
                                rawBuffer1 = (float*)(outputs[second]->data(gY, gX, (30 * 3) + 6));
                                rawBuffer2 = (float*)(outputs[second]->data(gY, gX, (30 * 3) + 6 + 6));
                                rawBuffer3 = (float*)(outputs[second]->data(gY, gX, (34 * 3) + 6 + 6));
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
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };

        void getBoxesFromSCRFDFormat(std::vector<std::shared_ptr<dxrt::Tensor>> outputs,  int classScoreIdx, int locationIdx, int kptIdx, int kpt_count)
        {
            int boxIdx = 0;
            float score, score1;
            float *data, *kpt;

            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int idx=0;idx<2;idx++)
                        {
                            score1 = *(float*)(outputs[classScoreIdx]->data(gY,gX,idx));
                            score = dxapp::yolo::sigmoid(score1);
                            if(score > _params._score_threshold)
                            {
                                _scoreIndices[0].emplace_back(score, boxIdx);
                                data = (float*)(outputs[locationIdx]->data(gY,gX,idx*4));
                                kpt = (float*)(outputs[kptIdx]->data(gY,gX,idx*kpt_count*2));
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
    
        void getBoxesFromYoloXFormat(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, int locationIndex, int boxScoreIndex, int classScoreIndex)
        {
            int boxIdx = 0;
            float score, score1;
            float *data, *classScore;

            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        bool boxDecoded = false;  
                        score1 = *(float*)(outputs[boxScoreIndex]->data(gY,gX,0));
                        if(score1 > _params._score_threshold)
                        {
                            data = (float*)(outputs[locationIndex]->data(gY,gX,0));
                            classScore = (float*)(outputs[classScoreIndex]->data(gY,gX,0));
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
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };
        
        void getBoxesFromYoloFormat(std::vector<std::shared_ptr<dxrt::Tensor>> outputs)
        {
            int boxIdx = 0;
            float rawThreshold = inversSigmoid(_params._score_threshold);
            float score, score1;

            for(int i=0;i<(int)_params._layers.size();i++)
            {
                auto layer = _params._layers[i];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _params._input_shape[2] / layer.stride;
                int numGridY = _params._input_shape[1] / layer.stride;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int box=0; box<(int)layer.anchor_width.size(); box++)
                        { 
                            bool boxDecoded = false;  
                            float* data = (float*)(outputs[i]->data(gY, gX, box*(4 + 1 + _params._numOfClasses)));
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
        
        void getBoxesFromCustomPostProcessing(std::vector<std::shared_ptr<dxrt::Tensor>> outputs /* Users can add necessary parameters manually. */)
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

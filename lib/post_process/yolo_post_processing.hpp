#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include "dxapp_api.hpp"
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
        SCRFD,
        YOLO_POSE,
        CUSTOM_DECODE,
        BBOX,
        FACE,
        POSE,
    };
    enum BBoxFormat
    {
        CXCYWH=0,
        XYX2Y2,
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
        BBoxFormat _box_format;
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

        std::function<dxapp::common::BBox(std::function<float(float)>, float*, dxapp::common::Point, dxapp::common::Size, int, float)> _decode;

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
            default:
                _decode = dxapp::decode::yoloBasicDecode;
                break;
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
            if(_params._decode_method==Decode::YOLOX)
                getBoxes(outputs, _yoloxLocationIdx, _yoloxBoxScoreIdx, _yoloxClassScoreIdx);
            else
                getBoxes(outputs);
            
            dxapp::common::nms(_rawBoxes, _scoreIndices, _params._iou_threshold, _params._classes, _postprocPaddedSize, _postprocScaleRatio, _result);
        };
        
        void getBoxes(std::vector<std::shared_ptr<dxrt::Tensor>> outputs)
        {
            int boxIdx = 0;
            float rawThreshold = inversSigmoid(_params._score_threshold);
            float score, score1;

            if(_params._layers.front().anchor_height.size()>0)
            {
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
                            for(int box=0; box<(int)layer.anchor_width.size(); box++)
                            { 
                                float scale = layer.scale;
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
                                                    dxapp::common::BBox temp = _decode(_params._last_activation, data, dxapp::common::Point(gX, gY), dxapp::common::Size(layer.anchor_width[box], layer.anchor_height[box]), stride, scale);
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
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), std::greater<>());
            }
        };
    
        void getBoxes(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, int locationIndex, int boxScoreIndex, int classScoreIndex)
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
                                        dxapp::common::BBox temp = _decode(_params._last_activation, data, dxapp::common::Point(gX, gY), dxapp::common::Size(0, 0), stride, scale);
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
                sort(indices.begin(), indices.end(), std::greater<>());
            }
        };
    
    };
    

} // namespace yolo
} // namespace dxapp

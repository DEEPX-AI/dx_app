#pragma once

#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>

#include <dxrt/dxrt_api.h>
#include <common/objects.hpp>
#include <utils/box_decode.hpp>
#include <utils/nms.hpp>
#include <utils/common_util.hpp>

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
        YOLO_FACE,
        SCRFD,
        YOLOV8,
        YOLOV9,
        CUSTOM_DECODE,
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
        std::string name = "";
        int stride = 0;
        float scale = 0;
        std::vector<int64_t> shape = {};
        std::vector<float> anchor_width = {};
        std::vector<float> anchor_height = {};
    };
    struct Params 
    {
        // Configuration (small types first)
        bool _is_onnx_output = false;
        Decode _decode_method = Decode::YOLO_BASIC;
        BBoxFormat _box_format = BBoxFormat::CXCYWH;
        KeyPointOrder _kpt_order = KeyPointOrder::KPT_FRONT;
        int _kpt_count = 0;
        int _numOfClasses = 0;
        float _objectness_threshold = 0.25f;
        float _score_threshold = 0.3f;
        float _iou_threshold = 0.5f;
        
        // Large objects (vectors, functions, maps)
        dxapp::common::Size _input_size = {0, 0};
        std::vector<std::vector<int64_t>> _outputShape = {};
        std::function<float(float)> _last_activation = [](float x){return x;};
        std::vector<Layers> _layers = {};
        std::vector<std::string> _final_outputs = {};
        std::map<uint16_t, std::string> _classes = {};
        std::vector<std::pair<int, int>> _outputTensorIndexMap = {};
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

        std::vector<float> _rawVector;

        std::function<dxapp::common::BBox(std::function<float(float)>, std::vector<float>, dxapp::common::Point, dxapp::common::Size, int, float)> _decode;

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
            case Decode::YOLO_FACE:
                _decode = dxapp::decode::yoloFaceDecode;
                break;
            case Decode::YOLOSCALE:
                _decode = dxapp::decode::yoloScaledDecode;
                break;    
            case Decode::CUSTOM_DECODE:
                _decode = dxapp::decode::yoloCustomDecode;
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

        void initialize_results()
        {
            for(auto &indices:_scoreIndices){
                indices.clear();
            }
            _rawBoxes.clear();
            _result._detections.clear();
            _result._num_of_detections = 0;
        }

        void run(dxrt::TensorPtrs output_data)
        {
            initialize_results();
            if(_params._is_onnx_output && dxapp::common::checkOrtLinking()) // ONNX outputs shape : [1, 16384, 85], numBoxes = 16384
                getBoxesFromONNXOutputs(output_data); 
            else if(_params._decode_method==Decode::YOLOX) // anchor free detector
                getBoxesFromYoloXFormat(output_data, _yoloxLocationIdx, _yoloxBoxScoreIdx, _yoloxClassScoreIdx);
            else if(_params._decode_method==Decode::YOLO_FACE) // yolo face
                getBoxesFromYoloFaceFormat(output_data, _params._kpt_count);
            else if(_params._decode_method==Decode::YOLO_POSE) // yolo pose 
                getBoxesFromYoloPoseFormat(output_data, _params._kpt_order, _params._kpt_count);
            else if(_params._decode_method==Decode::YOLOV8)
                getBoxesFromYoloV8Format(output_data);
            else if(_params._decode_method==Decode::YOLOV9)
                getBoxesFromYoloV9Format(output_data);
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

        void getBoxesFromONNXOutputs(dxrt::TensorPtrs outputs)
        {
            int boxIdx = 0;
            float objectness_score, class_score;
            if (_params._decode_method == dxapp::yolo::Decode::YOLOV8 ||
                _params._decode_method == dxapp::yolo::Decode::YOLOV9)
            {
                /**
                 * @note Ultralytics models, which make yolov9/v8/v5/v7 has same post processing methods
                 *      https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-CPP/inference.cpp
                 */
                auto layer_idx = _params._outputTensorIndexMap[0].first;
                auto tensor_key_idx = _params._outputTensorIndexMap[0].second;
                auto strideNum = _params._layers[layer_idx].shape[2]; // 8400
                auto signalResultNum = _params._layers[layer_idx].shape[1]; // 84
                auto *raw_data = static_cast<float*>(outputs[tensor_key_idx]->data());
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
            else if (_params._decode_method == dxapp::yolo::Decode::YOLO_FACE)
            {
                auto layer_idx = _params._outputTensorIndexMap[0].first;
                auto tensor_key_idx = _params._outputTensorIndexMap[0].second;
                auto num_detections = _params._layers[layer_idx].shape[1]; // 252000
                auto num_elements = _params._layers[layer_idx].shape[2]; // 16
                auto *raw_data = static_cast<float*>(outputs[tensor_key_idx]->data());

                for(int64_t i=0;i<num_detections;i++)
                {
                    float* data = raw_data + num_elements * i; // num_elements: (4 + 1 + (_params._kpt_count * 2) + 1)
                    float obj_conf = data[4];
                    float cls_conf = data[15];
                    float conf = obj_conf * cls_conf;
                    
                    if (conf < _params._score_threshold) {
                        continue;
                    }

                    _scoreIndices[0].emplace_back(conf, boxIdx);
                    dxapp::common::BBox temp {
                                data[0] - data[2]/2.f,
                                data[1] - data[3]/2.f,
                                data[0] + data[2]/2.f,
                                data[1] + data[3]/2.f,
                                data[2],
                                data[3],
                                {dxapp::common::Point_f(-1, -1, -1)}
                    };

                    temp._kpts.clear();
                    for(int idx = 0; idx < _params._kpt_count; idx++)
                    {
                        int kptIdx = (idx * 2) + (5);
                        temp._kpts.emplace_back(dxapp::common::Point_f(
                            data[kptIdx + 0],
                            data[kptIdx + 1],
                            0.5f
                        ));
                    }
                    _rawBoxes.emplace_back(temp);
                    boxIdx++;
                }
            }
            else if (_params._decode_method == dxapp::yolo::Decode::SCRFD)
            {
                std::vector<int> strides = {8, 16, 32};

                auto *score_8 = static_cast<float*>(outputs[_params._outputTensorIndexMap[0].second]->data());
                auto *score_16 = static_cast<float*>(outputs[_params._outputTensorIndexMap[1].second]->data());
                auto *score_32 = static_cast<float*>(outputs[_params._outputTensorIndexMap[2].second]->data());
                std::vector<float*> score_list = {score_8, score_16, score_32};
                auto *bbox_8 = static_cast<float*>(outputs[_params._outputTensorIndexMap[3].second]->data());
                auto *bbox_16 = static_cast<float*>(outputs[_params._outputTensorIndexMap[4].second]->data());
                auto *bbox_32 = static_cast<float*>(outputs[_params._outputTensorIndexMap[5].second]->data());
                std::vector<float*> bbox_list = {bbox_8, bbox_16, bbox_32};
                auto *kpt_8 = static_cast<float*>(outputs[_params._outputTensorIndexMap[6].second]->data());
                auto *kpt_16 = static_cast<float*>(outputs[_params._outputTensorIndexMap[7].second]->data());
                auto *kpt_32 = static_cast<float*>(outputs[_params._outputTensorIndexMap[8].second]->data());
                std::vector<float*> kpt_list = {kpt_8, kpt_16, kpt_32};
                for(int i=0;i<static_cast<int>(strides.size());i++)
                {
                    int num_detections = std::pow(_inputSize._width / strides[i], 2) * 2;
                    for(int j=0;j<num_detections;j++)
                    {
                        int box_idx = j * 4;
                        int kpt_idx = j * 10;
                        auto *score_ptr = score_list[i];
                        auto *bbox_ptr = bbox_list[i];
                        auto *kpt_ptr = kpt_list[i];
                        if(score_ptr[j] > _params._score_threshold)
                        {
                            _scoreIndices[0].emplace_back(score_ptr[j], boxIdx);

                            int grid_x = (j / 2) % (_inputSize._width / strides[i]);
                            int grid_y = (j / 2) / (_inputSize._height / strides[i]);

                            float cx = static_cast<float>(grid_x * strides[i]);
                            float cy = static_cast<float>(grid_y * strides[i]);

                            dxapp::common::BBox temp {
                                        cx - (bbox_ptr[box_idx + 0] * strides[i]),
                                        cy - (bbox_ptr[box_idx + 1] * strides[i]),
                                        cx + (bbox_ptr[box_idx + 2] * strides[i]),
                                        cy + (bbox_ptr[box_idx + 3] * strides[i]),
                                        (bbox_ptr[box_idx + 0] + bbox_ptr[box_idx + 2]) * strides[i],
                                        (bbox_ptr[box_idx + 1] + bbox_ptr[box_idx + 3]) * strides[i],
                                        {dxapp::common::Point_f(-1, -1, -1)}
                            };
                            temp._kpts.clear();
                            for(int idx = 0; idx < _params._kpt_count; idx++)
                            {
                                temp._kpts.emplace_back(dxapp::common::Point_f(
                                                cx + (kpt_ptr[kpt_idx + (idx * 2) + 0] * strides[i]),
                                                cy + (kpt_ptr[kpt_idx + (idx * 2) + 1] * strides[i]),
                                                0.5f
                                                ));
                            }
                            _rawBoxes.emplace_back(temp);
                            boxIdx++;
                        }
                    }
                }
            }
            else if (_params._decode_method <= dxapp::yolo::Decode::YOLO_POSE)
            {
                auto layer_idx = _params._outputTensorIndexMap[0].first;
                auto tensor_key_idx = _params._outputTensorIndexMap[0].second;
                auto num_detections = _params._layers[layer_idx].shape[1]; // 25200
                auto num_elements = _params._layers[layer_idx].shape[2]; // 85 or pose : 17 * 3 + 4 + 1 + numOfClasses
                auto *raw_data = static_cast<float*>(outputs[tensor_key_idx]->data());

                for(int64_t i=0;i<num_detections;i++)
                {
                    float* data = raw_data + (num_elements * i);
                    objectness_score = data[4];
                    if(objectness_score > _params._objectness_threshold)
                    {
                        int max_cls = -1;
                        float max_score = _params._score_threshold;
                        for(int cls=0;cls<_params._numOfClasses;cls++)
                        {
                            class_score = objectness_score * data[5+cls];
                            if(class_score > max_score)
                            {
                                max_cls = cls;
                                max_score = class_score;
                            }
                        }
                        if(max_cls > -1)
                        {
                            _scoreIndices[max_cls].emplace_back(max_score, boxIdx);
                            dxapp::common::BBox temp {
                                        data[0] - data[2]/2.f,
                                        data[1] - data[3]/2.f,
                                        data[0] + data[2]/2.f,
                                        data[1] + data[3]/2.f,
                                        data[2],
                                        data[3],
                                        {dxapp::common::Point_f(-1, -1, -1)}
                            };
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
                            _rawBoxes.emplace_back(temp);
                            boxIdx++;
                        }
                    }
                }
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }
        };

        void getBoxesFromYoloFaceFormat(dxrt::TensorPtrs outputs, int kpt_count)
        {

            int boxIdx = 0;
            float objectness_score, class_score;

            for(int i=0;i<(int)_params._outputTensorIndexMap.size();i++)
            {
                auto layer_idx = _params._outputTensorIndexMap[i].first;
                auto tensor_key_idx = _params._outputTensorIndexMap[i].second;
                auto layer = _params._layers[layer_idx];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _inputSize._width / layer.stride;
                int numGridY = _inputSize._height / layer.stride;
                auto output_shape = _params._outputShape[tensor_key_idx];
                auto* output_per_layer = static_cast<float*>(outputs[tensor_key_idx]->data());
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int box=0; box<(int)layer.anchor_width.size(); box++)
                        { 
                            objectness_score = _params._last_activation(output_per_layer[((box * (_params._numOfClasses + 15) + 15) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX]);

                            if(objectness_score > _params._objectness_threshold)
                            {
                                int max_cls = -1;
                                float max_score = _params._score_threshold;
                                for(int cls=0; cls<_params._numOfClasses;cls++)
                                {
                                    float score = output_per_layer[((box * (_params._numOfClasses + 15) + 4 + cls) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX];
                                    class_score = objectness_score * _params._last_activation(score); 
                                    if (class_score > max_score)
                                    {
                                        max_score = class_score;
                                        max_cls = cls;
                                    }
                                }
                                if(max_cls > -1)
                                {
                                    _scoreIndices[max_cls].emplace_back(max_score, boxIdx);
                                    _rawVector.clear();
                                    for(int j = 0; j < 4; j++)
                                    {
                                        _rawVector.emplace_back(output_per_layer[((box * (_params._numOfClasses + 15) + j) * numGridY * numGridX)
                                                                                    + (gY * numGridX) 
                                                                                    + gX]);
                                    }
                                    for(int j = 0; j < kpt_count; j++)
                                    {
                                        _rawVector.emplace_back(output_per_layer[((box * (_params._numOfClasses + 15) + 5 + (j * 2)) * numGridY * numGridX)
                                                                                    + (gY * numGridX) 
                                                                                    + gX]);
                                        _rawVector.emplace_back(output_per_layer[((box * (_params._numOfClasses + 15) + 6 + (j * 2)) * numGridY * numGridX)
                                                                                    + (gY * numGridX) 
                                                                                    + gX]);
                                    }
                                    dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), dxapp::common::Size(layer.anchor_width[box], layer.anchor_height[box]), stride, scale);
                                    _rawBoxes.emplace_back(temp);
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
        
        void getBoxesFromYoloPoseFormat(dxrt::TensorPtrs outputs, KeyPointOrder kpt_order, int kpt_count)
        {
            (void)outputs;  // Suppress unused parameter warning
            (void)kpt_order; // Suppress unused parameter warning
            (void)kpt_count; // Suppress unused parameter warning
            std::cout << "[ERR:dx-app] not supported yolo pose decode function." << std::endl;
            std::cout << "[ERR:dx-app] please using \"USE_ORT=ON\" option in dx_rt." << std::endl;
        };
    
        void getBoxesFromYoloXFormat(dxrt::TensorPtrs outputs, int locationIndex, int boxScoreIndex, int classScoreIndex)
        {
            (void)outputs;         // Suppress unused parameter warning
            (void)locationIndex;    // Suppress unused parameter warning
            (void)boxScoreIndex;    // Suppress unused parameter warning
            (void)classScoreIndex;  // Suppress unused parameter warning
            std::cout << "[ERR:dx-app] not supported yolo x decode function." << std::endl;
            std::cout << "[ERR:dx-app] please using \"USE_ORT=ON\" option in dx_rt." << std::endl;
        };
        
        void getBoxesFromYoloFormat(dxrt::TensorPtrs outputs)
        {
            int boxIdx = 0;
            float objectness_score, class_score;
            for(int i=0;i<(int)_params._outputTensorIndexMap.size();i++)
            {
                auto layer_idx = _params._outputTensorIndexMap[i].first;
                auto tensor_key_idx = _params._outputTensorIndexMap[i].second;
                auto layer = _params._layers[layer_idx];
                float scale = layer.scale;
                int stride = layer.stride;
                int numGridX = _inputSize._width / layer.stride;
                int numGridY = _inputSize._height / layer.stride;
                auto output_shape = _params._outputShape[tensor_key_idx];
                auto* output_per_layer = static_cast<float*>(outputs[tensor_key_idx]->data());
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        for(int box=0; box<(int)layer.anchor_width.size(); box++)
                        { 
                            objectness_score = _params._last_activation(output_per_layer[((box * (_params._numOfClasses + 5) + 4) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX]);

                            if(objectness_score > _params._objectness_threshold)
                            {
                                int max_cls = -1;
                                float max_score = _params._score_threshold;
                                for(int cls=0; cls<_params._numOfClasses;cls++)
                                {
                                    float score = output_per_layer[((box * (_params._numOfClasses + 5) + 5 + cls) * numGridY * numGridX)
                                                                        + (gY * numGridX) 
                                                                        + gX];
                                    class_score = objectness_score * _params._last_activation(score); 
                                    if (class_score > max_score)
                                    {
                                        max_score = class_score;
                                        max_cls = cls;
                                    }
                                }
                                if(max_cls > -1)
                                {
                                    _scoreIndices[max_cls].emplace_back(max_score, boxIdx);
                                    _rawVector.clear();
                                    for(int j = 0; j < 4; j++)
                                    {
                                        _rawVector.emplace_back(output_per_layer[((box * (_params._numOfClasses + 5) + j) * numGridY * numGridX)
                                                                                    + (gY * numGridX) 
                                                                                    + gX]);
                                    }
                                    dxapp::common::BBox temp = _decode(_params._last_activation, _rawVector, dxapp::common::Point(gX, gY), dxapp::common::Size(layer.anchor_width[box], layer.anchor_height[box]), stride, scale);
                                    _rawBoxes.emplace_back(temp);
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

        void getBoxesFromYoloV8Format(dxrt::TensorPtrs outputs)
        {
            /**
             * @note Ultralytics models, which make yolov9/v8/v5/v7 has same post processing methods
             *      https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-ONNXRuntime-CPP/inference.cpp
             * 
             * yolov8n.dxnn output shapes are follow.
             *     Outputs
             *      /model.22/Sigmoid_output_0, FLOAT, [1, 80, 8400 ]
             *      /model.22/dfl/conv/Conv_output_0, FLOAT, [1, 1, 4, 8400]
             */

            int boxIdx = 0;
            float score;

            int scores_tensor_idx = _params._outputTensorIndexMap[0].second;
            int boxes_tensor_idx = _params._outputTensorIndexMap[1].second;
            float* boxes_output_tensor = static_cast<float*>(outputs[boxes_tensor_idx]->data());
            float* scores_output_tensor = static_cast<float*>(outputs[scores_tensor_idx]->data());
            int boxes_pitch_size = outputs[boxes_tensor_idx]->shape()[3];
            int score_pitch_size = outputs[scores_tensor_idx]->shape()[2];
            std::vector<int> feature_strides = {8, 16, 32};
            int index = -1;
            for(int i=0;i<(int)feature_strides.size();i++)
            {
                int stride = feature_strides[i];
                int numGridX = _params._input_size._width / stride;
                int numGridY = _params._input_size._height / stride;
                for(int gY=0; gY<numGridY; gY++)
                {
                    for(int gX=0; gX<numGridX; gX++)
                    {
                        index++;
                        int max_cls = -1;
                        float max_score = _params._score_threshold;
                        for(int cls=0;cls<_params._numOfClasses;cls++)
                        {
                            // score = *(float*)(scores_output_tensor + (score_pitch_size * index) + cls);
                            score = scores_output_tensor[(cls * score_pitch_size) + index];
                            if(score > max_score)
                            {
                                max_cls = cls;
                                max_score = score;
                            }
                        }
                        if(max_cls > -1)
                        {
                            _scoreIndices[max_cls].emplace_back(max_score, boxIdx);
                            std::vector<float> data(4);
                            float _605output01 = boxes_output_tensor[(0 * boxes_pitch_size) + index];
                            float _605output02 = boxes_output_tensor[(1 * boxes_pitch_size) + index];
                            float _608output01 = boxes_output_tensor[(2 * boxes_pitch_size) + index];
                            float _608output02 = boxes_output_tensor[(3 * boxes_pitch_size) + index];

                            float _605output01_s = (_605output01 * (-1) + (0.5f + gX));
                            float _605output02_s = (_605output02 * (-1) + (0.5f + gY));
                            float _608output01_s = (_608output01 + (0.5f + gX));
                            float _608output02_s = (_608output02 + (0.5f + gY));

                            _605output01 = _608output01_s - _605output01_s;
                            _605output02 = _608output02_s - _605output02_s;
                            _608output01 = (_608output01_s + _605output01_s) * 0.5; // 613
                            _608output02 = (_608output02_s + _605output02_s) * 0.5; // 613

                            data[0] = _608output01 * stride;
                            data[1] = _608output02 * stride;
                            data[2] = _605output01 * stride;
                            data[3] = _605output02 * stride;
                            dxapp::common::BBox temp {
                                data[0] - data[2]/2.f,
                                data[1] - data[3]/2.f,
                                data[0] + data[2]/2.f,
                                data[1] + data[3]/2.f,
                                data[2],
                                data[3],
                                {dxapp::common::Point_f(-1, -1, -1)}
                            };
                            _rawBoxes.emplace_back(temp);
                            boxIdx++;
                        }
                    }
                }
            }
            for(auto &indices:_scoreIndices)
            {
                sort(indices.begin(), indices.end(), scoreComapre);
            }

        };

        void getBoxesFromYoloV9Format(dxrt::TensorPtrs outputs)
        {
            (void)outputs; // Suppress unused parameter warning
            std::cout << "[ERR:dx-app] not supported yolov9 decode function." << std::endl;
            std::cout << "[ERR:dx-app] please using \"USE_ORT=ON\" option in dx_rt." << std::endl;
        };
        
        void getBoxesFromCustomPostProcessing(dxrt::TensorPtrs outputs /* Users can add necessary parameters manually. */)
        {
            (void)outputs; // Suppress unused parameter warning
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

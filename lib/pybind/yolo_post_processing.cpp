#include <yolo_post_processing.hpp>

namespace py = pybind11;

YoloPostProcess::YoloPostProcess(py::dict config)
{
    if (config.contains("model"))
    {
        py::dict model_dict = config["model"].cast<py::dict>();

        if (model_dict.contains("param"))
        {

            py::dict param_dict = model_dict["param"].cast<py::dict>();

            float conf_threshold = param_dict.contains("conf_threshold") ? param_dict["conf_threshold"].cast<float>() : 0.25f;
            _param._confThreshold = conf_threshold;

            float score_threshold = param_dict.contains("score_threshold") ? param_dict["score_threshold"].cast<float>() : 0.3f;
            _param._scoreThreshold = score_threshold;

            float iou_threshold = param_dict.contains("iou_threshold") ? param_dict["iou_threshold"].cast<float>() : 0.4f;
            _param._iouThreshold = iou_threshold;

            if (param_dict.contains("last_activation"))
            {
                std::string last_activation = param_dict["last_activation"].cast<std::string>();

                if (last_activation == "sigmoid")
                {
                    _lastActivation = [](float x)
                    { return 1.0f / (1.0f + std::exp(-x)); };
                }
                else if (last_activation == "exp")
                {
                    _lastActivation = [](float x)
                    { return x; };
                }
                else
                {
                    std::cerr << "[Error] last_activation '" << last_activation << "' is not supported !" << std::endl;
                    std::terminate();
                }
            }
            else
            {
                _lastActivation = [](float x)
                { return 1.0f / (1.0f + std::exp(-x)); };
            }

            std::string decoding_method = param_dict.contains("decoding_method") ? param_dict["decoding_method"].cast<std::string>() : "yolo_basic";
            if (decoding_method == "yolo_basic" || decoding_method == "yolo_scale" || decoding_method == "yolox" || decoding_method == "yolo_pose" || decoding_method == "scrfd")
            {
                _param._decodeMethod = decoding_method;

                if (decoding_method == "yolo_pose")
                {
                    _param._numKeypoints = 17;
                }
                else if (decoding_method == "scrfd")
                {
                    _param._numKeypoints = 5;
                }
                else
                {
                    _param._numKeypoints = 0;
                }
            }
            else
            {
                std::cerr << "[Error] decoding_method '" << decoding_method << "' is not supported !" << std::endl;
                std::terminate();
            }

            std::string box_format = param_dict.contains("box_format") ? param_dict["box_format"].cast<std::string>() : "center";
            if (box_format == "center" || box_format == "corner")
            {
                _param._boxFormat = box_format;
            }
            else
            {
                std::cerr << "[Error] box_format '" << box_format << "' is not supported !" << std::endl;
                std::terminate();
            }

            if (param_dict.contains("layer"))
            {
                py::list layers = param_dict["layer"].cast<py::list>();

                for (size_t i = 0; i < layers.size(); i++)
                {
                    YoloLayerParam layerparam;

                    py::dict layer = layers[i].cast<py::dict>();

                    if (layer.contains("stride"))
                    {
                        int stride = layer["stride"].cast<int>();
                        layerparam._stride = stride;
                    }

                    if (layer.contains("anchor_width"))
                    {
                        py::list anchor_width = layer["anchor_width"].cast<py::list>();
                        layerparam._anchorWidth = py::cast<std::vector<float>>(anchor_width);
                    }

                    if (layer.contains("anchor_height"))
                    {
                        py::list anchor_height = layer["anchor_height"].cast<py::list>();
                        layerparam._anchorHeight = py::cast<std::vector<float>>(anchor_height);
                    }

                    if (layerparam._anchorWidth.size() != layerparam._anchorHeight.size())
                    {
                        std::cerr << "[Error] Size of 'anchor_width' and 'anchor_height' must be same !" << std::endl;
                        std::terminate();
                    }

                    float scale_x_y = layer.contains("scale_x_y") ? layer["scale_x_y"].cast<float>() : 0.0f;
                    layerparam._scaleXY = scale_x_y;

                    _param._layers.push_back(layerparam);
                }
            }
        }
        else
        {
            std::cerr << "[Error] 'param' key is missing in the config['model'] !" << std::endl;
            std::terminate();
        }
    }
    else
    {
        std::cerr << "[Error] 'model' key is missing in the config !" << std::endl;
        std::terminate();
    }

    if (config.contains("output"))
    {

        py::dict output_dict = config["output"].cast<py::dict>();

        if (output_dict.contains("classes"))
        {
            py::list classes = output_dict["classes"].cast<py::list>();
            _param._numClasses = classes.size();
        }
        else
        {
            std::cerr << "[Error] 'classes' key is missing in the config['output'] !" << std::endl;
            std::terminate();
        }
    }
    else
    {
        std::cerr << "[Error] 'output' key is missing in the config !" << std::endl;
        std::terminate();
    }
}

float YoloPostProcess::CalcIOU(float *box1, float *box2)
{
    float ovr_left = std::max(box1[0], box2[0]);
    float ovr_right = std::min(box1[2], box2[2]);
    float ovr_top = std::max(box1[1], box2[1]);
    float ovr_bottom = std::min(box1[3], box2[3]);
    float ovr_width = ovr_right - ovr_left;
    float ovr_height = ovr_bottom - ovr_top;
    if (ovr_width < 0 || ovr_height < 0)
        return 0;
    float overlap_area = ovr_width * ovr_height;
    float union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) +
                       (box2[2] - box2[0]) * (box2[3] - box2[1]) -
                       overlap_area;
    return overlap_area * 1.0 / union_area;
}

bool YoloPostProcess::ScoreCompare(std::pair<float, int> &a,
                                   std::pair<float, int> &b)
{
    if (a.first > b.first)
        return true;
    else
        return false;
}

void YoloPostProcess::NMS(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                          std::vector<float> &Boxes,
                          std::vector<float> &Keypoints,
                          std::vector<float> &Results)
{
    for (int cls = 0; cls < _param._numClasses; cls++)
    {
        if (ScoreIndices[cls].size() > 0)
        {
            int numCandidates = ScoreIndices[cls].size();
            std::vector<bool> valid(numCandidates);
            std::fill_n(valid.begin(), numCandidates, true);
            float iou;
            for (int i = 0; i < numCandidates; i++)
            {
                if (!valid[i])
                {
                    continue;
                }

                Results.push_back(Boxes[4 * ScoreIndices[cls][i].second]);
                Results.push_back(Boxes[4 * ScoreIndices[cls][i].second + 1]);
                Results.push_back(Boxes[4 * ScoreIndices[cls][i].second + 2]);
                Results.push_back(Boxes[4 * ScoreIndices[cls][i].second + 3]);
                Results.push_back(ScoreIndices[cls][i].first);
                Results.push_back((float)cls);

                if (Keypoints.size() > 0)
                {

                    for (int k = 0; k < _param._numKeypoints; k++)
                    {
                        Results.push_back(Keypoints[3 * _param._numKeypoints * ScoreIndices[cls][i].second + 3 * k]);
                        Results.push_back(Keypoints[3 * _param._numKeypoints * ScoreIndices[cls][i].second + 3 * k + 1]);
                        Results.push_back(Keypoints[3 * _param._numKeypoints * ScoreIndices[cls][i].second + 3 * k + 2]);
                    }
                }

                for (int j = i + 1; j < numCandidates; j++)
                {
                    if (!valid[j])
                    {
                        continue;
                    }
                    iou = CalcIOU(&Boxes[4 * ScoreIndices[cls][j].second],
                                  &Boxes[4 * ScoreIndices[cls][i].second]);
                    if (iou > _param._iouThreshold)
                    {
                        valid[j] = false;
                    }
                }
            }
        }
    }
}

void YoloPostProcess::ProcessPPU(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                                 std::vector<float> &Boxes,
                                 std::vector<float> &Keypoints,
                                 py::list &ie_output)
{
    int boxIdx = 0;
    int box0, box1, box2, box3;
    py::array output_arr = py::cast<py::array>(ie_output[0]);
    py::buffer_info output_arr_info = output_arr.request();
    int8_t *output_arr_ptr = static_cast<int8_t *>(output_arr_info.ptr);

    for (int i = 0; i < output_arr_info.shape[0]; i++)
    {

        int8_t *data = output_arr_ptr + i * output_arr_info.shape[1];

        float *box = reinterpret_cast<float *>(data);

        uint8_t gY = *(data + 16);
        uint8_t gX = *(data + 17);
        uint8_t anchorIdx = *(data + 18);
        uint8_t layerIdx = *(data + 19);

        float score = *reinterpret_cast<float *>(data + 20);

        uint32_t label;
        float *kpts;
        if (output_arr_info.shape[1] == 32)
        {
            label = *reinterpret_cast<uint32_t *>(data + 24);
        }
        else if (output_arr_info.shape[1] == 64)
        {
            label = 0;
            kpts = reinterpret_cast<float *>(data + 24);
        }
        else if (output_arr_info.shape[1] == 256)
        {
            label = 0;
            kpts = reinterpret_cast<float *>(data + 28);
        }
        else
        {
            std::cerr << "[Error] Unknown PPU Type !" << std::endl;
            std::terminate();
        }

        ScoreIndices[label].emplace_back(score, boxIdx);

        YoloLayerParam layer = _param._layers[layerIdx];

        if (_param._decodeMethod == "yolo_basic" || _param._decodeMethod == "yolo_pose")
        {
            box0 = (box[0] * 2. - 0.5 + gX) * layer._stride;
            box1 = (box[1] * 2. - 0.5 + gY) * layer._stride;
            box2 = (box[2] * box[2] * 4.) * layer._anchorWidth[anchorIdx];
            box3 = (box[3] * box[3] * 4.) * layer._anchorHeight[anchorIdx];
        }
        else if (_param._decodeMethod == "yolo_scale")
        {
            box0 = (box[0] * layer._scaleXY - 0.5 * (layer._scaleXY - 1) + gX) * layer._stride;
            box1 = (box[1] * layer._scaleXY - 0.5 * (layer._scaleXY - 1) + gY) * layer._stride;
            box2 = (box[2] * box[2] * 4.) * layer._anchorWidth[anchorIdx];
            box3 = (box[3] * box[3] * 4.) * layer._anchorHeight[anchorIdx];
        }
        else if (_param._decodeMethod == "yolox")
        {
            box0 = (gX + box[0]) * layer._stride;
            box1 = (gY + box[1]) * layer._stride;
            box2 = exp(box[2]) * layer._stride;
            box3 = exp(box[3]) * layer._stride;
        }
        else if (_param._decodeMethod == "scrfd")
        {
            box0 = (gX - box[0]) * layer._stride;
            box1 = (gY - box[1]) * layer._stride;
            box2 = (gX + box[2]) * layer._stride;
            box3 = (gY + box[3]) * layer._stride;
        }

        if (_param._boxFormat == "corner")
        {
            Boxes.push_back(box0); /*x1*/
            Boxes.push_back(box1); /*y1*/
            Boxes.push_back(box2); /*x2*/
            Boxes.push_back(box3); /*y2*/
        }
        else if (_param._boxFormat == "center")
        {
            Boxes.push_back(box0 - box2 / 2.); /*x1*/
            Boxes.push_back(box1 - box3 / 2.); /*y1*/
            Boxes.push_back(box0 + box2 / 2.); /*x2*/
            Boxes.push_back(box1 + box3 / 2.); /*y2*/
        }

        if (_param._decodeMethod == "yolo_pose")
        {
            for (int k = 0; k < _param._numKeypoints; k++)
            {
                Keypoints.push_back((kpts[3 * k + 0] * 2 - 0.5 + gX) * layer._stride);
                Keypoints.push_back((kpts[3 * k + 1] * 2 - 0.5 + gY) * layer._stride);
                Keypoints.push_back(_lastActivation(kpts[3 * k + 2]));
            }
        }
        else if (_param._decodeMethod == "scrfd")
        {
            for (int k = 0; k < _param._numKeypoints; k++)
            {
                Keypoints.push_back((kpts[2 * k + 0] + gX) * layer._stride);
                Keypoints.push_back((kpts[2 * k + 1] + gY) * layer._stride);
                Keypoints.push_back(0.5f);
            }
        }

        boxIdx++;
    }
}

void YoloPostProcess::ProcessONNX(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                                  std::vector<float> &Boxes,
                                  py::list &ie_output)
{
    int boxIdx = 0;
    int box0, box1, box2, box3;
    py::array output_arr = py::cast<py::array>(ie_output[0]);
    py::buffer_info output_arr_info = output_arr.request();
    float *output_arr_ptr = static_cast<float *>(output_arr_info.ptr);

    if (output_arr_info.shape[1] < output_arr_info.shape[2])
    {

        // yolov8, yolov9 has an output of shape (batchSize, 84,  8400)

        int n_dets = output_arr_info.shape[2];

        for (int i = 0; i < n_dets; i++)
        {

            int ClassId = 0;
            float maxClassScore = *(output_arr_ptr + 4 * n_dets + i);

            for (int class_idx = 0; class_idx < _param._numClasses; class_idx++)
            {

                float ClassScore = *(output_arr_ptr + (4 + class_idx) * n_dets + i);

                if (ClassScore > maxClassScore)
                {
                    maxClassScore = ClassScore;
                    ClassId = class_idx;
                }
            }

            if (maxClassScore > _param._scoreThreshold)
            {
                ScoreIndices[ClassId].emplace_back(maxClassScore, boxIdx);

                box0 = *(output_arr_ptr + i);
                box1 = *(output_arr_ptr + 1 * n_dets + i);
                box2 = *(output_arr_ptr + 2 * n_dets + i);
                box3 = *(output_arr_ptr + 3 * n_dets + i);

                if (_param._boxFormat == "corner")
                {
                    Boxes.push_back(box0); /*x1*/
                    Boxes.push_back(box1); /*y1*/
                    Boxes.push_back(box2); /*x2*/
                    Boxes.push_back(box3); /*y2*/
                }
                else if (_param._boxFormat == "center")
                {
                    Boxes.push_back(box0 - box2 / 2.); /*x1*/
                    Boxes.push_back(box1 - box3 / 2.); /*y1*/
                    Boxes.push_back(box0 + box2 / 2.); /*x2*/
                    Boxes.push_back(box1 + box3 / 2.); /*y2*/
                }

                boxIdx++;
            }
        }
    }
    else
    {
        // yolov5 has an output of shape (batchSize, 25200, 85)

        auto n_dets = output_arr_info.shape[1];
        auto dimensions = output_arr_info.shape[2];

        for (int i = 0; i < n_dets; i++)
        { // 25200

            float *data = output_arr_ptr + (dimensions * i);

            if (data[4] > _param._confThreshold)
            {

                float *classesScores = data + 5;

                int ClassId = 0;
                float maxClassScore = classesScores[0];

                for (int class_idx = 0; class_idx < _param._numClasses; class_idx++)
                {
                    if (classesScores[class_idx] > maxClassScore)
                    {
                        maxClassScore = classesScores[class_idx];
                        ClassId = class_idx;
                    }
                }

                float score = maxClassScore * data[4];
                if (score > _param._scoreThreshold)
                {

                    ScoreIndices[ClassId].emplace_back(score, boxIdx);

                    box0 = data[0];
                    box1 = data[1];
                    box2 = data[2];
                    box3 = data[3];

                    if (_param._boxFormat == "corner")
                    {
                        Boxes.push_back(box0); /*x1*/
                        Boxes.push_back(box1); /*y1*/
                        Boxes.push_back(box2); /*x2*/
                        Boxes.push_back(box3); /*y2*/
                    }
                    else if (_param._boxFormat == "center")
                    {
                        Boxes.push_back(box0 - box2 / 2.); /*x1*/
                        Boxes.push_back(box1 - box3 / 2.); /*y1*/
                        Boxes.push_back(box0 + box2 / 2.); /*x2*/
                        Boxes.push_back(box1 + box3 / 2.); /*y2*/
                    }

                    boxIdx++;
                }
            }
        }
    }
}

void YoloPostProcess::ProcessRAW(std::vector<std::vector<std::pair<float, int>>> &ScoreIndices,
                                 std::vector<float> &Boxes,
                                 py::list &ie_output)
{
    int boxIdx = 0;
    int box0, box1, box2, box3;
    float rawThreshold = std::log(_param._confThreshold / (1 - _param._confThreshold));

    for (size_t i = 0; i < ie_output.size(); i++)
    {
        YoloLayerParam layer = _param._layers[i];

        py::array output_arr = py::cast<py::array>(ie_output[i]);
        py::buffer_info output_arr_info = output_arr.request();
        float *output_arr_ptr = static_cast<float *>(output_arr_info.ptr);

        for (int gY = 0; gY < output_arr_info.shape[1]; gY++)
        {
            for (int gX = 0; gX < output_arr_info.shape[2]; gX++)
            {
                for (size_t anchorIdx = 0; anchorIdx < layer._anchorWidth.size(); anchorIdx++)
                {

                    float *data = output_arr_ptr + output_arr_info.shape[3] * output_arr_info.shape[2] * gY + output_arr_info.shape[3] * gX + anchorIdx * (4 + 1 + _param._numClasses);

                    if (data[4] > rawThreshold)
                    {

                        float objectness = _lastActivation(data[4]);

                        int ClassId = 0;
                        float maxClassScore = _lastActivation(data[5]);

                        for (int cls = 0; cls < _param._numClasses; cls++)
                        {
                            if (_lastActivation(data[5 + cls]) > maxClassScore)
                            {
                                maxClassScore = _lastActivation(data[5 + cls]);
                                ClassId = cls;
                            }
                        }

                        float score = objectness * maxClassScore;
                        if (score > _param._scoreThreshold)
                        {
                            ScoreIndices[ClassId].emplace_back(score, boxIdx);

                            if (_param._decodeMethod == "yolo_basic" || _param._decodeMethod == "yolo_pose")
                            {
                                box0 = (_lastActivation(data[0]) * 2. - 0.5 + gX) * layer._stride;
                                box1 = (_lastActivation(data[1]) * 2. - 0.5 + gY) * layer._stride;
                                box2 = pow(_lastActivation(data[2]) * 2, 2) * layer._anchorWidth[anchorIdx];
                                box3 = pow(_lastActivation(data[3]) * 2, 2) * layer._anchorHeight[anchorIdx];
                            }
                            else if (_param._decodeMethod == "yolo_scale")
                            {
                                box0 = (_lastActivation(data[0]) * layer._scaleXY - 0.5 * (layer._scaleXY - 1) + gX) * layer._stride;
                                box1 = (_lastActivation(data[1]) * layer._scaleXY - 0.5 * (layer._scaleXY - 1) + gY) * layer._stride;
                                box2 = pow(_lastActivation(data[2]) * 2, 2) * layer._anchorWidth[anchorIdx];
                                box3 = pow(_lastActivation(data[3]) * 2, 2) * layer._anchorHeight[anchorIdx];
                            }
                            else if (_param._decodeMethod == "yolox")
                            {
                                box0 = (gX + data[0]) * layer._stride;
                                box1 = (gY + data[1]) * layer._stride;
                                box2 = _lastActivation(data[2]) * layer._stride;
                                box3 = _lastActivation(data[3]) * layer._stride;
                            }
                            else if (_param._decodeMethod == "scrfd")
                            {
                                box0 = (gX - data[0]) * layer._stride;
                                box1 = (gY - data[1]) * layer._stride;
                                box2 = (gX + data[2]) * layer._stride;
                                box3 = (gY + data[3]) * layer._stride;
                            }

                            if (_param._boxFormat == "corner")
                            {
                                Boxes.push_back(box0); /*x1*/
                                Boxes.push_back(box1); /*y1*/
                                Boxes.push_back(box2); /*x2*/
                                Boxes.push_back(box3); /*y2*/
                            }
                            else if (_param._boxFormat == "center")
                            {
                                Boxes.push_back(box0 - box2 / 2.); /*x1*/
                                Boxes.push_back(box1 - box3 / 2.); /*y1*/
                                Boxes.push_back(box0 + box2 / 2.); /*x2*/
                                Boxes.push_back(box1 + box3 / 2.); /*y2*/
                            }

                            boxIdx++;
                        }
                    }
                }
            }
        }
    }
}

py::array_t<float> YoloPostProcess::Run(py::list ie_output)
{
    std::vector<float> Boxes;
    std::vector<float> Keypoints;
    std::vector<float> Results;
    std::vector<std::vector<std::pair<float, int>>> ScoreIndices;

    for (int cls = 0; cls < _param._numClasses; cls++)
    {
        std::vector<std::pair<float, int>> v;
        ScoreIndices.push_back(v);
    }

    if (ie_output.size() == 0)
    {
        std::cerr << "[Error] There is no inference engine output!" << std::endl;
        return py::array_t<float>();
    }
    else if (ie_output.size() == 1)
    {

        py::array output_arr = py::cast<py::array>(ie_output[0]);
        py::buffer_info output_arr_info = output_arr.request();

        if (output_arr_info.ndim == 2)
        {
            ProcessPPU(ScoreIndices, Boxes, Keypoints, ie_output);
        }
        else if (output_arr_info.ndim == 3)
        {
            ProcessONNX(ScoreIndices, Boxes, ie_output);
        }
        else
        {
            return py::array_t<float>();
        }
    }
    else if (ie_output.size() == 3)
    {
        ProcessRAW(ScoreIndices, Boxes, ie_output);
    }
    else
    {
        std::cerr << "[Error] Output Size '" << ie_output.size() << "' is not supported !" << std::endl;
        return py::array_t<float>();
    }

    for (int cls = 0; cls < _param._numClasses; cls++)
    {
        std::sort(ScoreIndices[cls].begin(), ScoreIndices[cls].end(), ScoreCompare);
    }

    NMS(ScoreIndices, Boxes, Keypoints, Results);

    size_t n_rows, n_cols;
    if (Keypoints.size() > 0)
    {
        n_cols = 6 + 3 * _param._numKeypoints; // x1,y1,x2,y2,score,class,kpt1_x,kpt1_y,kpt1_score,...
        n_rows = Results.size() / n_cols;
    }
    else
    {
        n_cols = 6; // x1,y1,x2,y2,score,class
        n_rows = Results.size() / n_cols;
    }

    py::array_t<float> output = py::array_t<float>(
        {n_rows, n_cols},
        Results.data());

    Boxes.clear();
    Boxes.shrink_to_fit();
    Keypoints.clear();
    Keypoints.shrink_to_fit();
    Results.clear();
    Results.shrink_to_fit();
    ScoreIndices.clear();
    ScoreIndices.shrink_to_fit();

    return output;
}
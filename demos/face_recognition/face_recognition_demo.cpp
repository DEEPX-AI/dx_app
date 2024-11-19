#include <getopt.h>
#include <cmath>
#include <chrono>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "dxrt/dxrt_api.h"
#include "ssd.h"
#include "utils.h"

#ifndef UNUSEDVAR
#define UNUSEDVAR(x) (void)(x);
#endif
// camera frame resolution (1920, 1080), (1280, 720), (800, 600)
#define CAMERA_FRAME_WIDTH 1920
#define CAMERA_FRAME_HEIGHT 1080

#define FD_INPUT_WIDTH 512
#define FD_INPUT_HEIGHT 512
#define FR_INPUT_WIDTH 112
#define FR_INPUT_HEIGHT 112

#define FR_THRESHOLD 0.5

static bool g_resultSaveMode = false;
static bool g_usingOrt = false;

static std::vector<string> age_classes = {
    "0-3", "14-24", "25-36", "37-48", "4-6", "49-58", "59-100", "7-13"
};

static std::vector<string> gender_classes = {"F", "M"};

std::vector<int> age_gender_post_processing(dxrt::TensorPtrs outputs)
{
    std::string age_name = "onnx::Concat_568", gender_name = "onnx::Concat_572";
    std::vector<int> result(2);
    int max_idx = 0;
    if(g_usingOrt)
    {
        float *data = (float*)outputs.front()->data();        
        for(int i=0;i<8;i++){
            if(data[i] > data[max_idx])
                max_idx = i;
        }
        result[0] = max_idx;
        for(int i=max_idx=8;i<10;i++){
            if(data[i] > data[max_idx]){
                max_idx = i;
            }
        }
        result[1] = max_idx;
        return result;
    }

    for (auto &output: outputs)
    {
        int max_idx = 0;
        float* data = (float*)output->data();
        if(output->name() == age_name)
        {
            for(int i=0;i<8;i++){
                if(data[i] > data[max_idx])
                    max_idx = i;
            }
            result[0] = max_idx;
        }
        else if (output->name() == gender_name)
        {
            for(int i=0;i<2;i++){
                if(data[i] > data[max_idx])
                    max_idx = i;
            }
            result[1] = max_idx;
        }
    }
    return result;
}

std::vector<int> run_age_gender(dxrt::InferenceEngine *ie, cv::Mat& image, int id)
{
    // RESIZE BY INTER AREA
    cv::Mat resized, input;
    cv::cvtColor(image, input, cv::COLOR_BGR2RGB);
    resize(input, resized, cv::Size(FR_INPUT_WIDTH, FR_INPUT_HEIGHT), 0.0, 0.0, cv::INTER_AREA);
    
    dxrt::TensorPtrs tensors;
    if(g_usingOrt) 
    {
        tensors = ie->Run(image.data);
    }
    else
    {
        std::vector<uint8_t>input_tensor(ie->input_size());
        data_pre_processing(resized.data, input_tensor.data(), FR_INPUT_WIDTH, 48);
        tensors = ie->Run(input_tensor.data());
    }
    std::vector<int> result = age_gender_post_processing(tensors);
    return result;
}

std::vector<cv::Point2f> run_landmark(dxrt::InferenceEngine *ie, cv::Mat image, cv::Rect crop)
{
    cv::Mat fl_cropped = image(crop);
    cv::Mat fl_input = preprocess(fl_cropped, cv::Size(192, 192));

    auto fl_tensors = ie->Run(fl_input.data);
    float *fl_data = (float *)fl_tensors[0]->data();

    auto landmark = get_landmark(fl_data, fl_cropped.cols, fl_cropped.rows, crop.x, crop.y);
    return landmark;
}

FaceData run_recognition(dxrt::InferenceEngine *ie, cv::Mat image, int id)
{
    cv::Mat input = preprocess(image, cv::Size(FR_INPUT_WIDTH, FR_INPUT_HEIGHT));
    dxrt::TensorPtrs tensors;
    cv::cvtColor(image, input, cv::COLOR_BGR2RGB);
    if (g_usingOrt) {
        tensors = ie->Run(image.data);
    } else {
        std::vector<uint8_t> input_tensor(ie->input_size());
        data_pre_processing(image.data, input_tensor.data(), FR_INPUT_WIDTH, 48);
        tensors = ie->Run(input_tensor.data());
    }
    float *feature_vector = (float *)tensors[0]->data();
    return FaceData(id, image, feature_vector);
}

std::vector<FaceData> get_gallary(std::string dir, Ssd *detector, dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr)
{
    std::vector<FaceData> gallary;
    if (dir == "")
    {
        cv::Mat face_image = cv::Mat::zeros(FR_INPUT_HEIGHT, FR_INPUT_WIDTH, CV_8UC3);
        auto face_data = run_recognition(ie_fr, face_image, 0);
        gallary.emplace_back(face_data);
    }
    else
    {
        auto files = dxrt::GetFileList(dir);
        for (auto &file_name : files)
        {
            std::string file_path = dir + "/" + file_name;
            cv::Mat image = cv::imread(file_path);

            cv::Mat fd_input = preprocess(image, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));

            auto fd_tensor = ie_fd->Run(fd_input.data);
            auto fd_result = detector->PostProc(fd_tensor);
            if (fd_result.size() == 0)
            {
                continue;
            }
            else
            {
                auto detected = fd_result[0];
                cv::Rect rect = get_rect(detected.box, image.cols, image.rows);
                auto landmark = run_landmark(ie_fl, image, rect);
                cv::Mat fr_warped = warp(image, landmark);
                auto face_data = run_recognition(ie_fr, fr_warped, 0);
                gallary.emplace_back(face_data);
            }
        }
    }
    return gallary;
}

cv::Mat make_gallary_view(std::vector<FaceData> gallary)
{
    std::vector<cv::Mat> gallary_images;
    for (size_t i = 0; i < gallary.size(); i++)
    {
        auto face_image = gallary[i].image.clone();
        cv::putText(face_image, " ID: " + std::to_string(i), cv::Point(0, 16), 0, 0.6, cv::Scalar(0, 255, 255), 2);
        gallary_images.emplace_back(face_image);
    }
    cv::Mat gallary_view;
    cv::hconcat(gallary_images, gallary_view);
    return gallary_view;
}

void run_image(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender, SsdParam fdCfg, std::string dbPath, std::string image_path, float frThreshold)
{
    if(!g_resultSaveMode)
    {
        cv::namedWindow("view");
        cv::moveWindow("view", 0, 0);
        cv::namedWindow("gallary");
        cv::moveWindow("gallary", 0, 780);
    }
    int key = 0;
    auto fdDataInfo = ie_fd->outputs();
    Ssd detector = Ssd(fdCfg, fdDataInfo);

    auto gallary = get_gallary(dbPath, &detector, ie_fd, ie_fl, ie_fr);

    std::vector<bool> visited_indices;
    for (size_t i = 0; i < gallary.size(); i++)
    {
        visited_indices.emplace_back(false);
    }

    cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);
    cv::Mat view = frame.clone();

    cv::Mat fd_input = preprocess(frame, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));
    auto fd_tensors = ie_fd->Run(fd_input.data);
    auto fd_result = detector.PostProc(fd_tensors);
    for (size_t i = 0; i < fd_result.size(); i++)
    {
        auto detected = fd_result[i];
        cv::Rect rect = get_rect(detected.box, frame.cols, frame.rows);

        // Visualization
        cv::Scalar color(0, 255, 0);
        cv::rectangle(view, rect, color, 2);
        cv::putText(view, detected.labelname, cv::Point(rect.x + 8, rect.y + 16), 0, 0.5, color, 1);

        auto landmark = run_landmark(ie_fl, frame, rect);
        visualize_landmark(view, landmark);

        cv::Mat fr_warped = warp(frame, landmark);
        
        auto face_data = run_recognition(ie_fr, fr_warped, i);
        if(ie_gender != 0)
        {
            auto age_gender_data = run_age_gender(ie_gender, fr_warped, i);
            face_data.age_idx = age_gender_data[0];
            face_data.gender_idx = age_gender_data[1];
        }

        float *feature_vector = face_data.feature_vector;
        float similarity_max = 0;
        int similarity_max_index = -1;
        for (size_t j = 0; j < gallary.size(); j++)
        {
            float similarity = cos_sim(gallary[j].feature_vector, feature_vector, 512);
            if (similarity > similarity_max && visited_indices[j] == false)
            {
                similarity_max = similarity;
                similarity_max_index = j;
            }
        }

        // caption : id 0 (similarity), age 14-26, (M)
        std::string id_str = "id ";
        if (similarity_max > frThreshold)
        {
            id_str += std::to_string(similarity_max_index);
            visited_indices[similarity_max_index] = true;
        }
        else
        {
            id_str += "?";
        }
        std::string caption_str = id_str + " (" + std::to_string(similarity_max) + ")";
        
        if(face_data.age_idx > 0) 
        {
            caption_str += ", age " + age_classes[face_data.age_idx] + ", (" + gender_classes[face_data.gender_idx] + ")";
        }
        int txtBaseline = 0;
        auto textSize = cv::getTextSize(caption_str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
        cv::rectangle(view, rect, color, 2);
        cv::rectangle(view, cv::Point(rect.x, rect.y - textSize.height -5), cv::Point(rect.x + textSize.width + 10, rect.y + 5), color, -1);
        cv::putText(view, caption_str, cv::Point(rect.x + 5, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }
    if(!g_resultSaveMode)
    {
        cv::imshow("view", view);
        cv::imshow("gallary", make_gallary_view(gallary));
        key = cv::waitKey(0);
    }
    else
    {
        cv::imwrite("./result.jpg", view);
        std::cout << "save detection and landmark result file : result.jpg " << std::endl;
    }
    UNUSEDVAR(key);    
}

void run_image2(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender, std::string image1_path, std::string image2_path)
{
    UNUSEDVAR(ie_fl);
    UNUSEDVAR(ie_fd);
    int key = 0;
    cv::Mat frame1 = cv::imread(image1_path, cv::IMREAD_COLOR);
    cv::Mat frame2 = cv::imread(image2_path, cv::IMREAD_COLOR);

    cv::Mat image1, image2;
    cv::resize(frame1, image1, cv::Size(FR_INPUT_WIDTH, FR_INPUT_HEIGHT));
    cv::resize(frame2, image2, cv::Size(FR_INPUT_WIDTH, FR_INPUT_HEIGHT));

    auto face_data1 = run_recognition(ie_fr, image1, 1);
    auto face_data2 = run_recognition(ie_fr, image2, 2);

    if(ie_gender != 0)
    {
        auto age_gender_data1 = run_age_gender(ie_gender, frame1, 1);
        face_data1.age_idx = age_gender_data1[0];
        face_data1.gender_idx = age_gender_data1[1];
        std::string face_data1_caption = "ID " + std::to_string(face_data1.id) 
                                    + " : age " + age_classes[face_data1.age_idx] 
                                    + ", (" + gender_classes[face_data1.gender_idx] + ")";
        std::cout << face_data1_caption << std::endl;

        auto age_gender_data2 = run_age_gender(ie_gender, frame2, 2);
        face_data2.age_idx = age_gender_data2[0];
        face_data2.gender_idx = age_gender_data2[1];
        std::string face_data2_caption = "ID " + std::to_string(face_data2.id) 
                                    + " : age " + age_classes[face_data2.age_idx] 
                                    + ", (" + gender_classes[face_data2.gender_idx] + ")";
        std::cout << face_data2_caption << std::endl;
    }

    float similarity = cos_sim(face_data1.feature_vector, face_data2.feature_vector, 512);

    cv::Mat view1, view;
    cv::hconcat(image1, image2, view1);

    cv::Mat log = cv::Mat::zeros(view1.rows, view1.cols, CV_8UC3);
    std::string sim_str = "Similarity: " + std::to_string(similarity);
    cv::putText(log, sim_str, cv::Point(8, 32), 0, 0.6, cv::Scalar(0, 255, 255), 2);

    cv::vconcat(view1, log, view);
    if(!g_resultSaveMode)
    {
        cv::imshow("view", view);
        key = cv::waitKey(0);
    }
    else
    {
        cv::imwrite("./result.jpg", view);
        std::cout << "save result file : result.jpg " << std::endl;
    }

    UNUSEDVAR(key);
}

void run_image3(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender, SsdParam fdCfg, std::string image1_path, std::string image2_path)
{
    int key = 0;
    auto fdDataInfo = ie_fd->outputs();
    Ssd detector = Ssd(fdCfg, fdDataInfo);

    cv::Mat frame1 = cv::imread(image1_path, cv::IMREAD_COLOR);
    cv::Mat frame2 = cv::imread(image2_path, cv::IMREAD_COLOR);
    cv::Mat view1 = frame1.clone();
    cv::Mat view2 = frame2.clone();

    cv::Mat fd_input1 = preprocess(frame1, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));
    cv::Mat fd_input2 = preprocess(frame2, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));

    cv::Scalar color(0, 255, 0);

    // for image1
    auto fd_tensors1 = ie_fd->Run(fd_input1.data);
    auto fd_result1 = detector.PostProc(fd_tensors1);
    auto detected1 = fd_result1[0];
    cv::Rect rect1 = get_rect(detected1.box, frame1.cols, frame1.rows);
    // Visualization
    cv::rectangle(view1, rect1, color, 2);
    cv::putText(view1, detected1.labelname, cv::Point(rect1.x + 8, rect1.y + 16), 0, 0.5, color, 1);
    auto landmark1 = run_landmark(ie_fl, frame1, rect1);
    visualize_landmark(view1, landmark1);
    cv::Mat fr_warped1 = warp(frame1, landmark1);
    
    // for image2
    auto fd_tensors2 = ie_fd->Run(fd_input2.data);
    auto fd_result2 = detector.PostProc(fd_tensors2);
    auto detected2 = fd_result2[0];
    cv::Rect rect2 = get_rect(detected2.box, frame2.cols, frame2.rows);
    // Visualization
    cv::rectangle(view2, rect2, color, 2);
    cv::putText(view2, detected2.labelname, cv::Point(rect2.x + 8, rect2.y + 16), 0, 0.5, color, 1);
    auto landmark2 = run_landmark(ie_fl, frame2, rect2);
    visualize_landmark(view2, landmark2);
    cv::Mat fr_warped2 = warp(frame2, landmark2);
    
    auto face_data1 = run_recognition(ie_fr, fr_warped1, 1);
    auto face_data2 = run_recognition(ie_fr, fr_warped2, 2);

    if(ie_gender != 0)
    {
        int txtBaseline = 0;
        auto age_gender_data1 = run_age_gender(ie_gender, fr_warped1, 1);
        face_data1.age_idx = age_gender_data1[0];
        face_data1.gender_idx = age_gender_data1[1];
        std::string face_data1_caption = "ID " + std::to_string(face_data1.id) 
                                    + " : age " + age_classes[face_data1.age_idx] 
                                    + ", (" + gender_classes[face_data1.gender_idx] + ")";
        std::cout << face_data1_caption << std::endl;
        
        auto textSize1 = cv::getTextSize(face_data1_caption, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
        cv::rectangle(view1, cv::Point(rect1.x, rect1.y - textSize1.height -5), cv::Point(rect1.x + textSize1.width + 10, rect1.y + 5), color, -1);
        cv::putText(view1, face_data1_caption, cv::Point(rect1.x + 5, rect1.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

        auto age_gender_data2 = run_age_gender(ie_gender, fr_warped2, 2);
        face_data2.age_idx = age_gender_data2[0];
        face_data2.gender_idx = age_gender_data2[1];
        std::string face_data2_caption = "ID " + std::to_string(face_data2.id) 
                                    + " : age " + age_classes[face_data2.age_idx] 
                                    + ", (" + gender_classes[face_data2.gender_idx] + ")";
        std::cout << face_data2_caption << std::endl;
                
        auto textSize2 = cv::getTextSize(face_data2_caption, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
        cv::rectangle(view2, cv::Point(rect2.x, rect2.y - textSize2.height -5), cv::Point(rect2.x + textSize2.width + 10, rect2.y + 5), color, -1);
        cv::putText(view2, face_data2_caption, cv::Point(rect2.x + 5, rect2.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
    }

    if(!g_resultSaveMode)
        cv::imshow("view1", view1), cv::imshow("view2", view2);

    float similarity = cos_sim(face_data1.feature_vector, face_data2.feature_vector, 512);

    cv::Mat view, face_warped;
    cv::hconcat(fr_warped1, fr_warped2, face_warped);

    cv::Mat log = cv::Mat::zeros(face_warped.rows, face_warped.cols, CV_8UC3);
    std::string sim_str = "Similarity: " + std::to_string(similarity);
    cv::putText(log, sim_str, cv::Point(8, 32), 0, 0.6, cv::Scalar(0, 255, 255), 2);

    cv::vconcat(face_warped, log, view);
    
    if(!g_resultSaveMode)
    {
        cv::imshow("view", view);
        key = cv::waitKey(0);
    }
    else
    {
        cv::imwrite("./result.jpg", view);
        std::cout << "save result file : result.jpg " << std::endl;
    }
    
    UNUSEDVAR(key);
}

void run_video_sync(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender, SsdParam fdCfg, std::string dbPath, std::string videoFile, bool cameraInput, float frThreshold)
{
    auto fdDataInfo = ie_fd->outputs();
    Ssd detector = Ssd(fdCfg, fdDataInfo);

    std::vector<int> face_ids;
    std::vector<cv::Mat> face_images;
    for (int i = 0; i < 10; i++)
    {
        face_ids.emplace_back(-1);
        face_images.emplace_back(cv::Mat::zeros(FR_INPUT_HEIGHT, FR_INPUT_WIDTH, CV_8UC3));
    }

    auto gallary = get_gallary(dbPath, &detector, ie_fd, ie_fl, ie_fr);
    std::vector<bool> visited_indices;
    for (size_t i = 0; i < gallary.size(); i++)
    {
        visited_indices.emplace_back(false);
    }

    cv::VideoCapture cap;
    if (cameraInput)
    {
        cap.open(0, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
    }
    else
    {
        cap.open(videoFile);
    }

    cv::namedWindow("view");
    cv::moveWindow("view", 0, 0);
    cv::namedWindow("gallary");
    cv::moveWindow("gallary", 0, 780);

    bool running = true;
    while (running)
    {
        // double timestamp = GetTimestamp();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        cv::Mat view = frame.clone();

        cv::Mat fd_input = preprocess(frame, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));
        auto fd_tensors = ie_fd->Run(fd_input.data);
        auto fd_result = detector.PostProc(fd_tensors);

        for (size_t i = 0; i < fd_result.size(); i++)
        {
            auto detected = fd_result[i];
            cv::Rect rect = get_rect(detected.box, frame.cols, frame.rows);

            // Visualization
            cv::Scalar color(0, 255, 0);
            cv::rectangle(view, rect, color, 2);
            cv::putText(view, detected.labelname, cv::Point(rect.x + 8, rect.y + 16), 0, 0.5, color, 1);

            auto landmark = run_landmark(ie_fl, frame, rect);
            visualize_landmark(view, landmark);

            cv::Mat fr_warped = warp(frame, landmark);

            auto face_data = run_recognition(ie_fr, fr_warped, 0);

            if(ie_gender != 0)
            {
                auto age_gender_data = run_age_gender(ie_gender, fr_warped, i);
                face_data.age_idx = age_gender_data[0];
                face_data.gender_idx = age_gender_data[1];
            }

            float *feature_vector = face_data.feature_vector;
            float similarity_max = 0;
            int similarity_max_index = -1;
            for (size_t j = 0; j < gallary.size(); j++)
            {
                float similarity = cos_sim(gallary[j].feature_vector, feature_vector, 512);
                if (similarity > similarity_max && visited_indices[j] == false)
                {
                    similarity_max = similarity;
                    similarity_max_index = j;
                }
            }

            std::string id_str = "id ";
            if (similarity_max > frThreshold)
            {
                id_str += std::to_string(similarity_max_index);
                visited_indices[similarity_max_index] = true;
            }
            else
            {
                id_str += "?";
            }
            std::string caption_str = id_str + " (" + std::to_string(similarity_max) + ")";
            
            if(face_data.age_idx > 0) 
            {
                caption_str += ", age " + age_classes[face_data.age_idx] + ", (" + gender_classes[face_data.gender_idx] + ")";
            }
            int txtBaseline = 0;
            auto textSize = cv::getTextSize(caption_str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
            cv::rectangle(view, cv::Point(rect.x, rect.y - textSize.height -5), cv::Point(rect.x + textSize.width + 10, rect.y + 5), color, -1);
            cv::putText(view, caption_str, cv::Point(rect.x + 5, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        }

        cv::imshow("view", view);
        cv::imshow("gallary", make_gallary_view(gallary));

        int key = cv::waitKey(1);
        if (key > 0)
        {
            std::cout << "Key: " << key << std::endl;
        }
        switch (key)
        {
        case 27:
            running = false;
            break;
        }

        // double time_diff = GetTimestamp() - timestamp;
        //std::cout << "Latency: " << time_diff << "    FPS : " << 1 / time_diff << std::endl;
    }
}

void run_tracker_video_sync(dxrt::InferenceEngine *ie_fd, dxrt::InferenceEngine *ie_fl, dxrt::InferenceEngine *ie_fr, dxrt::InferenceEngine *ie_gender, SsdParam fdCfg, std::string dbPath, std::string videoFile, bool cameraInput, float frThreshold)
{
    auto fdDataInfo = ie_fd->outputs();
    Ssd detector = Ssd(fdCfg, fdDataInfo);
    Tracker tracker(0.25);

    int face_image_idx = 0;
    int selected = 0;
    std::vector<int> face_ids;
    std::vector<cv::Mat> face_images;
    for (int i = 0; i < 20; i++)
    {
        face_ids.emplace_back(-1);
        face_images.emplace_back(cv::Mat::zeros(FR_INPUT_HEIGHT, FR_INPUT_WIDTH, CV_8UC3));
    }

    auto gallary = get_gallary(dbPath, &detector, ie_fd, ie_fl, ie_fr);

    cv::VideoCapture cap;
    if (cameraInput)
    {
        cap.open(0, cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, CAMERA_FRAME_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, CAMERA_FRAME_HEIGHT);
    }
    else
    {
        cap.open(videoFile);
    }

    cv::namedWindow("view");
    cv::moveWindow("view", 0, 0);
    cv::namedWindow("gallary");
    cv::moveWindow("gallary", 0, 780);

    bool running = true;
    while (running)
    {
        // double timestamp = GetTimestamp();

        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        cv::Mat view = frame.clone();

        cv::Mat fd_input = preprocess(frame, cv::Size(FD_INPUT_WIDTH, FD_INPUT_HEIGHT));
        auto fd_tensors = ie_fd->Run(fd_input.data);
        auto fd_result = detector.PostProc(fd_tensors);

        std::vector<cv::Rect> D;
        for (size_t i = 0; i < fd_result.size(); i++)
        {
            auto detected = fd_result[i];
            cv::Rect rect = get_rect(detected.box, frame.cols, frame.rows);

            D.emplace_back(rect);

            // Visualization
            cv::Scalar color(0, 255, 0);
            cv::rectangle(view, rect, color, 2);
            cv::putText(view, detected.labelname, cv::Point(rect.x + 8, rect.y + 16), 0, 0.5, color, 1);
        }
        std::vector<bool> visited_indices;
        for (size_t i = 0; i < gallary.size(); i++)
        {
            visited_indices.emplace_back(false);
        }

        // Tracking
        tracker.run(D);

        face_image_idx = 0;
        for (size_t i = 0; i < tracker.T.size(); i++)
        {
            auto tracked = tracker.T[i];
            
            auto landmark = run_landmark(ie_fl, frame, tracker.T[i].box);

            visualize_landmark(view, landmark);

            cv::Mat fr_warped = warp(frame, landmark);
	
	        face_image_idx = i; 
            (face_image_idx) %= face_images.size();
            face_images[face_image_idx] = fr_warped;
            face_ids[face_image_idx] = tracker.T[i].id;

            auto face_data = run_recognition(ie_fr, face_images[face_image_idx], face_ids[face_image_idx]);

            if(ie_gender != 0)
            {
                auto age_gender_data = run_age_gender(ie_gender, face_images[face_image_idx], i);
                face_data.age_idx = age_gender_data[0];
                face_data.gender_idx = age_gender_data[1];
            }

            float *feature_vector = face_data.feature_vector;
            float similarity_max = 0;
            int similarity_max_index = -1;
            for (size_t j = 0; j < gallary.size(); j++)
            {
                float similarity = cos_sim(gallary[j].feature_vector, feature_vector, 512);
                if (similarity > similarity_max && visited_indices[j] == false)
                {
                    similarity_max = similarity;
                    similarity_max_index = j;
                }
            }
	    
            cv::Scalar color;
            if(face_ids[selected] != tracked.id) 
            {
                cv::Scalar temp(0, 255, 0);
                color = temp;
            }
            else 
            {
                cv::Scalar temp(0, 0, 255);
                color = temp;
            }

            std::string id_str = "id ";
            if (similarity_max > frThreshold)
            {
                id_str += std::to_string(similarity_max_index);
                visited_indices[similarity_max_index] = true;
            }
            else
            {
                id_str += "?";
            }
            std::string caption_str = id_str + " (" + std::to_string(similarity_max) + ")";
            
            if(face_data.age_idx > 0) 
            {
                caption_str += ", age " + age_classes[face_data.age_idx] + ", (" + gender_classes[face_data.gender_idx] + ")";
            }
            int txtBaseline = 0;
            auto textSize = cv::getTextSize(caption_str, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
            cv::rectangle(view, cv::Point(tracked.box.x, tracked.box.y - textSize.height -5), cv::Point(tracked.box.x + textSize.width + 10, tracked.box.y + 5), color, -1);
            cv::putText(view, caption_str, cv::Point(tracked.box.x + 5, tracked.box.y), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
	    }

        //std::cout << "detector result: " << fd_result.size() << " / tracker : " << tracker.T.size() << " / gallary: " << gallary.size() << std::endl;
        std::cout << "tracker.T.size(): " << tracker.T.size() << " / faceidx: " << face_image_idx << " / selected: " << selected << std::endl;

        cv::imshow("view", view);
        cv::imshow("gallary", make_gallary_view(gallary));

        int key = cv::waitKey(1);
        if (key > 0)
        {
            std::cout << "Key: " << key << std::endl;
        }
        switch (key)
        {
        case 27:
            running = false;
            break;
        case 'A':
        case 'a':
        {
            auto face_data = run_recognition(ie_fr, face_images[selected], 0);
            gallary.emplace_back(face_data);
            break;
        }
        case 'D':
        case 'd':
        {
            if (gallary.size() > 1)
            {
                gallary.pop_back();
            }
            break;
        }
        case 'S':
        case 's':
	    {
	        ++selected %= tracker.T.size();
            //if (tracker.T.size())
            //{
            //    tracker.T.erase(tracker.T.begin());
            //}
        }
        }

        // double time_diff = GetTimestamp() - timestamp;
        //std::cout << "Latency: " << time_diff << "    FPS : " << 1 / time_diff << std::endl;
    }
}

const char *usage =
    "Face Recognition Demo\n"
    "  -th, --threshold          Similarity Threshold\n"
    "  -m0, --fd_modelpath       face detection model dxnn file path\n"
    "  -m1, --lm_modelpath       face align model dxnn file path\n"
    "  -m2, --fr_modelpath       face recognition model dxnn file path\n"
    "  -m3, --age_sex_modelpath  age and sex estimation model dxnn file path\n"
    "  -p,  --dbpath             face database directory\n"
    "  -l,  --left               first image file to compare\n"
    "  -r,  --right              second image file to compare\n"
    "  -i,  --image              use image file input\n"
    "  -v,  --video              use video file input\n"
    "  -c,  --camera             use camera input\n"
    "  -t,  --tracker            use tracker function\n"
    "  -d,  --detect             include face detection (for dual image test)\n"
    "  -s,  --savemode           save result image (It is not supported in camera or video mode)\n"
    "  -h,  --help               show help\n";
void help()
{
    cout << usage << endl;
}

int main(int argc, char *argv[])
{
    int i = 1;
    std::string dbPath = "", videoFile = "", imgFile[2] = {""};
    std::string fd_modelPath = "", lm_modelPath = "", fr_modelPath = "", gender_modelPath = "" ;
    bool cameraInput = false, tracking = false;
    bool withFaceDetection = false, resultSaveMode = false, classifier_gender = false;
    float frThreshold = FR_THRESHOLD;
    if (argc == 1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }

    while (i < argc){
        std::string arg(argv[i++]);
        if(arg == "-th")
                                frThreshold = stof(argv[i++]);
        else if(arg == "-m0")
                                fd_modelPath = strdup(argv[i++]);
        else if(arg == "-m1")
                                lm_modelPath = strdup(argv[i++]);
        else if(arg == "-m2")
                                fr_modelPath = strdup(argv[i++]);
        else if(arg == "-m3")
                                gender_modelPath = strdup(argv[i++]), classifier_gender = true;
        else if(arg == "-p")
                                dbPath = strdup(argv[i++]);
        else if(arg == "-l")
                                imgFile[0] = strdup(argv[i++]);
        else if(arg == "-r")
                                imgFile[1] = strdup(argv[i++]);
        else if(arg == "-i")
                                imgFile[0] = strdup(argv[i++]);
        else if(arg == "-v")
                                videoFile = strdup(argv[i++]);
        else if(arg == "-c")
                                cameraInput = true;
        else if(arg == "-t")
                                tracking = true;
        else if(arg == "-d")
                                withFaceDetection = true;
        else if(arg == "-s")
                                resultSaveMode = true;
        else if(arg == "-h")
                                help(), exit(0);
        else
                                help(), exit(0);
    }

    if (fd_modelPath.empty() || lm_modelPath.empty() || fr_modelPath.empty()){
        std::cout<<"[NOTICE] model path is required to run Face ID"<<std::endl;
        help();
        exit(0);
    }
    g_resultSaveMode = resultSaveMode;
    dxrt::InferenceEngine ie_fd(fd_modelPath);
    dxrt::InferenceEngine ie_lm(lm_modelPath);
    dxrt::InferenceEngine ie_fr(fr_modelPath);
    std::shared_ptr<dxrt::InferenceEngine> ie_gender;
    if(classifier_gender)
        ie_gender = std::make_shared<dxrt::InferenceEngine>(gender_modelPath);
    SsdParam FDCfg = {
                .image_size = 512,
                .use_softmax = true,
                .score_threshold = 0.25,
                .iou_threshold = 0.25,
                .num_classes = 4,
                .start_class = 2,
                .class_names = {"BACKGROUND", "person", "no_mask", "mask"},
                .score_names = {"1275", "1305", "1335", "1365", "1392", "1416", "1440"},
                .loc_names = {"1290", "1320", "1350", "1380", "1404", "1428", "1452"},
                .priorBoxes = {
                    .num_layers = 7,
                    .min_scale = 0.2,
                    .max_scale = 0.95,
                    .center_variance = 0.1,
                    .size_variance = 0.2,
                    .dim = {
                        {64, 64, 6},
                        {32, 32, 6},
                        {16, 16, 6},
                        {8, 8, 6},
                        {4, 4, 6},
                        {2, 2, 4},
                        {1, 1, 4},
                    },
                    .data_file = "./sample/face_prior_boxes.bin" // temp
                },
            };

    if (!imgFile[0].empty() && imgFile[1].empty())
    {
        run_image(&ie_fd, &ie_lm, &ie_fr, &(*ie_gender), FDCfg, dbPath, imgFile[0], frThreshold);
    }
    else if (!imgFile[0].empty() && !imgFile[1].empty())
    {
        if (withFaceDetection)
        {
            run_image3(&ie_fd, &ie_lm, &ie_fr, &(*ie_gender), FDCfg, imgFile[0], imgFile[1]);
        }
        else
        {
            run_image2(&ie_fd, &ie_lm, &ie_fr, &(*ie_gender), imgFile[0], imgFile[1]);
        }
    }
    else if (!videoFile.empty() || cameraInput)
    {
        if (tracking)
        {
            run_tracker_video_sync(&ie_fd, &ie_lm, &ie_fr, &(*ie_gender), FDCfg, dbPath, videoFile, cameraInput, frThreshold);
        }
        else
        {
            run_video_sync(&ie_fd, &ie_lm, &ie_fr, &(*ie_gender), FDCfg, dbPath, videoFile, cameraInput, frThreshold);
        }
    }else{
        std::cout<<"[NOTICE] has no tasks"<<std::endl;
        help();
    }

    return 0;
}

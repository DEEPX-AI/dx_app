#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __linux
#include <sys/mman.h>
#include <unistd.h>
#elif _WIN32
#include <Windows.h>
#endif

#include <string.h>
#include <errno.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "display.h"

#include "od.h"

#include "utils/common_util.hpp"

using namespace std;
using namespace cv;
using namespace rapidjson;

#define DISPLAY_WINDOW_NAME "Object Detection"
#define EXPAND_WINDOW_NAME "Expand Window"
#define EXPAND_WINDOW 1

/* Tuning point */
#define INPUT_CAPTURE_PERIOD_MS 33

static bool g_full_screan = false;

/**
 * @brief AppConfig Definition
 *      application_type : 0 (single), 1 (multi)
 */
struct AppConfig
{
    int application_type;
    string model_path;
    string model_name;
    vector<pair<string, string>> video_sources;
    vector<int> pre_saved_frame_count;
    string display_label;

    int input_capture_period_ms;
    
    int board_width;
    int board_height;

    int is_show_fps;
    int is_fill_blank;
    int is_expand_mode;
};

// pre/post parameter table
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, yolox_s_512, yolov7_640, yolov7_512, yolox_s_640;
YoloParam yoloParams[] = {
    yolov5s_320,
    yolov5s_512,
    yolov5s_640,
    yolox_s_512,
    yolov7_640,
    yolov7_512,
    yolox_s_640
};

const char* usage =
"yolo demo\n"
"  -c, --config        use config json file for run application\n"
"                      e.g. sudo yolo_multi -c _multi_od_.json -a \n"
"      --window_size    FPS by average over the last {window_size} seconds (default: 60)\n"
"                      e.g. sudo yolo_multi -c _multi_od_.json --window_size 60\n"
"  -h, --help          show help\n"
;

void help()
{
    cout << usage << endl;
}

int ApplicationJsonParser(string configPath, AppConfig* dst)
{
    ifstream ifs(configPath);
    DXRT_ASSERT(ifs.is_open(), "can't open " + configPath );
    string json((istreambuf_iterator<char>(ifs)), (istreambuf_iterator<char>()));
    Document doc;
    doc.Parse(json.c_str());
    StringBuffer buffer;
    PrettyWriter<StringBuffer> writer(buffer);
    doc.Accept(writer);
    cout << buffer.GetString() << endl;
    if(doc.IsObject())
    {   
        DXRT_ASSERT(doc.HasMember("usage"), "ERR. usage argument not placed");
        DXRT_ASSERT(doc["usage"].IsString(), "ERR. usage argument must be str");
        dst->application_type = string(doc["usage"].GetString()) == "multi" ? 1 : 0;

        DXRT_ASSERT(doc.HasMember("model_path"), "ERR. model_path argument not placed");
        DXRT_ASSERT(doc["model_path"].IsString(), "ERR. model_path argument must be str");
        dst->model_path = doc["model_path"].GetString();

        DXRT_ASSERT(doc.HasMember("model_name"), "ERR. model_name argument not placed");
        DXRT_ASSERT(doc["model_name"].IsString(), "ERR. model_name argument must be str");
        dst->model_name = doc["model_name"].GetString();

        DXRT_ASSERT(doc.HasMember("display_config"), "ERR. display_config argument not placed");
        DXRT_ASSERT(doc["display_config"].IsObject(), "ERR. display_config must be json object");

        const Value& displayConfig = doc["display_config"];
        
        DXRT_ASSERT(displayConfig.HasMember("display_label"), "ERR. display_label argument not placed");
        DXRT_ASSERT(displayConfig["display_label"].IsString(), "ERR. display_label must be str");
        dst->display_label = displayConfig["display_label"].GetString();

        if(!displayConfig.HasMember("output_width")) 
        {
            g_full_screan = true;
#ifdef __linux__
            std::ifstream graphics_info_file("/sys/class/graphics/fb0/viertual_size");
            if(!graphics_info_file)
            {
                std::cout << "Failed to open framebuffer info, It will be set FHD size" << std::endl;
                dst->board_width = 1920;
                dst->board_height = 1080;
            }
            else 
            {
                int graphics_info_w, graphics_info_h;
                char comma;
                graphics_info_file >> graphics_info_w >> comma >> graphics_info_h;
                dst->board_width = graphics_info_w;
                dst->board_height = graphics_info_h;
            }
#elif _WIN32
            dst->board_width = GetSystemMetrics(SM_CXSCREEN);
            dst->board_height = GetSystemMetrics(SM_CYSCREEN);
#endif
        }
        else
        {
            DXRT_ASSERT(displayConfig["output_width"].IsInt(), "ERR. output_width must be integer");
            dst->board_width = displayConfig["output_width"].GetInt();

            DXRT_ASSERT(displayConfig.HasMember("output_height"), "ERR. output_height argument not placed");
            DXRT_ASSERT(displayConfig["output_height"].IsInt(), "ERR. output_height must be integer");
            dst->board_height = displayConfig["output_height"].GetInt();
        }
        if(displayConfig.HasMember("capture_period"))
        {
            DXRT_ASSERT(displayConfig["capture_period"].IsInt(), "ERR. capture_period must be integer");
            dst->input_capture_period_ms = displayConfig["capture_period"].GetInt();    
        }
        else
        {
            dst->input_capture_period_ms = INPUT_CAPTURE_PERIOD_MS;
        }

        if(displayConfig.HasMember("show_fps"))
        {
            DXRT_ASSERT(displayConfig["show_fps"].IsBool(), "ERR. show_fps must be boolean");
            dst->is_show_fps = displayConfig["show_fps"].GetBool();    
        }
        else
        {
            dst->is_show_fps = true;
        }
        
        if(displayConfig.HasMember("fill_blank"))
        {
            DXRT_ASSERT(displayConfig["fill_blank"].IsBool(), "ERR. fill_blank must be boolean");
            dst->is_fill_blank = displayConfig["fill_blank"].GetBool();    
        }
        else
        {
            dst->is_fill_blank = true;
        }

        if(displayConfig.HasMember("expand_mode"))
        {
            DXRT_ASSERT(displayConfig["expand_mode"].IsBool(), "ERR. expand_mode must be boolean");
            dst->is_expand_mode = displayConfig["expand_mode"].GetBool();    
        }
        else
        {
            dst->is_expand_mode = false;
        }
        
        DXRT_ASSERT(doc.HasMember("video_sources"), "ERR. video_sources argument not placed");
        DXRT_ASSERT(doc["video_sources"].IsArray(), "ERR. video_sources must be array");
        const Value& videoSources = doc["video_sources"];
        for(SizeType i = 0; i < videoSources.Size(); i++){
            const Value& videoSource = videoSources[i];
            pair<string, string> videoSourceInfo(pair<string, string>(videoSource[0].GetString(), videoSource[1].GetString()));
#if __riscv
            if(string(videoSource[1].GetString()) == "isp"){
                dst->video_sources.clear();
                dst->pre_saved_frame_count.clear();
                dst->video_sources.emplace_back(videoSourceInfo);
                dst->pre_saved_frame_count.emplace_back(-1);
                return 1;
            }
#endif
            if(string(videoSource[1].GetString()) == "offline")
            {
                if(videoSource.Size() == 2)
                {
                    dst->pre_saved_frame_count.emplace_back(0);
                }
                else if(videoSource.Size() == 3)
                {
                    dst->pre_saved_frame_count.emplace_back(videoSource[2].GetInt());
                }
            }else{
                dst->pre_saved_frame_count.emplace_back(-1);
            }
            dst->video_sources.emplace_back(videoSourceInfo);
        }
    }else{
        return -1;
    }
    return 1;
}

YoloParam getYoloParameter(string model_name){
    if(model_name == "yolov5s_320")
        return yolov5s_320;
    else if(model_name == "yolov5s_512")
        return yolov5s_512;
    else if(model_name == "yolov5s_640")
        return yolov5s_640;
    else if(model_name == "yolox_s_512")
        return yolox_s_512;
    else if(model_name == "yolov7_640")
        return yolov7_640;
    else if(model_name == "yolov7_512")
        return yolov7_512;
    else if(model_name == "yoloxs_640")
        return yolox_s_640;
    return yolov5s_512;
}
YoloParam yoloParam;

static int devideBoard(int numImages)
{
    return (int)ceil(sqrt(numImages));
}

int main(int argc, char *argv[])
{
DXRT_TRY_CATCH_BEGIN
    int arg_idx = 1;
    string configPath = "";
    float fps = 0.f; double frameCount = 0.0, window_size = 60;
    bool loggingVersion = false;
    char mainCaption[100];
    char subCaption[100];

    AppConfig appConfig;

    if(argc==1)
    {
        cout << "Error: no arguments." << endl;
        help();
        return -1;
    }

    while (arg_idx < argc) {
        std::string arg(argv[arg_idx++]);
        if (arg == "-c" || arg == "--config")
                        configPath = strdup(argv[arg_idx++]);
        else if (arg == "-t" || arg == "--test")
                        loggingVersion = true;
        else if (arg == "--window_size")
                        window_size = stoi(argv[arg_idx++]);
        else if (arg == "-h" || arg == "--help")
                        help(), exit(0);
        else
                        help(), exit(0);
    }
    if(configPath.empty()){
        std::cout << "error: no config json file arguments." << std::endl;
        help();
        return -1;
    }

    if(ApplicationJsonParser(configPath, &appConfig) < 0)
    {
        return -1;
    }

    LOG_VALUE(configPath);

    const int BOARD_WIDTH = appConfig.board_width;
    const int BOARD_HEIGHT = appConfig.board_height;

    int div = devideBoard(appConfig.video_sources.size());
    LOG_VALUE(div)
    int divWidth = BOARD_WIDTH / div;
    int divHeight = BOARD_HEIGHT / div;
    if(appConfig.is_expand_mode && appConfig.video_sources.size()!=33 && appConfig.video_sources.size()!=73 && appConfig.video_sources.size()!= 61 && appConfig.video_sources.size()!=41) {
        appConfig.is_expand_mode = false;
    }
    cv::Mat outFrame = cv::Mat(cv::Size(BOARD_WIDTH, BOARD_HEIGHT), CV_8UC3, cv::Scalar(0, 0, 0));
    
    auto ie = std::make_shared<dxrt::InferenceEngine>(appConfig.model_path);
    yoloParam = getYoloParameter(appConfig.model_name);
    Yolo yolo = Yolo(yoloParam);
    auto& profiler = dxrt::Profiler::GetInstance();
    vector<shared_ptr<ObjectDetection>> apps;
    unsigned int allFrameCount = 0;
    bool calcFps = false;
    
    if(appConfig.is_expand_mode)
    {
	    int position_index=0;
        int Window_scale = 2;
        if(appConfig.video_sources.size() == 41 || appConfig.video_sources.size() == 73){
            Window_scale = 3;
        }
        for(int i=0;i<(int)appConfig.video_sources.size(); i++)
        {
		if(appConfig.video_sources.size() == 33){
            if(i < 14){
				position_index = i;
			} else if(i < 18){
				position_index = i+2;
			} else if (i < 32) {
				position_index = i+4;
			} else {
				position_index = 14;
			}
        }else if(appConfig.video_sources.size() == 41){
            if(i < 16){
				position_index = i;
			} else if(i < 20){
				position_index = i+3;
			} else if (i < 24) {
				position_index = i+6;
			} else if (i < 40) {
				position_index = i+9;
			} else {
				position_index = 16;
			}
        }else if(appConfig.video_sources.size() == 73){
            if(i < 30){
				position_index = i;
			} else if(i < 36){
				position_index = i+3;
			} else if (i < 42) {
				position_index = i+6;
			} else if (i < 72) {
				position_index = i+9;
			} else {
				position_index = 30;
			}
        }else if(appConfig.video_sources.size() == 61){
		    if(i < 27){
                position_index = i;
            } else if(i < 33){
                    position_index = i+2;
            } else if (i < 60) {
                    position_index = i+4;
            } else {
                    position_index = 27;
            }
	}
            
	  
	   if( i == (int)appConfig.video_sources.size() - 1){ 
                apps.emplace_back(
                    make_shared<ObjectDetection>(
                        ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                        divWidth*Window_scale, divHeight*Window_scale, divWidth*(position_index%div), divHeight*(position_index/div), appConfig.pre_saved_frame_count[i]
                    ) 
                );
	   } else {
                apps.emplace_back(
                    make_shared<ObjectDetection>(
                        ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                        divWidth, divHeight, divWidth*(position_index%div), divHeight*(position_index/div), appConfig.pre_saved_frame_count[i]
                    ) 
                );
	    }

            cout << *apps.back() << endl;
        }
    }else
    {
        for(int i=0;i<(int)appConfig.video_sources.size(); i++)
        {
            apps.emplace_back(
                make_shared<ObjectDetection>(
                    ie, appConfig.video_sources[i], i, yoloParam.width, yoloParam.height,
                    divWidth, divHeight, divWidth*(i%div), divHeight*(i/div), appConfig.pre_saved_frame_count[i]
                ) 
            );
            cout << *apps.back() << endl;
        }
        if(appConfig.is_fill_blank && !appConfig.is_expand_mode)
        {
            for(int i=appConfig.video_sources.size();i<div*div;i++)
            {
                apps.emplace_back(
                    make_shared<ObjectDetection>(
                    ie, i, divWidth, divHeight, divWidth*(i%div), divHeight*(i/div)
                    )
                );
            }
        }
    }
    

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)> postProcCallBack = \
        [&](std::vector<shared_ptr<dxrt::Tensor>> outputs, void* arg)
        {
            ObjectDetection *app = (ObjectDetection *)arg;
            int64_t outputLength = outputs.front()->shape()[0];
            if(dxapp::common::compareVersions(DXRT_VERSION, "2.6.3"))
                outputLength = outputs.front()->shape()[1];
            app->PostProc((void*)app->GetOutputMemory(), outputLength);
            return 0;
        };
    ie->RegisterCallBack(postProcCallBack);

#if !__riscv
    namedWindow(DISPLAY_WINDOW_NAME, cv::WINDOW_NORMAL);
    setWindowProperty(DISPLAY_WINDOW_NAME, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    moveWindow(DISPLAY_WINDOW_NAME, 0, 0);
#endif

    for(auto &app:apps)
    {
        app->Run(appConfig.input_capture_period_ms);
    }
#ifdef __linux__
    usleep(500 * 1000);
#elif _WIN32
    Sleep(500);
#endif
    profiler.Add("spread");
    /* Debugging */
    std::vector<cv::Rect> dstPoint = std::vector<cv::Rect>(apps.size(), cv::Rect(0, 0, 0, 0));
    for(int i = 0; i < (int)apps.size(); i++)
    {
        dstPoint[i].x = apps[i]->Position().first;
        dstPoint[i].y = apps[i]->Position().second;
        dstPoint[i].width = apps[i]->Resolution().first;
        dstPoint[i].height = apps[i]->Resolution().second;
    }
    dxapp::common::StatusLog sl;
    sl.period = 500, sl.threadStatus.store(0);
    std::thread log_thread = std::thread(&dxapp::common::logThreadFunction, &sl);
    auto start = std::chrono::high_resolution_clock::now();
    long long duration = 0;
    long long passTime = 0;
    std::deque<unsigned int> callbackCount;
    int queueMaxSize = -1;

    while(true)
    {
        fps = 0.0f;
        frameCount = 0.1;
        float resultFps = 0.f;
        
        for(int i = 0; i < (int)apps.size(); i++)
        {
            cv::Mat roi = outFrame(dstPoint[i]);
            apps[i]->ResultFrame().copyTo(roi);
        }

        allFrameCount++;

        if(calcFps)
        {
            int checkSum = 0;
            for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
            {
                checkSum += apps[i]->GetPostProcessCount();
                apps[i]->SetZeroPostProcessCount();
            }
            if(queueMaxSize < callbackCount.size())
            {
                while(callbackCount.size() > queueMaxSize)
                {
                    callbackCount.pop_front();
                }
            }
            callbackCount.push_back(checkSum);
        }


        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        if(passTime != -1) passTime = duration;
        if(passTime > 1000 && calcFps == false)
        {
            calcFps = true;
            passTime = 0;
            start = std::chrono::high_resolution_clock::now();
            for(int i = 0; i < (int)appConfig.video_sources.size(); i++)
            {
                apps[i]->SetZeroPostProcessCount();
            }
        }
        if(passTime > 1000 * window_size){
            passTime = 0;
            queueMaxSize = callbackCount.size();
            start = std::chrono::high_resolution_clock::now();
        }

        for(int i = 0; i < (int)callbackCount.size(); i++)
        {
            frameCount += callbackCount[i];
        }

        if(calcFps)
        {
            if(queueMaxSize > 0)
            {
                resultFps = round(frameCount * 1000)/(1000 * window_size);
            }
            else
                resultFps = round(frameCount * 1000)/duration;

        }
        
        if(appConfig.is_show_fps)
        {
            cv::rectangle(outFrame, Point(0, 0), Point(500, 50), Scalar(0, 0, 255), cv::FILLED);
            snprintf(mainCaption, sizeof(mainCaption), " %dch Real-time Processing ",(int)apps.size());
            cv::putText(outFrame, mainCaption, Point(0, 35), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, LINE_AA);
            cv::rectangle(outFrame, Point(500, 0), Point(BOARD_WIDTH, 50), Scalar(0, 0, 0), cv::FILLED);
            if(calcFps)
                snprintf(subCaption, sizeof(subCaption), "        AI Model : %s        AI Performance : %.2f FPS ", appConfig.model_name.c_str(), resultFps);
            else
                snprintf(subCaption, sizeof(subCaption), "        AI Model : %s        AI Performance :       FPS ", appConfig.model_name.c_str());
            cv::putText(outFrame, subCaption, Point(500, 35), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2, LINE_AA);
        }
        sl.frameNumber = allFrameCount; 
        sl.runningTime = duration;
        if (loggingVersion)
            sl.threadStatus.store(2);
        else
            sl.threadStatus.store(1);
        
        sl.statusCheckCV.notify_one();
        
#if __riscv
        cout << "press 'q' and enter to exit. " << endl;
        int key = getchar();
#else
        cv::imshow(DISPLAY_WINDOW_NAME, outFrame);

        int key = cv::waitKey(1);
#endif
        if(key == 0x1B || key == 0x71) //'ESC'
        {
            sl.threadStatus.store(-1);
            for(auto &app:apps)
            {
                app->Stop();
            }
            log_thread.join();
            break;
        }
        else if(key == 0x74) // 't'
        {
            for(auto &app:apps)
            {
                app->Toggle();
            }
        }
        
    }
    profiler.Erase("spread");
#ifdef __linux__
    sleep(1);
#elif _WIN32
    Sleep(1000);
#endif
DXRT_TRY_CATCH_END    
    return 0;
}

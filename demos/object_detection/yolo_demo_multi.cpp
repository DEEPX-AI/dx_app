#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "display.h"

#include "od.h"

using namespace std;
using namespace cv;
using namespace rapidjson;

#define DISPLAY_WINDOW_NAME "Object Detection"
#define EXPAND_WINDOW_NAME "Expand Window"
#define EXPAND_WINDOW 1

/* Tuning point */
#define INPUT_CAPTURE_PERIOD_MS 33

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
extern YoloParam yolov5s_320, yolov5s_512, yolov5s_640, yolox_s_512, yolov7_640, yolov7_512, yolov4_608;
YoloParam yoloParams[] = {
    [0] = yolov5s_320,
    [1] = yolov5s_512,
    [2] = yolov5s_640,
    [3] = yolox_s_512,
    [4] = yolov7_640,
    [5] = yolov7_512,
    [6] = yolov4_608
};

/////////////////////////////////////////////////////////////////////////////////////////////////

const char* usage =
"yolo demo\n"
"  -c, --config    use config json file for run application\n"
"                  e.g. sudo yolo_multi -c _multi_od_.json -a \n"
"  -h, --help      show help\n"
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
        
        DXRT_ASSERT(displayConfig.HasMember("output_width"), "ERR. output_width argument not placed");
        DXRT_ASSERT(displayConfig["output_width"].IsInt(), "ERR. output_width must be integer");
        dst->board_width = displayConfig["output_width"].GetInt();
        
        DXRT_ASSERT(displayConfig.HasMember("output_height"), "ERR. output_height argument not placed");
        DXRT_ASSERT(displayConfig["output_height"].IsInt(), "ERR. output_height must be integer");
        dst->board_height = displayConfig["output_height"].GetInt();
        
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
    else if(model_name == "yolov4_608")
        return yolov4_608;
    return yolov5s_512;
}
YoloParam yoloParam;

static int devideBoard(int numImages)
{
    return (int)ceil(sqrt(numImages));
}

int main(int argc, char *argv[])
{
    int arg_idx = 1;
    string configPath = "";
    float fps = 0.f; double frameCount = 0.0;
    char mainCaption[100];

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
    int calcFpsPassTimes = 0;
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
            app->PostProc(outputs);
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
    usleep(500*1000);
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

    while(true)
    {
        fps = 0.0f;
        frameCount = 0;
        
        for(int i = 0; i < (int)apps.size(); i++)
        {
            cv::Mat roi = outFrame(dstPoint[i]);
            
            apps[i]->ResultFrame().copyTo(roi);
            if(i < (int)appConfig.video_sources.size() && calcFps)
            {
                fps += 1000000.0 / apps[i]->GetProcessingTime();
                frameCount++;
            }
        }
        calcFpsPassTimes++;
        if(calcFpsPassTimes > 30){
            calcFps = true;
        }

        float resultFps = round(fps * 100) / 100;
        if(appConfig.is_show_fps)
        {
            cv::rectangle(outFrame, Point(BOARD_WIDTH - 900, 0), Point(BOARD_WIDTH, 40), Scalar(120, 120, 120), cv::FILLED);
            snprintf(mainCaption, sizeof(mainCaption), "  Real time Processing... / %s, %.2f FPS ", appConfig.model_name.c_str(), resultFps);
            cv::putText(outFrame, mainCaption, Point(BOARD_WIDTH - 900, 25), 0, 1, cv::Scalar(0, 210, 210), 2, LINE_AA);
        }
        
#if __riscv
        cout << "press 'q' and enter to exit. " << endl;
        int key = getchar();
#else
        cv::imshow(DISPLAY_WINDOW_NAME, outFrame);

        int key = cv::waitKey(1);
#endif
        if(key == 0x1B || key == 0x71) //'ESC'
        {
            for(auto &app:apps)
            {
                app->Stop();
            }
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
    sleep(1);
    profiler.Show();
    return 0;
}

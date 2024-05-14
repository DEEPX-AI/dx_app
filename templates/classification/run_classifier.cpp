#include "classifier.hpp"

using namespace std;
using namespace rapidjson;

const char *usage =
    "classifier template\n"
    "  -c, --config       use config json file for run application\n"
    "  -h, --help         show help\n";

void help()
{
    std::cout << usage << std::endl;
}

int main(int argc, char *argv[])
{
    int arg_idx = 1;
    string configPath = "";

    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        std::terminate();
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
    if(configPath.empty())
    {
        std::cout << "error : no config json file arguments. " << std::endl;
        help();
        std::terminate();
    }

    dxapp::AppConfig appConfig(configPath);
    Classifier classifier(appConfig);

    uint8_t* input_tensor = new uint8_t[classifier.inputSize];

    for(auto &sources:appConfig.sourcesInfo){
        if(dxapp::common::pathValidation(sources.inputPath)){
            if(sources.inputType==AppInputType::IMAGE)
            {
                auto image = cv::imread(sources.inputPath, cv::IMREAD_COLOR);
                dxapp::classification::image_pre_processing(image, input_tensor, classifier.preConfig);
            }
            else if(sources.inputType==AppInputType::BINARY)
            {
                dxapp::common::readBinary(sources.inputPath, input_tensor, 1);
            }
            auto outputs = classifier.inferenceEngine->Run(input_tensor);
            if(outputs.size() == 0){
                continue;
            }
            auto result = dxapp::classification::postProcessing(outputs, classifier.postConfig);
            std::cout << "["<< sources.inputPath <<"] Top1 Result : class " << result._top1 <<" ("<< appConfig.classes.at(result._top1)<<")" <<std::endl;

            if(appConfig.appType==AppType::REALTIME)
            {
                if(sources.inputType==AppInputType::IMAGE)
                {
                    auto view = dxapp::classification::resultViewer(sources.inputPath, result);
                    cv::imshow("result", view);
                    if(cv::waitKey(0)=='q'){
                        break;
                    }
                }
                else
                {
                    std::cerr << sources.inputPath << " is not image file. " << std::endl;
                }
            }
            else if(appConfig.appType==AppType::OFFLINE)
            {
                char output_filename[100];
                auto view = dxapp::classification::resultViewer(sources.inputPath, result);
                auto filename = dxapp::common::getFileName(sources.inputPath);
                snprintf(output_filename, 100, "./%s-result.jpg", filename.c_str());
                cv::imwrite(output_filename, view);
                std::cout << "save file : " << output_filename << std::endl;
            }
        }
        else
        {
            std::cerr << sources.inputPath << " is invalid path. plz insert regular file. " << std::endl;
        }
    }

    return 0;
}
#include "detector.hpp"

using namespace std;
using namespace rapidjson;

const char *usage =
    "detector template\n"
    "  -c, --config       use config json file for run application\n"
    "  -h, --help         show help\n";

void help()
{
    std::cout << usage << std::endl;
}

int main(int argc, char *argv[])
{
    
    int arg_idx = 1;
    std::string configPath = "";
    char key;

    if (argc == 1)
    {
        std::cout << "Error: no arguments." << std::endl;
        help();
        exit(-1);
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
        exit(-1);
    }

    dxapp::AppConfig appConfig(configPath);
    Detector detector(appConfig);
    detector.makeThread();
    detector.startThread();
    while(true)
    {
        if(detector.status() != true)
        {
            detector.quitThread();
            break;
        }
#if __riscv
        key = getchar();
#else
        if(appConfig.appType == REALTIME)
        {
            cv::imshow("result", detector.totalView());
            key = (char)cv::waitKey(1);
        }
        else
        {
            if(detector.is_all_image && appConfig.appType == OFFLINE)
            {
                usleep(100000);
            }
            else
            {
                std::cout << "press 'q' to quit. " << std::endl;
                key = (char)getchar();
                std::cout << "pressed key " << key << std::endl;
            }
        }
#endif
        switch (key)
        {
        case 'q':
        case 0x1B:
            detector.quitThread();
            break;
        default:
            break;
        }
    }

    detector.joinThread();

    std::cout << " detector application End. " << std::endl;
    
    return 0;
}

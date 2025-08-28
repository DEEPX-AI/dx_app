#include "detector.hpp"

#include <cxxopts.hpp>

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
    DXRT_TRY_CATCH_BEGIN
    std::string configPath = "";
    char key = 0;

    cxxopts::Options options("run_detector", "detector template application usage ");
    options.add_options()
        ("c, config", "(* required) use config json file for run application", cxxopts::value<std::string>(configPath))
        ("h, help", "print usage")
    ;
    auto cmd = options.parse(argc, argv);
    if(cmd.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    if(configPath.empty())
    {
        std::cout << "error : no config json file arguments. " << std::endl;
        std::cout << "Use -h or --help for usage information." << std::endl;
        exit(0);
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
            key = static_cast<char>(cv::waitKey(1));
        }
        else
        {
            if(detector.is_all_image && appConfig.appType == OFFLINE)
            {
                std::this_thread::sleep_for(std::chrono::microseconds(100000));
            }
            else
            {
                std::cout << "press 'q' to quit. " << std::endl;
                key = static_cast<char>(getchar());
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
DXRT_TRY_CATCH_END    
    return 0;
}

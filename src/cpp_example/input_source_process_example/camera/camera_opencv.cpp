#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    std::string devicePath = "";
    int processCount = 0;
    bool appQuit = false;
    uint32_t input_w = 1920, input_h = 1080;

    std::mutex g_lock;
    std::queue<uint8_t> keyQueue;

    std::string app_name = "camera test";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "device", "camera device path", cxxopts::value<std::string>(devicePath)->default_value(""))(
        "width, input_width", "model input width size",
        cxxopts::value<uint32_t>(input_w)->default_value("640"))(
        "height, input_height", "model input height size",
        cxxopts::value<uint32_t>(input_h)->default_value("640"))("h, help", "print usage");
    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Validate required arguments
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    LOG_VALUE(modelPath)

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    std::function<int(void)> postprocessingThread = [&](void) -> int {
        while (keyQueue.size() < 1) {
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        std::cout << "[DXAPP] [INFO] post processing thread start" << std::endl;

        do {
            while (keyQueue.size() < 1 && !appQuit) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            std::unique_lock<std::mutex> lk(g_lock);
            auto outputs = ie.Wait(keyQueue.front());
            std::cout << "[DXAPP] [INFO] post processing result: " << outputs.size() << " items"
                      << std::endl;
            keyQueue.pop();
            processCount++;
        } while (!appQuit);

        std::cout << "[DXAPP] [INFO] post processing thread exit" << std::endl;

        return 0;
    };

    std::thread postThread(postprocessingThread);

    auto s = std::chrono::high_resolution_clock::now();

    int index = 0;
    cv::VideoCapture cap;
    if (devicePath.empty() == false)
        cap.open(devicePath, cv::CAP_V4L2);  // 특정 장치 경로
    else
        cap.open(0, cv::CAP_V4L2);  // 기본 카메라

    if (!cap.isOpened()) {
        std::cerr << "[ERROR] Could not open camera." << std::endl;
        return -1;
    }
    cv::Mat frame;
    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& inputTensor : inputTensors) {
        inputTensor = std::vector<uint8_t>(ie.GetInputSize());
    }
    s = std::chrono::high_resolution_clock::now();
    while (true) {
        cap >> frame;  // 프레임 읽기
        if (frame.empty()) {
            std::cerr << "[ERROR] Empty frame." << std::endl;
            break;
        }
        cv::Mat resized = cv::Mat(input_h, input_w, CV_8UC3, inputTensors[index].data());
        cv::resize(frame, resized, cv::Size(input_w, input_h), cv::INTER_LINEAR);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        keyQueue.push(ie.RunAsync(resized.data));
        index = (index + 1) % inputTensors.size();
        cv::imshow("Camera", frame);
        auto key = cv::waitKey(1);
        if (key == 27 || key == 'q') {  // ESC key
            break;
        }
    }
    cap.release();
    cv::destroyAllWindows();
    appQuit = true;

    postThread.join();

    auto e = std::chrono::high_resolution_clock::now();
    std::cout << "[DXAPP] [INFO] total time : "
              << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us"
              << std::endl;
    std::cout << "[DXAPP] [INFO] per frame time : "
              << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() / processCount
              << " us" << std::endl;
    std::cout << "[DXAPP] [INFO] fps : "
              << processCount /
                     (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0)
              << std::endl;
    DXRT_TRY_CATCH_END
    return 0;
}

// image_test.cpp
// Read JPG/PNG images from a folder or single file and run DXRT inference
// asynchronously

#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string modelPath;
    std::string inputPath;
    int loop = 100;  // number of times to loop through images
    uint32_t input_w = 224, input_h = 224;

    cxxopts::Options options("image_test", "Async image input example for DXRT");
    options.add_options()("m,model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i,input", "input image file", cxxopts::value<std::string>(inputPath))(
        "width", "model input width", cxxopts::value<uint32_t>(input_w)->default_value("640"))(
        "height", "model input height", cxxopts::value<uint32_t>(input_h)->default_value("640"))(
        "h,help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path required (-m)" << std::endl;
        return 1;
    }
    if (inputPath.empty()) {
        std::cerr << "[ERROR] Input path required (-i)" << std::endl;
        return 1;
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] model/runtime version mismatch" << std::endl;
        return -1;
    }

    // gather image files
    if (!dxapp::common::pathValidation(inputPath)) {
        std::cerr << "[ERROR] Input path not found: " << inputPath << std::endl;
        return 1;
    }

    input_w = ie.GetInputs()[0].shape()[1];
    input_h = ie.GetInputs()[0].shape()[2];
    std::cout << "[INFO] model input shape: " << input_w << "x" << input_h << std::endl;

    std::mutex g_lock;
    std::queue<int> keyQueue;
    int processCount = 0;
    bool appQuit = false;

    // postprocessing thread: waits on keys and calls Wait()
    std::thread postThread([&]() {
        while (keyQueue.size() < 1) std::this_thread::sleep_for(std::chrono::microseconds(10));
        while (!appQuit) {
            std::unique_lock<std::mutex> lk(g_lock);
            if (keyQueue.empty()) {
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }
            auto key = keyQueue.front();
            keyQueue.pop();
            lk.unlock();
            auto outputs = ie.Wait(key);
            // minimal demo: print result count
            std::cout << "[INFO] postprocessing got " << outputs.size() << " outputs" << std::endl;
            processCount++;
        }
    });

    // prepare ring buffers for input
    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& t : inputTensors) t = std::vector<uint8_t>(ie.GetInputSize());
    size_t idx = 0;

    auto s = std::chrono::high_resolution_clock::now();
    // main loop: iterate images and push async tasks
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    for (int j = 0; j < loop; ++j) {
        cv::Mat resized(input_h, input_w, CV_8UC3, inputTensors[idx].data());
        cv::resize(img, resized, cv::Size(input_w, input_h), cv::INTER_LINEAR);
        auto key = ie.RunAsync(resized.data);
        {
            std::lock_guard<std::mutex> lk(g_lock);
            keyQueue.push(key);
        }
        std::cout << "[INFO] pushed async task to queue key: " << std::dec << key << std::endl;
        idx = (idx + 1) % inputTensors.size();
    }

    // optionally wait until queue drained
    while (!keyQueue.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

// image_multi_model_test.cpp
// Read JPG/PNG images from a folder or single file and run DXRT inference
// asynchronously on multiple models with per-model FPS measurement

#include <dxrt/dxrt_api.h>

#include <atomic>
#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// Structure to hold per-model performance data
struct ModelPerformance {
    std::string modelPath;
    std::string modelName;
    std::unique_ptr<dxrt::InferenceEngine> engine;
    std::mutex taskQueueMutex;
    std::queue<int> taskQueue;
    std::atomic<int> processedCount{0};
    std::atomic<bool> shouldStop{false};
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    uint32_t input_w, input_h;

    // Preprocessing related members
    std::mutex preprocessQueueMutex;
    std::queue<std::vector<uint8_t>> preprocessedQueue;
    std::atomic<bool> preprocessFinished{false};

    ModelPerformance(const std::string &path) : modelPath(path) {
        // Extract model name from path
        size_t lastSlash = path.find_last_of("/\\");
        modelName = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
    }
};

int main(int argc, char *argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string modelPath;
    std::string inputPath;
    int loop = 100;  // number of times to loop through images

    cxxopts::Options options("image_multi_model_test",
                             "Async multi-model image inference example for DXRT");
    options.add_options()("m,model_paths",
                          "sample model files (ex. -m "
                          "model1.dxnn,model2.dxnn,model3.dxnn / required)",
                          cxxopts::value<std::string>(modelPath))(
        "i,input", "input image file", cxxopts::value<std::string>(inputPath))(
        "l,loop", "number of inference loops per model", cxxopts::value<int>(loop))(
        "delay_ms", "delay between inferences in milliseconds",
        cxxopts::value<int>()->default_value("0"))(
        "duration_ms", "total duration of inference in milliseconds",
        cxxopts::value<int>()->default_value("0"))("h,help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    int delay_ms = cmd["delay_ms"].as<int>();
    int duration_ms = cmd["duration_ms"].as<int>();

    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path(s) required (-m)" << std::endl;
        return 1;
    }

    auto modelPaths = dxapp::common::split(modelPath, ',');
    if (modelPaths.empty()) {
        std::cerr << "[ERROR] No valid model paths found" << std::endl;
        return 1;
    }

    if (inputPath.empty()) {
        std::cerr << "[ERROR] Input path required (-i)" << std::endl;
        return 1;
    }

    std::cout << "[INFO] Number of models to test: " << modelPaths.size() << std::endl;
    if (delay_ms > 0) {
        std::cout << "[INFO] Delay between inferences: " << delay_ms << " ms" << std::endl;
    }
    if (duration_ms > 0) {
        std::cout << "[INFO] Duration limit: " << duration_ms << " ms (loop count ignored)"
                  << std::endl;
        std::cout << "[INFO] Inference loops per model: unlimited" << std::endl;
    } else {
        std::cout << "[INFO] Inference loops per model: " << loop << std::endl;
    }

    // Create performance tracking structures for each model
    std::vector<std::unique_ptr<ModelPerformance>> modelPerfs;
    for (const auto &path : modelPaths) {
        std::unique_ptr<ModelPerformance> perf(new ModelPerformance(path));

        // Initialize inference engine for each model
        dxrt::InferenceOption io;
        try {
            perf->engine.reset(new dxrt::InferenceEngine(path, io));

            if (!dxapp::common::minversionforRTandCompiler(perf->engine.get())) {
                std::cerr << "[DXAPP] model/runtime version mismatch for " << path << std::endl;
                return -1;
            }

            perf->input_w = perf->engine->GetInputs()[0].shape()[1];
            perf->input_h = perf->engine->GetInputs()[0].shape()[1];
            std::cout << "[INFO] Model: " << perf->modelName << " input shape: " << perf->input_w
                      << "x" << perf->input_h << std::endl;

            modelPerfs.push_back(std::move(perf));
        } catch (const std::exception &e) {
            std::cerr << "[ERROR] Failed to load model " << path << ": " << e.what() << std::endl;
            return -1;
        }
    }

    // Validate input path
    if (!dxapp::common::pathValidation(inputPath)) {
        std::cerr << "[ERROR] Input path not found: " << inputPath << std::endl;
        return 1;
    }

    // Load input image
    cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "[ERROR] Failed to load image: " << inputPath << std::endl;
        return 1;
    }

    // Create processing threads for each model
    std::vector<std::thread> preprocessingThreads;
    std::vector<std::thread> processingThreads;
    std::vector<std::thread> postprocessingThreads;

    // Start preprocessing for each model
    for (auto &perf : modelPerfs) {
        // Preprocessing thread for this model
        preprocessingThreads.emplace_back([&perf, &img, loop, delay_ms, duration_ms]() {
            bool use_duration = (duration_ms > 0);
            int effective_loop = use_duration ? INT_MAX : loop;

            for (int j = 0; j < effective_loop && !perf->shouldStop.load(); ++j) {
                if (delay_ms > 0 && j > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
                }

                // shouldStop 체크 (duration timer에 의해 설정될 수 있음)
                if (perf->shouldStop.load()) {
                    break;
                }

                std::vector<uint8_t> preprocessedBuffer(perf->engine->GetInputSize());
                cv::Mat resized(perf->input_h, perf->input_w, CV_8UC3, preprocessedBuffer.data());
                cv::resize(img, resized, cv::Size(perf->input_w, perf->input_h), cv::INTER_LINEAR);
                {
                    std::lock_guard<std::mutex> lk(perf->preprocessQueueMutex);
                    perf->preprocessedQueue.push(std::move(preprocessedBuffer));
                }
            }
            perf->preprocessFinished = true;
        });
    }

    // Start processing for each model
    for (auto &perf : modelPerfs) {
        // Postprocessing thread for this model
        postprocessingThreads.emplace_back([&perf]() {
            while (!perf->shouldStop.load() || !perf->taskQueue.empty()) {
                // shouldStop이 설정되면 즉시 종료 (큐에 남은 작업 무시)
                if (perf->shouldStop.load()) {
                    break;
                }

                std::unique_lock<std::mutex> lk(perf->taskQueueMutex);
                if (perf->taskQueue.empty()) {
                    lk.unlock();
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                    continue;
                }
                auto key = perf->taskQueue.front();
                perf->taskQueue.pop();
                lk.unlock();

                try {
                    auto outputs = perf->engine->Wait(key);
                    perf->processedCount++;

                    if (perf->processedCount % 20 == 0) {
                        std::cout << "[INFO] " << perf->modelName << " processed "
                                  << perf->processedCount << " frames" << std::endl;
                    }
                } catch (const std::exception &e) {
                    std::cerr << "[ERROR] " << perf->modelName << " wait failed: " << e.what()
                              << std::endl;
                }
            }
        });

        // Processing thread for this model
        processingThreads.emplace_back([&perf]() {
            perf->startTime = std::chrono::high_resolution_clock::now();

            // Main inference loop for this model
            while ((!perf->preprocessFinished.load() || !perf->preprocessedQueue.empty()) &&
                   !perf->shouldStop.load()) {
                std::vector<uint8_t> preprocessedData;

                // Get preprocessed data from queue
                {
                    std::unique_lock<std::mutex> lk(perf->preprocessQueueMutex);
                    if (perf->preprocessedQueue.empty()) {
                        lk.unlock();
                        std::this_thread::sleep_for(std::chrono::microseconds(10));
                        continue;
                    }
                    preprocessedData = std::move(perf->preprocessedQueue.front());
                    perf->preprocessedQueue.pop();
                }

                try {
                    auto key = perf->engine->RunAsync(preprocessedData.data());
                    {
                        std::lock_guard<std::mutex> lk(perf->taskQueueMutex);
                        perf->taskQueue.push(key);
                    }
                } catch (const std::exception &e) {
                    std::cerr << "[ERROR] " << perf->modelName << " RunAsync failed: " << e.what()
                              << std::endl;
                    break;
                }
            }

            // Wait for all tasks to complete (but respect shouldStop)
            while (!perf->taskQueue.empty() && !perf->shouldStop.load()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }

            perf->endTime = std::chrono::high_resolution_clock::now();
            perf->shouldStop = true;
        });
    }

    // duration_ms 제한이 있으면 타이머 스레드 생성
    std::thread duration_thread;
    std::atomic<bool> duration_expired{false};
    if (duration_ms > 0) {
        duration_thread = std::thread([&modelPerfs, duration_ms, &duration_expired]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(duration_ms));
            duration_expired = true;
            std::cout << "[INFO] Duration limit reached (" << duration_ms
                      << "ms), stopping all models." << std::endl;
            for (auto &perf : modelPerfs) {
                perf->shouldStop = true;
                perf->preprocessFinished = true;
            }
        });
    }

    // Wait for all preprocessing threads to complete
    for (auto &thread : preprocessingThreads) {
        thread.join();
    }

    // Wait for all processing threads to complete
    for (auto &thread : processingThreads) {
        thread.join();
    }

    // Wait for all postprocessing threads to complete
    for (auto &thread : postprocessingThreads) {
        thread.join();
    }

    if (duration_thread.joinable()) duration_thread.join();

    double totalFPS = 0;
    for (const auto &perf : modelPerfs) {
        auto totalTime =
            std::chrono::duration_cast<std::chrono::milliseconds>(perf->endTime - perf->startTime)
                .count();
        double avgTime =
            perf->processedCount.load() > 0 ? totalTime / (double)perf->processedCount.load() : 0.0;
        double fps = perf->processedCount.load() > 0
                         ? perf->processedCount.load() / (totalTime / 1000.0)
                         : 0.0;
        totalFPS += fps;

        std::cout << "---------------------------------- RESULT "
                     "-----------------------"
                  << std::endl;
        std::cout << "\t Model: " << perf->modelName << std::endl;
        std::cout << "\t Total time: " << totalTime << " ms" << std::endl;
        std::cout << "\t average time per frame: " << avgTime << " ms" << std::endl;
        std::cout << std::fixed << std::setprecision(2) << "\t FPS: " << fps << " fps" << std::endl;
        std::cout << "-------------------------------------------------"
                     "----------------"
                  << std::endl;
    }

    std::cout << "Combined FPS: " << std::fixed << std::setprecision(2) << totalFPS << std::endl;

    DXRT_TRY_CATCH_END
    return 0;
}
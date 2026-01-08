#include <dxrt/inference_option.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>
#include <vector>

#include "object_detection_util.h"

// Declarations are provided by object_detection_util.h

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <model_path> <src1> [src2 src3 ...]" << std::endl;
        return 0;
    }
    std::string modelPath = argv[1];

    std::vector<std::string> sources;
    for (int i = 2; i < argc; ++i) sources.emplace_back(argv[i]);
    if (sources.empty()) {
        std::cerr << "No sources provided" << std::endl;
        return 1;
    }

    // Class names are handled by postprocess/draw in util; not needed here.
    dxrt::InferenceOption io;
    auto global_ie = std::make_shared<dxrt::InferenceEngine>(modelPath, io);
    std::vector<ChannelRunner> channels;
    const int cols = 2;
    const cv::Size winSize(640, 360);
    for (size_t i = 0; i < sources.size(); ++i) {
        int r = static_cast<int>(i) / cols;
        int c = static_cast<int>(i) % cols;
        cv::Point pos(50 + c * (winSize.width + 40), 50 + r * (winSize.height + 60));
        channels.emplace_back(makeChannel(global_ie, sources[i], winSize, pos));
        const std::string winName = sources[i];
        cv::namedWindow(winName, cv::WINDOW_NORMAL);
        cv::resizeWindow(winName, winSize.width, winSize.height);
        cv::moveWindow(winName, pos.x, pos.y);
    }
    std::cout << "channel size : " << channels.size() << std::endl;

    // 각 채널별 실행 상태 추적을 위한 원자적 플래그
    std::vector<std::atomic<bool>> channel_running(channels.size());
    std::atomic<bool> global_running{true};

    for (auto& running : channel_running) {
        running.store(true);
    }

    // 각 채널별로 별도 스레드 생성
    std::vector<std::thread> channel_feed_threads;
    std::vector<std::thread> channel_pp_threads;
    for (size_t i = 0; i < channels.size(); ++i) {
        channel_feed_threads.emplace_back([&channels, &channel_running, &global_running, i]() {
            while (global_running.load() && channel_running[i].load()) {
                bool success = processOnce(channels[i]);
                if (!success) {
                    std::cout << "Channel " << i << " finished or encountered error" << std::endl;
                    channel_running[i].store(false);
                    break;
                }
                // 스레드 간 적절한 CPU 양보
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });

        channel_pp_threads.emplace_back([&channels, &channel_running, &global_running, i]() {
            while (global_running.load() && channel_running[i].load()) {
                bool success = renderOnce(channels[i]);
                if (!success) {
                    // renderOnce가 실패하면 잠시 대기 후 계속
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                // 스레드 간 적절한 CPU 양보
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }

    // 메인 스레드에서 키 입력 처리
    while (global_running.load()) {
        int key = cv::waitKey(30);
        if (key == 27 || key == 'q') {
            std::cout << "Exit key pressed. Stopping all channels..." << std::endl;
            global_running.store(false);
            break;
        }

        // 모든 채널이 종료되었는지 확인
        bool any_running = false;
        for (const auto& running : channel_running) {
            if (running.load()) {
                any_running = true;
                break;
            }
        }

        if (!any_running) {
            std::cout << "All channels finished. Exiting..." << std::endl;
            global_running.store(false);
            break;
        }
    }

    // 모든 스레드 종료 대기
    std::cout << "Waiting for all threads to complete..." << std::endl;
    // for (auto& thread : channel_feed_threads) {
    for (size_t i = 0; i < channels.size(); ++i) {
        auto fThread = &channel_feed_threads[i];
        auto pThread = &channel_pp_threads[i];
        if (fThread->joinable()) {
            fThread->join();
        }
        if (pThread->joinable()) {
            pThread->join();
        }
    }

    std::cout << "All channels stopped." << std::endl;

    return 0;
}

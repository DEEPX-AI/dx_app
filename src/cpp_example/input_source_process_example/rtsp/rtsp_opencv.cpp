#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    std::string rtspURL = "";
    int processCount = 0;
    bool appQuit = false;
    uint32_t input_w = 1920, input_h = 1080;

    std::mutex g_lock;
    std::queue<uint8_t> keyQueue;

    std::string app_name = "rtsp test";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "r, rtsp_url", "RTSP stream URL", cxxopts::value<std::string>(rtspURL))(
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
    LOG_VALUE(rtspURL)

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

    if (!rtspURL.empty()) {
        int index = 0;
        cv::VideoCapture cap(rtspURL);

        // RTSP 최적화 설정
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // 버퍼 크기 최소화 (지연 시간 감소)
        cap.set(cv::CAP_PROP_FPS, 30);        // 예상 FPS 설정

// OpenCV 4.x에서 사용 가능한 설정들
#ifdef CV_CAP_PROP_RTSP_TRANSPORT
        cap.set(cv::CAP_PROP_RTSP_TRANSPORT, 1);  // TCP transport 사용
#endif

        if (!cap.isOpened()) {
            std::cerr << "[ERROR] Could not open RTSP stream." << std::endl;
            return -1;
        }

        // 스트림 정보 출력
        std::cout << "[INFO] RTSP Stream Info:" << std::endl;
        std::cout << "[INFO] Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
        std::cout << "[INFO] Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        std::cout << "[INFO] FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
        std::cout << "[INFO] Codec: " << cap.get(cv::CAP_PROP_FOURCC) << std::endl;
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
            keyQueue.push(ie.RunAsync(resized.data));
            index = (index + 1) % inputTensors.size();
            cv::imshow("RTSP Stream", frame);
            auto key = cv::waitKey(1);
            if (key == 27 || key == 'q') {  // ESC key
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();

    } else {
        std::cout << "[INFO] Running Without RTSP Input" << std::endl;
        std::vector<uint8_t> inputTensor(ie.GetInputSize(), 0);
        int loop = 100;
        do {
            keyQueue.push(ie.RunAsync(inputTensor.data()));
            std::cout << "[INFO] Pushed async task to queue : loop " << 100 - loop << std::endl;
        } while (--loop);
    }
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

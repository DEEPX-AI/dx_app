#include <dxrt/dxrt_api.h>

#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#define FRAME_BUFFERS 5

int getArgMax(float* output_data, int number_of_classes) {
    int max_idx = 0;
    for (int i = 0; i < number_of_classes; i++) {
        if (output_data[max_idx] < output_data[i]) {
            max_idx = i;
        }
    }
    return max_idx;
}

struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    int infer_completed = 0;
    std::chrono::high_resolution_clock::time_point infer_first_ts;
    std::chrono::high_resolution_clock::time_point infer_last_ts;
    bool first_inference = true;
    std::mutex metrics_mutex;
};

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec) {
    if (metrics.infer_completed == 0) return;

    double avg_read = metrics.sum_read / metrics.infer_completed;
    double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
    double avg_inf = metrics.sum_inference / metrics.infer_completed;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double inf_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;

    auto window = std::chrono::duration<double>(metrics.infer_last_ts - metrics.infer_first_ts).count();
    double infer_tp = (window > 0) ? metrics.infer_completed / window : 0.0;

    std::cout << "\n==================================================" << std::endl;
    std::cout << "               PERFORMANCE SUMMARY                " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << " Pipeline Step   Avg Latency     Throughput     " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Read" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_read << " ms     " << std::setw(6)
              << std::setprecision(1) << read_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Preprocess" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_pre << " ms     " << std::setw(6)
              << std::setprecision(1) << pre_fps << " FPS" << std::endl;
    std::cout << " " << std::left << std::setw(15) << "Inference" << std::right << std::setw(8)
              << std::fixed << std::setprecision(2) << avg_inf << " ms     " << std::setw(6)
              << std::setprecision(1) << inf_fps << " FPS" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " * Async inference throughput : " << std::fixed << std::setprecision(1)
              << infer_tp << " FPS" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;

    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

struct ClassificationArgs {
    std::vector<std::vector<uint8_t>>* inputBuffers;
    std::vector<std::vector<uint8_t>>* outputBuffers;
    std::mutex output_process_lk;
    ProfilingMetrics* metrics = nullptr;
    std::vector<std::chrono::high_resolution_clock::time_point>* infer_start_times = nullptr;
    int process_count = 0;
    int frame_idx = 0;
};

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "", imgFile = "";
    int loopTest = 1;
    uint32_t input_w = 224, input_h = 224, class_size = 1000;
    std::string app_name = "efficientnetB0 async example";
    cxxopts::Options options("classification_async",
                             app_name + " application usage ");
    options.add_options()("m, model_path",
                          "classification model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i, image_path", "input image file path(jpg, png, jpeg ..., required)",
        cxxopts::value<std::string>(imgFile))(
        "width, input_width", "model input width size",
        cxxopts::value<uint32_t>(input_w)->default_value("224"))(
        "height, intpu_height", "model input height size",
        cxxopts::value<uint32_t>(input_h)->default_value("224"))(
        "class, class_size", "number of classes",
        cxxopts::value<uint32_t>(class_size)->default_value("1000"))(
        "l, loop", "Number of inference iterations to run",
        cxxopts::value<int>(loopTest)->default_value("30"))(
        "h, help", "print usage");
    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    // Validate required arguments
    if (modelPath.empty()) {
        std::cerr
            << "[ERROR] Model path is required. Use -m or --model_path option."
            << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if (imgFile.empty()) {
        std::cerr
            << "[ERROR] Image path is required. Use -i or --image_path option."
            << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }

    LOG_VALUE(modelPath)
    LOG_VALUE(imgFile)
    LOG_VALUE(loopTest)

    std::queue<int> frame_index_queue;
    std::mutex frame_queue_lk;

    dxrt::InferenceOption io;
    io.useORT = false;
    dxrt::InferenceEngine ie(modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    std::vector<std::vector<uint8_t>> inputBuffers(FRAME_BUFFERS);
    std::vector<std::vector<uint8_t>> outputBuffers(FRAME_BUFFERS);
    for (int i = 0; i < FRAME_BUFFERS; i++) {
        inputBuffers[i] = std::vector<uint8_t>(ie.GetInputSize());
        outputBuffers[i] = std::vector<uint8_t>(ie.GetOutputSize());
    }
    ClassificationArgs args;
    args.inputBuffers = &inputBuffers;
    args.outputBuffers = &outputBuffers;
    ProfilingMetrics metrics;
    std::vector<std::chrono::high_resolution_clock::time_point> infer_start_times(
        FRAME_BUFFERS);
    args.metrics = &metrics;
    args.infer_start_times = &infer_start_times;

    int frame_count = 0;

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)>
        cls_postProcCallBack =
            [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void* arg) {
                auto arguments = (ClassificationArgs*)arg;
                {
                    std::unique_lock<std::mutex> lk(
                        arguments->output_process_lk);

                    int current_index = -1;
                    {
                        std::lock_guard<std::mutex> qlk(frame_queue_lk);
                        if (!frame_index_queue.empty()) {
                            current_index = frame_index_queue.front();
                            frame_index_queue.pop();
                        }
                    }

                    auto now = std::chrono::high_resolution_clock::now();
                    if (current_index >= 0 && arguments->infer_start_times) {
                        auto inf_ms = std::chrono::duration<double, std::milli>(
                                          now - arguments->infer_start_times->at(current_index))
                                          .count();
                        if (arguments->metrics) {
                            std::lock_guard<std::mutex> mlk(arguments->metrics->metrics_mutex);
                            arguments->metrics->sum_inference += inf_ms;
                            if (arguments->metrics->first_inference) {
                                arguments->metrics->infer_first_ts = now;
                                arguments->metrics->first_inference = false;
                            }
                            arguments->metrics->infer_last_ts = now;
                            arguments->metrics->infer_completed++;
                        }
                    }

                    if (outputs.front()->type() == dxrt::DataType::FLOAT) {
                        auto result = getArgMax((float*)outputs.front()->data(),
                                                class_size);
                        if (arguments->process_count + 1 == loopTest) {
                            std::cout << "Top1 Result : class " << result
                                      << std::endl;
                        }
                    } else {
                        auto result = *(uint16_t*)outputs.front()->data();
                        if (arguments->process_count + 1 == loopTest) {
                            std::cout << "Top1 Result : class " << result
                                      << std::endl;
                        }
                    }
                    arguments->process_count = arguments->process_count + 1;
                    arguments->frame_idx = arguments->frame_idx + 1;
                }
                return 0;
            };

    ie.RegisterCallback(cls_postProcCallBack);

    if (!imgFile.empty()) {
        int index = 0;

        cv::Mat original_image, resized_image, input;
        original_image = cv::imread(imgFile, cv::IMREAD_COLOR);
        if (original_image.empty()) {
            std::cerr << "[ERROR] Failed to read image: " << imgFile << std::endl;
            return -1;
        }

        auto total_start = std::chrono::high_resolution_clock::now();
        printf("Waiting for inference to complete...\n");

        do {
            index = frame_count % FRAME_BUFFERS;
            while (true) {
                bool available = false;
                {
                    std::lock_guard<std::mutex> qlk(frame_queue_lk);
                    available = frame_index_queue.size() < FRAME_BUFFERS;
                }
                if (available) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            auto& inputBuf = args.inputBuffers->at(index);
            auto& outputBuf = args.outputBuffers->at(index);

            auto pre_start = std::chrono::high_resolution_clock::now();
            cv::resize(original_image, resized_image,
                       cv::Size(input_w, input_h));
            cv::cvtColor(resized_image, input, cv::COLOR_BGR2RGB);
            memcpy(&inputBuf[0], &input.data[0], ie.GetInputSize());
            auto pre_end = std::chrono::high_resolution_clock::now();

            {
                std::lock_guard<std::mutex> mlk(metrics.metrics_mutex);
                metrics.sum_preprocess +=
                    std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
            }

            args.infer_start_times->at(index) = std::chrono::high_resolution_clock::now();

            std::ignore =
                ie.RunAsync(inputBuf.data(), &args, (void*)outputBuf.data());
            {
                std::lock_guard<std::mutex> qlk(frame_queue_lk);
                frame_index_queue.push(index);
            }
            frame_count++;

        } while (--loopTest);

        while (true) {
            if (args.process_count == frame_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration<double, std::milli>(total_end - total_start).count() / 1000.0;

        print_performance_summary(metrics, args.process_count, total_time);

        return 0;
    }
    DXRT_TRY_CATCH_END
    return 0;
}

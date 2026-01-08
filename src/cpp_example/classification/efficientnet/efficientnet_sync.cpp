#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
};

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec) {
    if (total_frames == 0) return;

    double avg_read = metrics.sum_read / total_frames;
    double avg_pre = metrics.sum_preprocess / total_frames;
    double avg_inf = metrics.sum_inference / total_frames;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double inf_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;

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

int getArgMax(float* output_data, int number_of_classes) {
    int max_idx = 0;
    for (int i = 0; i < number_of_classes; i++) {
        if (output_data[max_idx] < output_data[i]) {
            max_idx = i;
        }
    }
    return max_idx;
}

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "", imgFile = "";
    int loopTest = 1, processCount = 0;
    uint32_t input_w = 224, input_h = 224, class_size = 1000;

    std::string app_name = "efficientnetB0 example";
    cxxopts::Options options(app_name, app_name + " application usage ");
    options.add_options()("m, model_path",
                          "classification model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i, image_path", "input image file path(jpg, png, jpeg ..., required)",
        cxxopts::value<std::string>(imgFile))(
        "width, input_width", "model input width size",
        cxxopts::value<uint32_t>(input_w)->default_value("224"))(
        "height, input_height", "model input height size",
        cxxopts::value<uint32_t>(input_h)->default_value("224"))(
        "class, class_size", "number of classes",
        cxxopts::value<uint32_t>(class_size)->default_value("1000"))(
        "l, loop", "Number of inference iterations to run",
        cxxopts::value<int>(loopTest)->default_value("1"))("h, help",
                                                           "print usage");
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
    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the "
                     "version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    if (!imgFile.empty()) {
        auto total_start = std::chrono::high_resolution_clock::now();
        ProfilingMetrics metrics;

        do {
            auto read_start = std::chrono::high_resolution_clock::now();
            cv::Mat image = cv::imread(imgFile, cv::IMREAD_COLOR);
            auto read_end = std::chrono::high_resolution_clock::now();
            if (image.empty()) {
                std::cerr << "[ERROR] Failed to read image: " << imgFile << std::endl;
                return -1;
            }

            cv::Mat resized, input;
            auto pre_start = std::chrono::high_resolution_clock::now();
            cv::resize(image, resized, cv::Size(input_w, input_h));
            cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);
            auto pre_end = std::chrono::high_resolution_clock::now();

            auto inf_start = std::chrono::high_resolution_clock::now();
            auto outputs = ie.Run(input.data);
            auto inf_end = std::chrono::high_resolution_clock::now();

            metrics.sum_read += std::chrono::duration<double, std::milli>(read_end - read_start).count();
            metrics.sum_preprocess +=
                std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
            metrics.sum_inference +=
                std::chrono::duration<double, std::milli>(inf_end - inf_start).count();

            processCount++;
            if (!outputs.empty()) {
                if (ie.GetOutputs().front().type() == dxrt::DataType::FLOAT) {
                    auto result = getArgMax((float*)outputs.front()->data(), class_size);
                    std::cout << "Top1 Result : class " << result << std::endl;
                } else {
                    auto result = *(uint16_t*)outputs.front()->data();
                    std::cout << "Top1 Result : class " << result << std::endl;
                }
            }
        } while (--loopTest);

        auto total_end = std::chrono::high_resolution_clock::now();
        double total_time =
            std::chrono::duration<double, std::milli>(total_end - total_start).count() / 1000.0;

        print_performance_summary(metrics, processCount, total_time);
    }
    DXRT_TRY_CATCH_END
    return 0;
}

#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

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
        auto s = std::chrono::high_resolution_clock::now();

        do {
            cv::Mat image, resized, input;
            image = cv::imread(imgFile, cv::IMREAD_COLOR);
            cv::resize(image, resized, cv::Size(input_w, input_h));
            cv::cvtColor(resized, input, cv::COLOR_BGR2RGB);

            auto outputs = ie.Run(input.data);
            processCount++;
            if (!outputs.empty()) {
                if (ie.GetOutputs().front().type() == dxrt::DataType::FLOAT) {
                    auto result =
                        getArgMax((float*)outputs.front()->data(), class_size);
                    std::cout << "Top1 Result : class " << result << std::endl;
                } else {
                    auto result = *(uint16_t*)outputs.front()->data();
                    std::cout << "Top1 Result : class " << result << std::endl;
                }
            }
        } while (--loopTest);

        auto e = std::chrono::high_resolution_clock::now();
        std::cout << "[DXAPP] [INFO] total time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e -
                                                                           s)
                         .count()
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] per frame time : "
                  << std::chrono::duration_cast<std::chrono::microseconds>(e -
                                                                           s)
                             .count() /
                         processCount
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : "
                  << processCount /
                         (std::chrono::duration_cast<std::chrono::milliseconds>(
                              e - s)
                              .count() /
                          1000.0)
                  << std::endl;
    }
    DXRT_TRY_CATCH_END
    return 0;
}

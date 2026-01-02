#include <dxrt/dxrt_api.h>

#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
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

struct ClassificationArgs {
    std::vector<std::vector<uint8_t>>* inputBuffers;
    std::vector<std::vector<uint8_t>>* outputBuffers;
    std::mutex output_process_lk;
    int process_count = 0;
    int frame_idx = 0;
};

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "", imgFile = "";
    int loopTest = 1;
    uint32_t input_w = 224, input_h = 224, class_size = 1000;
    bool fps_only = false;
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
        "fps_only", "will not show inference result visualize, only show fps",
        cxxopts::value<bool>(fps_only)->default_value("false"))("h, help",
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

    std::queue<int> frame_index_queue;

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

    int frame_count = 0;

    std::function<int(std::vector<std::shared_ptr<dxrt::Tensor>>, void*)>
        cls_postProcCallBack =
            [&](std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void* arg) {
                auto arguments = (ClassificationArgs*)arg;
                {
                    std::unique_lock<std::mutex> lk(
                        arguments->output_process_lk);

                    if (outputs.front()->type() == dxrt::DataType::FLOAT) {
                        auto result = getArgMax((float*)outputs.front()->data(),
                                                class_size);
                        if (!fps_only) {
                            std::cout << "index : " << arguments->frame_idx
                                      << " Top1 Result : class " << result
                                      << std::endl;
                        }
                    } else {
                        auto result = *(uint16_t*)outputs.front()->data();
                        if (!fps_only) {
                            std::cout << "index : " << arguments->frame_idx
                                      << " Top1 Result : class " << result
                                      << std::endl;
                        }
                    }
                    arguments->process_count = arguments->process_count + 1;
                    arguments->frame_idx = arguments->frame_idx + 1;

                    frame_index_queue.pop();
                }
                return 0;
            };

    ie.RegisterCallback(cls_postProcCallBack);

    if (!imgFile.empty()) {
        int index = 0;

        cv::Mat original_image, resized_image, input;
        original_image = cv::imread(imgFile, cv::IMREAD_COLOR);

        auto s = std::chrono::high_resolution_clock::now();
        if (fps_only) {
            printf("Waiting for inference to complete...\n");
        }

        do {
            index = frame_count % FRAME_BUFFERS;
            while (frame_index_queue.size() >= FRAME_BUFFERS) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
            auto& inputBuf = args.inputBuffers->at(index);
            auto& outputBuf = args.outputBuffers->at(index);

            cv::resize(original_image, resized_image,
                       cv::Size(input_w, input_h));
            cv::cvtColor(resized_image, input, cv::COLOR_BGR2RGB);
            memcpy(&inputBuf[0], &input.data[0], ie.GetInputSize());

            std::ignore =
                ie.RunAsync(inputBuf.data(), &args, (void*)outputBuf.data());
            frame_index_queue.push(index);
            frame_count++;

        } while (--loopTest);

        while (true) {
            if (args.process_count == frame_count) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }

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
                         args.process_count
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : "
                  << args.process_count /
                         (std::chrono::duration_cast<std::chrono::milliseconds>(
                              e - s)
                              .count() /
                          1000.0)
                  << std::endl;

        return 0;
    }
    DXRT_TRY_CATCH_END
    return 0;
}

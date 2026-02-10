#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>



// --- Structures ---

// Profiling metrics structure
struct ProfilingMetrics {
    double sum_read = 0.0;
    double sum_preprocess = 0.0;
    double sum_inference = 0.0;
    int infer_completed = 0;
};

// Command line arguments structure
struct CommandLineArgs {
    std::string modelPath;
    std::string imageFilePath;
    uint32_t input_w = 224;
    uint32_t input_h = 224;
    uint32_t class_size = 1000;
    int loopTest = 30;
};

// --- Helper functions ---

int getArgMax(const float* output_data, int number_of_classes) {
    int max_idx = 0;
    for (int i = 0; i < number_of_classes; i++) {
        if (output_data[max_idx] < output_data[i]) {
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * @brief Check if file extension indicates an image file.
 */
bool is_image_file(const std::string& extension) {
    return extension == ".jpg" || extension == ".jpeg" || 
           extension == ".png" || extension == ".bmp";
}

/**
 * @brief Load image files from a directory.
 */
std::vector<std::string> load_image_files_from_directory(const std::string& dirPath) {
    std::vector<std::string> imageFiles;
    
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!fs::is_regular_file(entry.path())) {
            continue;
        }
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (is_image_file(ext)) {
            imageFiles.push_back(entry.path().string());
        }
    }
    std::sort(imageFiles.begin(), imageFiles.end());
    return imageFiles;
}

/**
 * @brief Process image path (file or directory) and return list of image files.
 */
std::pair<std::vector<std::string>, int> process_image_path(
    const std::string& imageFilePath, int loopTest) {
    std::vector<std::string> imageFiles;
    
    if (fs::is_directory(imageFilePath)) {
        imageFiles = load_image_files_from_directory(imageFilePath);
        if (imageFiles.empty()) {
            std::cerr << "[ERROR] No image files found in directory: " << imageFilePath << std::endl;
            exit(1);
        }
        if (loopTest == -1) {
            loopTest = static_cast<int>(imageFiles.size());
        }
    } else if (fs::is_regular_file(imageFilePath)) {
        imageFiles.push_back(imageFilePath);
        if (loopTest == -1) {
            loopTest = 1;
        }
    } else {
        std::cerr << "[ERROR] Invalid image path: " << imageFilePath << std::endl;
        exit(1);
    }
    
    return {imageFiles, loopTest};
}

// --- Performance summary ---

void print_performance_summary(const ProfilingMetrics& metrics, int total_frames,
                               double total_time_sec) {
    if (metrics.infer_completed == 0) return;

    double avg_read = metrics.sum_read / metrics.infer_completed;
    double avg_pre = metrics.sum_preprocess / metrics.infer_completed;
    double avg_inf = metrics.sum_inference / metrics.infer_completed;

    double read_fps = avg_read > 0 ? 1000.0 / avg_read : 0.0;
    double pre_fps = avg_pre > 0 ? 1000.0 / avg_pre : 0.0;
    double infer_fps = avg_inf > 0 ? 1000.0 / avg_inf : 0.0;

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
              << std::setprecision(1) << infer_fps << " FPS" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Frames"
              << " :    " << total_frames << std::endl;
    std::cout << " " << std::left << std::setw(19) << "Total Time"
              << " :    " << std::fixed << std::setprecision(1) << total_time_sec << " s"
              << std::endl;

    double overall_fps = (total_time_sec > 0) ? total_frames / total_time_sec : 0.0;
    std::cout << " " << std::left << std::setw(19) << "Overall FPS"
              << " :   " << std::fixed << std::setprecision(1) << overall_fps << " FPS"
              << std::endl;
    std::cout << "==================================================" << std::endl;
}

// --- Command line parsing and validation ---

// Parse and validate command line arguments
CommandLineArgs parse_command_line(int argc, char* argv[]) {
    CommandLineArgs args;
    std::string app_name = "yolov26cls sync example";
    cxxopts::Options options("classification_sync", app_name + " application usage ");
    options.add_options()("m, model_path", "classification model file (.dxnn, required)",
                          cxxopts::value<std::string>(args.modelPath))(
        "i, image_path", "input image file path or directory (jpg, png, jpeg ..., required)",
        cxxopts::value<std::string>(args.imageFilePath))(
        "width, input_width", "model input width size",
        cxxopts::value<uint32_t>(args.input_w)->default_value("224"))(
        "height, input_height", "model input height size",
        cxxopts::value<uint32_t>(args.input_h)->default_value("224"))(
        "class, class_size", "number of classes",
        cxxopts::value<uint32_t>(args.class_size)->default_value("1000"))(
        "l, loop", "Number of inference iterations to run",
        cxxopts::value<int>(args.loopTest)->default_value("30"))("h, help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }
    return args;
}

// Validate command line arguments
void validate_arguments(const CommandLineArgs& args) {
    if (args.modelPath.empty()) {
        std::cerr << "[ERROR] Model path is required. Use -m or --model_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
    if (args.imageFilePath.empty()) {
        std::cerr << "[ERROR] Image path is required. Use -i or --image_path option." << std::endl;
        std::cerr << "Use -h or --help for usage information." << std::endl;
        exit(1);
    }
}

// --- Classification result helpers ---

int get_classification_result(const std::vector<std::shared_ptr<dxrt::Tensor>>& outputs,
                              uint32_t class_size) {
    if (outputs.front()->type() == dxrt::DataType::FLOAT) {
        return getArgMax(static_cast<float*>(outputs.front()->data()), class_size);
    }
    return static_cast<int>(*static_cast<uint16_t*>(outputs.front()->data()));
}

std::string extract_filename(const std::string& filepath) {
    size_t pos = filepath.find_last_of("/\\");
    if (pos != std::string::npos) {
        return filepath.substr(pos + 1);
    }
    return filepath;
}

void print_classification_result(int frame_idx, int loopTest, const std::string& filepath,
                                 int result_class) {
    std::string filename = extract_filename(filepath);
    std::cout << "[" << (frame_idx + 1) << "/" << loopTest << "] "
              << filename << " -> Top1 class: " << result_class << std::endl;
}

// --- Process single frame ---

void process_single_frame(const cv::Mat& original_image, cv::Mat& resized_image, cv::Mat& input,
                          uint32_t input_w, uint32_t input_h, dxrt::InferenceEngine& ie,
                          ProfilingMetrics& metrics, int frame_idx, int loopTest,
                          const std::string& imagePath, uint32_t class_size,
                          double t_read) {
    auto pre_start = std::chrono::high_resolution_clock::now();
    cv::resize(original_image, resized_image, cv::Size(input_w, input_h));
    cv::cvtColor(resized_image, input, cv::COLOR_BGR2RGB);
    auto pre_end = std::chrono::high_resolution_clock::now();

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto outputs = ie.Run(input.data);
    auto infer_end = std::chrono::high_resolution_clock::now();

    metrics.sum_read += t_read;
    metrics.sum_preprocess +=
        std::chrono::duration<double, std::milli>(pre_end - pre_start).count();
    metrics.sum_inference +=
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
    metrics.infer_completed++;

    if (!outputs.empty()) {
        int result_class = get_classification_result(outputs, class_size);
        print_classification_result(frame_idx, loopTest, imagePath, result_class);
    }
}

// --- Process image frames ---

// Process image frames loop
void process_image_frames(const std::vector<std::string>& imageFiles, int loopTest,
                          uint32_t input_w, uint32_t input_h, dxrt::InferenceEngine& ie,
                          ProfilingMetrics& metrics, uint32_t class_size) {
    cv::Mat resized_image;
    cv::Mat input;

    for (int i = 0; i < loopTest; ++i) {
        std::string currentImagePath = imageFiles[i % imageFiles.size()];

        auto tr0 = std::chrono::high_resolution_clock::now();
        cv::Mat original_image = cv::imread(currentImagePath, cv::IMREAD_COLOR);
        auto tr1 = std::chrono::high_resolution_clock::now();
        double t_read = std::chrono::duration<double, std::milli>(tr1 - tr0).count();

        if (original_image.empty()) {
            std::cerr << "[ERROR] Failed to read image: " << currentImagePath << std::endl;
            continue;
        }

        process_single_frame(original_image, resized_image, input, input_w, input_h, ie,
                             metrics, i, loopTest, currentImagePath, class_size, t_read);
    }
}

// --- Main function ---
int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    // Parse and validate command line arguments
    CommandLineArgs args = parse_command_line(argc, argv);
    validate_arguments(args);

    LOG_VALUE(args.modelPath)
    LOG_VALUE(args.imageFilePath)
    LOG_VALUE(args.loopTest)

    // Initialize inference engine
    dxrt::InferenceOption io;
    io.useORT = false;
    dxrt::InferenceEngine ie(args.modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] [ER] The version of the compiled model is not "
                     "compatible with the version of the runtime. Please compile the model again."
                  << std::endl;
        return -1;
    }

    // Handle image file or directory
    auto result = process_image_path(args.imageFilePath, args.loopTest);
    std::vector<std::string> imageFiles = result.first;
    int loopTest = result.second;

    ProfilingMetrics metrics;

    auto total_start = std::chrono::high_resolution_clock::now();
    printf("Waiting for inference to complete...\n");

    // Process image frames
    process_image_frames(imageFiles, loopTest, args.input_w, args.input_h, ie,
                         metrics, args.class_size);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time =
        std::chrono::duration<double, std::milli>(total_end - total_start).count() / 1000.0;

    print_performance_summary(metrics, metrics.infer_completed, total_time);

    DXRT_TRY_CATCH_END
    return 0;
}

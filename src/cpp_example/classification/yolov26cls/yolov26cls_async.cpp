#include <dxrt/dxrt_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <experimental/filesystem>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::experimental::filesystem;


// --- Constants ---
constexpr size_t FRAME_BUFFERS = 5;

// --- Structures ---

// Profiling metrics structure
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

struct ClassificationArgs {
    std::vector<std::vector<uint8_t>>* inputBuffers;
    std::vector<std::vector<uint8_t>>* outputBuffers;
    std::mutex output_process_lk;
    ProfilingMetrics* metrics = nullptr;
    std::vector<std::chrono::high_resolution_clock::time_point>* infer_start_times = nullptr;
    std::vector<std::string>* imageFiles = nullptr;
    int process_count = 0;
    int frame_idx = 0;
    int loopTest = 0;
    uint32_t class_size = 1000;
    std::queue<int>* frame_index_queue = nullptr;
    std::mutex* frame_queue_lk = nullptr;
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

void update_metrics_with_lock(ProfilingMetrics* metrics, double inf_ms,
                              const std::chrono::high_resolution_clock::time_point& now) {
    std::lock_guard<std::mutex> mlk(metrics->metrics_mutex);
    metrics->sum_inference += inf_ms;
    if (metrics->first_inference) {
        metrics->infer_first_ts = now;
        metrics->first_inference = false;
    }
    metrics->infer_last_ts = now;
    metrics->infer_completed++;
}

void update_inference_metrics(ClassificationArgs* arguments, int current_index) {
    auto now = std::chrono::high_resolution_clock::now();
    if (current_index >= 0 && arguments->infer_start_times) {
        auto inf_ms = std::chrono::duration<double, std::milli>(
                          now - arguments->infer_start_times->at(current_index))
                          .count();
        if (arguments->metrics) {
            update_metrics_with_lock(arguments->metrics, inf_ms, now);
        }
    }
}

void print_classification_result(ClassificationArgs* arguments, int result_class) {
    if (arguments->imageFiles && !arguments->imageFiles->empty()) {
        int file_idx = arguments->process_count % arguments->imageFiles->size();
        std::string filename = (*arguments->imageFiles)[file_idx];
        size_t pos = filename.find_last_of("/\\");
        if (pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        std::cout << "[" << (arguments->process_count + 1) << "/" << arguments->loopTest << "] "
                  << filename << " -> Top1 class: " << result_class << std::endl;
    } else {
        std::cout << "[" << (arguments->process_count + 1) << "/" << arguments->loopTest << "] "
                  << "Top1 class: " << result_class << std::endl;
    }
}

// Helper function to pop frame index from queue
int pop_frame_index(std::queue<int>* frame_index_queue, std::mutex* frame_queue_lk) {
    std::lock_guard<std::mutex> qlk(*frame_queue_lk);
    if (!frame_index_queue->empty()) {
        int index = frame_index_queue->front();
        frame_index_queue->pop();
        return index;
    }
    return -1;
}

int classification_callback(std::vector<std::shared_ptr<dxrt::Tensor>> outputs, void* arg) {
    auto arguments = static_cast<ClassificationArgs*>(arg);
    std::unique_lock<std::mutex> lk(arguments->output_process_lk);

    int current_index = pop_frame_index(arguments->frame_index_queue, arguments->frame_queue_lk);

    update_inference_metrics(arguments, current_index);

    int result_class = -1;
    if (outputs.front()->type() == dxrt::DataType::FLOAT) {
        result_class = getArgMax(static_cast<float*>(outputs.front()->data()), arguments->class_size);
    } else {
        result_class = *static_cast<uint16_t*>(outputs.front()->data());
    }

    print_classification_result(arguments, result_class);

    arguments->process_count = arguments->process_count + 1;
    arguments->frame_idx = arguments->frame_idx + 1;
    return 0;
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
              << std::setprecision(1) << infer_tp << " FPS*" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << " * Actual throughput via async inference" << std::endl;
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
    std::string app_name = "efficientnetB0 async example";
    cxxopts::Options options("classification_async", app_name + " application usage ");
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

// Helper function to check if buffer is available
bool is_buffer_available(const std::queue<int>& frame_index_queue, std::mutex& frame_queue_lk) {
    std::lock_guard<std::mutex> qlk(frame_queue_lk);
    return frame_index_queue.size() < FRAME_BUFFERS;
}

// Helper function to wait for available buffer
void wait_for_buffer(const std::queue<int>& frame_index_queue, std::mutex& frame_queue_lk) {
    while (!is_buffer_available(frame_index_queue, frame_queue_lk)) {
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

// Helper function to update read and preprocess metrics
void update_read_preprocess_metrics(ProfilingMetrics& metrics, double t_read, double t_preprocess) {
    std::lock_guard<std::mutex> mlk(metrics.metrics_mutex);
    metrics.sum_read += t_read;
    metrics.sum_preprocess += t_preprocess;
}

// Helper function to push frame index to queue
void push_frame_index(std::queue<int>& frame_index_queue, std::mutex& frame_queue_lk, int index) {
    std::lock_guard<std::mutex> qlk(frame_queue_lk);
    frame_index_queue.push(index);
}

// --- Process image frames ---

// Process image frames loop
void process_image_frames(
    const std::vector<std::string>& imageFiles, int loopTest, ClassificationArgs& cls_args,
    std::queue<int>& frame_index_queue, std::mutex& frame_queue_lk,
    uint32_t input_w, uint32_t input_h, dxrt::InferenceEngine& ie,
    ProfilingMetrics& metrics, int& frame_count) {

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

        int index = frame_count % FRAME_BUFFERS;

        // Wait for available buffer
        wait_for_buffer(frame_index_queue, frame_queue_lk);

        auto& inputBuf = cls_args.inputBuffers->at(index);
        auto& outputBuf = cls_args.outputBuffers->at(index);

        auto pre_start = std::chrono::high_resolution_clock::now();
        cv::resize(original_image, resized_image, cv::Size(input_w, input_h));
        cv::cvtColor(resized_image, input, cv::COLOR_BGR2RGB);
        memcpy(&inputBuf[0], &input.data[0], ie.GetInputSize());
        auto pre_end = std::chrono::high_resolution_clock::now();
        double t_preprocess = std::chrono::duration<double, std::milli>(pre_end - pre_start).count();

        update_read_preprocess_metrics(metrics, t_read, t_preprocess);

        cls_args.infer_start_times->at(index) = std::chrono::high_resolution_clock::now();

        std::ignore = ie.RunAsync(inputBuf.data(), &cls_args, (void*)outputBuf.data());
        push_frame_index(frame_index_queue, frame_queue_lk, index);
        frame_count++;
    }
}

// Wait for processing to complete
void wait_for_completion(const ClassificationArgs& cls_args, int frame_count) {
    while (true) {
        if (cls_args.process_count == frame_count) break;
        std::this_thread::sleep_for(std::chrono::microseconds(10));
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

    // Initialize buffers
    std::vector<std::vector<uint8_t>> inputBuffers(FRAME_BUFFERS);
    std::vector<std::vector<uint8_t>> outputBuffers(FRAME_BUFFERS);
    for (size_t i = 0; i < FRAME_BUFFERS; i++) {
        inputBuffers[i] = std::vector<uint8_t>(ie.GetInputSize());
        outputBuffers[i] = std::vector<uint8_t>(ie.GetOutputSize());
    }

    // Initialize classification args
    ClassificationArgs cls_args;
    cls_args.inputBuffers = &inputBuffers;
    cls_args.outputBuffers = &outputBuffers;
    ProfilingMetrics metrics;
    std::vector<std::chrono::high_resolution_clock::time_point> infer_start_times(FRAME_BUFFERS);
    cls_args.metrics = &metrics;
    cls_args.infer_start_times = &infer_start_times;

    std::queue<int> frame_index_queue;
    std::mutex frame_queue_lk;
    int frame_count = 0;
    int loopTest = args.loopTest;

    // Set additional classification args
    cls_args.loopTest = loopTest;
    cls_args.class_size = args.class_size;
    cls_args.frame_index_queue = &frame_index_queue;
    cls_args.frame_queue_lk = &frame_queue_lk;

    // Register callback
    ie.RegisterCallback(classification_callback);

    // Handle image file or directory
    auto result = process_image_path(args.imageFilePath, loopTest);
    std::vector<std::string> imageFiles = result.first;
    loopTest = result.second;
    cls_args.imageFiles = &imageFiles;

    auto total_start = std::chrono::high_resolution_clock::now();
    printf("Waiting for inference to complete...\n");

    // Process image frames
    process_image_frames(imageFiles, loopTest, cls_args, frame_index_queue, frame_queue_lk,
                         args.input_w, args.input_h, ie, metrics, frame_count);

    // Wait for completion
    wait_for_completion(cls_args, frame_count);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time =
        std::chrono::duration<double, std::milli>(total_end - total_start).count() / 1000.0;

    print_performance_summary(metrics, cls_args.process_count, total_time);

    DXRT_TRY_CATCH_END
    return 0;
}

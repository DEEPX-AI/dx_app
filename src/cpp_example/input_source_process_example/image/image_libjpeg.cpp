// image_libjpeg_test.cpp
// Decode JPEG using libjpeg-turbo (turbojpeg), resize (nearest-neighbor)
// without OpenCV, and feed to DXRT asynchronously.

#include <dxrt/dxrt_api.h>
#include <turbojpeg.h>

#include <common_util.hpp>
#include <cxxopts.hpp>

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
// for C++17
#include <filesystem>
namespace fs = std::filesystem;
#else
// for C++11
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

static bool readFileToVector(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    out.resize((size_t)sz);
    ifs.read(reinterpret_cast<char*>(out.data()), sz);
    return true;
}

// simple nearest-neighbor resize from src RGB to dst BGR (direct write)
static void resize_nearest_rgb_to_bgr(const uint8_t* src, int srcW, int srcH, int dstW, int dstH,
                                      uint8_t* dst) {
    for (int y = 0; y < dstH; ++y) {
        int sy = (int)((y * (long long)srcH) / dstH);
        if (sy >= srcH) sy = srcH - 1;
        for (int x = 0; x < dstW; ++x) {
            int sx = (int)((x * (long long)srcW) / dstW);
            if (sx >= srcW) sx = srcW - 1;
            const uint8_t* s = src + (sy * srcW + sx) * 3;
            uint8_t* d = dst + (y * dstW + x) * 3;
            // write BGR order (swap RGB -> BGR)
            d[0] = s[2];
            d[1] = s[1];
            d[2] = s[0];
        }
    }
}

static bool decode_jpeg_to_bgr_resized(const std::vector<unsigned char>& jpegBuf, uint8_t* dst,
                                       int dstW, int dstH) {
    tjhandle handle = tjInitDecompress();
    if (!handle) return false;
    int width = 0, height = 0, jpegSubsamp = 0, jpegColorspace = 0;
    if (tjDecompressHeader3(handle, jpegBuf.data(), (unsigned int)jpegBuf.size(), &width, &height,
                            &jpegSubsamp, &jpegColorspace) != 0) {
        tjDestroy(handle);
        return false;
    }
    // decompress to RGB full size
    std::vector<uint8_t> rgb(width * height * 3);
    if (tjDecompress2(handle, jpegBuf.data(), (unsigned int)jpegBuf.size(), rgb.data(), width, 0,
                      height, TJPF_RGB, TJFLAG_FASTDCT) != 0) {
        tjDestroy(handle);
        return false;
    }
    // resize nearest and convert to BGR
    resize_nearest_rgb_to_bgr(rgb.data(), width, height, dstW, dstH, dst);
    tjDestroy(handle);
    return true;
}

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN

    std::string modelPath;
    std::string inputPath;
    int loop = 100;
    uint32_t input_w = 224, input_h = 224;

    cxxopts::Options options("image_libjpeg_test", "libjpeg-turbo image loader example for DXRT");
    options.add_options()("m,model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "i,input", "input image file or directory", cxxopts::value<std::string>(inputPath))(
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

    // gather jpeg files
    std::vector<std::string> images;
    fs::path p(inputPath);
    if (fs::exists(p) && fs::is_directory(p)) {
        for (auto& entry : fs::directory_iterator(p)) {
            if (!entry.is_regular_file()) continue;
            std::string ext = entry.path().extension().string();
            for (auto& c : ext) c = std::tolower(c);
            if (ext == ".jpg" || ext == ".jpeg") images.push_back(entry.path().string());
        }
        std::sort(images.begin(), images.end());
    } else if (fs::exists(p) && fs::is_regular_file(p)) {
        images.push_back(p.string());
    } else {
        std::cerr << "[ERROR] Input path not found: " << inputPath << std::endl;
        return 1;
    }
    if (images.empty()) {
        std::cerr << "[ERROR] No JPEG images found in " << inputPath << std::endl;
        return 1;
    }

    std::mutex g_lock;
    std::queue<int> keyQueue;
    int processCount = 0;
    bool appQuit = false;

    std::thread postThread([&]() {
        while (keyQueue.size() < 1) std::this_thread::sleep_for(std::chrono::microseconds(10));
        while (!appQuit) {
            std::unique_lock<std::mutex> lk(g_lock);
            if (keyQueue.empty()) {
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }
            int key = keyQueue.front();
            keyQueue.pop();
            lk.unlock();
            auto outputs = ie.Wait(key);
            std::cout << "[INFO] postprocessing got " << outputs.size() << " outputs" << std::endl;
            processCount++;
        }
    });

    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& t : inputTensors) t = std::vector<uint8_t>(ie.GetInputSize());
    size_t idx = 0;
    if (images.size() > 1) {
        loop = 1;
    }
    auto s = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < images.size(); ++i) {
        for (int j = 0; j < loop; ++j) {
            std::vector<unsigned char> filedata;
            if (!readFileToVector(images[i], filedata)) {
                std::cerr << "[WARN] failed to read " << images[i] << std::endl;
                continue;
            }
            // decode + resize into input buffer
            if (!decode_jpeg_to_bgr_resized(filedata, inputTensors[idx].data(), (int)input_w,
                                            (int)input_h)) {
                std::cerr << "[WARN] decode failed for " << images[i] << std::endl;
                continue;
            }
            int key = ie.RunAsync(inputTensors[idx].data());
            {
                std::lock_guard<std::mutex> lk(g_lock);
                keyQueue.push(key);
            }
            std::cout << "[INFO] pushed async task key=" << key << " file=" << images[i]
                      << std::endl;
            idx = (idx + 1) % inputTensors.size();
        }
    }

    while (!keyQueue.empty()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    appQuit = true;
    postThread.join();
    auto e = std::chrono::high_resolution_clock::now();
    std::cout << "[DXAPP] [INFO] total time : "
              << std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() << " us"
              << std::endl;
    if (processCount > 0) {
        std::cout << "[DXAPP] [INFO] per frame time : "
                  << (std::chrono::duration_cast<std::chrono::microseconds>(e - s).count() /
                      processCount)
                  << " us" << std::endl;
        std::cout << "[DXAPP] [INFO] fps : "
                  << (processCount /
                      (std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() /
                       1000.0))
                  << std::endl;
    }

    DXRT_TRY_CATCH_END
    return 0;
}

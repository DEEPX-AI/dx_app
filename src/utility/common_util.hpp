#pragma once

#include <stdio.h>
#include <sys/stat.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
// for C++17
#include <filesystem>
namespace fs = std::filesystem;
#else
// for C++11
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if __linux__
#include <dirent.h>
#include <unistd.h>
#elif _WIN32
#include <Windows.h>
#include <io.h>
#define fsync _commit
#define popen _popen
#define pclose _pclose
#define fileno _fileno
#endif

#include <dxrt/device_info_status.h>
#include <dxrt/dxrt_api.h>

// Color codes for console output
#define RED "\033[1;31m"
#define YELLOW "\033[1;33m"
#define GREEN "\033[1;32m"
#define RESET "\033[0m"

// Platform-specific setup file paths
#if _WIN32
#define SETUP_FILE_PATH "setup.bat"
#else
#define SETUP_FILE_PATH "setup.sh --force"
#endif

// Exception handling macros
#ifndef DXRT_EXCEPTION_UTIL
#define DXRT_EXCEPTION_UTIL
#define DXRT_TRY_CATCH_BEGIN try {
#define DXRT_TRY_CATCH_END                                                                       \
    }                                                                                            \
    catch (const dxrt::Exception &e) {                                                           \
        std::cerr << RED << e.what() << " error-code=" << e.code() << RESET << std::endl;        \
        fs::path dx_app_dir(fs::canonical(PROJECT_ROOT_DIR));                                    \
        fs::path setup_script = dx_app_dir / SETUP_FILE_PATH;                                    \
        std::cerr << "dx_app_dir: " << dx_app_dir.string() << std::endl;                         \
        if (e.code() == 257) {                                                                   \
            if (dx_app_dir != fs::canonical(fs::current_path())) {                               \
                std::cerr << GREEN << "[HINT] The current directory is '"                        \
                          << fs::current_path().string() << "'. Please move to '"                \
                          << dx_app_dir.string() << "' before running the application." << RESET \
                          << std::endl;                                                          \
            } else {                                                                             \
                std::cerr << GREEN << "[HINT] Please run '" << setup_script.string()             \
                          << "' to set up the model and input video files "                      \
                             "before running the application again."                             \
                          << RESET << std::endl;                                                 \
                std::cerr << YELLOW                                                              \
                          << "Would you like to run the setup script now? (y/n): " << RESET;     \
                std::string user_input;                                                          \
                std::cin >> user_input;                                                          \
                if (user_input == "y" || user_input == "Y") {                                    \
                    int ret = system(setup_script.string().c_str());                             \
                    if (ret != 0) {                                                              \
                        std::cerr << RED                                                         \
                                  << "Failed to run setup script. Please "                       \
                                     "check permissions or script content."                      \
                                  << RESET << std::endl;                                         \
                    }                                                                            \
                }                                                                                \
            }                                                                                    \
        }                                                                                        \
        return -1;                                                                               \
    }                                                                                            \
    catch (const std::exception &e) {                                                            \
        std::cerr << e.what() << std::endl;                                                      \
        return -1;                                                                               \
    }
#endif  // DXRT_EXCEPTION_UTIL

namespace dxapp {
namespace common {
// Status logging structure
struct StatusLog {
    unsigned int frameNumber;
    int64_t runningTime;  // milliseconds
    time_t period;
    std::condition_variable statusCheckCV;
    std::atomic<int> threadStatus;
};

// Function declarations
template <typename T>
void readBinary(const std::string &filePath, T *dst);

void dumpBinary(void *ptr, int dump_size, const std::string &file_name);

void readCSV(const std::string &filePath, float *dst, int size);

int divideBoard(int numImages);

bool pathValidation(const std::string &path);

std::string getAllPath(const std::string &path);

bool dirValidation(const std::string &path);

std::vector<std::string> loadFilesFromDir(const std::string &path);

bool checkOrtLinking();

std::string getLocalTimeString();

void logThreadFunction(void *args);

bool isVersionGreaterOrEqual(const std::string &v1, const std::string &v2);

bool minversionforRTandCompiler(dxrt::InferenceEngine *ie);

std::vector<std::string> split(const std::string &str, char delimiter);

// Class name utilities for object detection models
std::string get_coco_class_name(const int class_id);

std::string get_dota_class_name(const int class_id);

}  // namespace common
}  // namespace dxapp

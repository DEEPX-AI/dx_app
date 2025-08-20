#pragma once
#include <sys/stat.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <iomanip>

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
#include <dxrt/dxrt_api.h>
#include <dxrt/device_info_status.h>

#define RED     "\033[1;31m"
#define YELLOW  "\033[1;33m"
#define GREEN   "\033[1;32m"
#define RESET   "\033[0m"

#if _WIN32
#define SETUP_FILE_PATH "setup.bat"
#else
#define SETUP_FILE_PATH "setup.sh --force"
#endif

namespace dxapp
{
namespace common
{
    
#ifndef DXRT_EXCEPTION_UTIL
#define DXRT_EXCEPTION_UTIL
#define DXRT_TRY_CATCH_BEGIN try {
#define DXRT_TRY_CATCH_END } \
catch (const dxrt::Exception& e) { \
    std::cerr << RED << e.what() << " error-code=" << e.code() << RESET << std::endl; \
    fs::path dx_app_dir(fs::canonical(PROJECT_ROOT_DIR)); \
    fs::path setup_script = dx_app_dir / SETUP_FILE_PATH; \
    std::cerr << "dx_app_dir: " << dx_app_dir.string() << std::endl; \
    if (e.code() == 257 ) { \
        if (dx_app_dir != fs::canonical(fs::current_path())) { \
            std::cerr << GREEN << "[HINT] The current directory is '" << fs::current_path().string() << "'. Please move to '" << dx_app_dir.string() << "' before running the application." << RESET << std::endl; \
        } else { \
            std::cerr << GREEN << "[HINT] Please run '" << setup_script.string() << "' to set up the model and input video files before running the application again." << RESET << std::endl; \
            std::cerr << YELLOW << "Would you like to run the setup script now? (y/n): " << RESET; \
            std::string user_input; \
            std::cin >> user_input; \
            if (user_input == "y" || user_input == "Y") { \
                int ret = system(setup_script.string().c_str()); \
                if (ret != 0) { \
                    std::cerr << RED << "Failed to run setup script. Please check permissions or script content." << RESET << std::endl; \
                } \
            } \
        } \
    } \
    return -1; \
} \
catch (const std::exception& e) \
{ \
    std::cerr << e.what() << std::endl; \
    return -1; \
}
#endif // DXRT_EXCEPTION_UTIL

    struct StatusLog
    {
        unsigned int frameNumber;
        int64_t runningTime; //milliseconds
        time_t period;
        std::condition_variable statusCheckCV;
        std::atomic<int> threadStatus;
    };

    inline int get_align_factor(int length, int based)
    {
        return (length | (-based)) == (-based)? 0 : -(length | (-based));
    }    
    
    template<typename T>
    inline void readBinary(const std::string &filePath, T* dst)
    {
        std::FILE *fp = NULL;
        fp = std::fopen(filePath.c_str(), "rb");
        std::fseek(fp, 0, SEEK_END);
        auto size = ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        int read_size = fread((void*)dst, sizeof(T), size, fp);
        if(read_size != size)
            std::cout << "file size mismatch("<<read_size<<", " << size << "), fail to read file " << filePath << std::endl;
        fclose(fp);
    }

    inline void dumpBinary(void *ptr, int dump_size, const std::string &file_name)
    {
        std::ofstream outfile(file_name, std::ios::binary);
        if(!outfile.is_open())
        {
            std::cout << "can not open file " << file_name << std::endl;
            std::terminate();
        }
        outfile.write((char*)ptr, dump_size);
        outfile.close();
    }
    
    inline void readCSV(const std::string &filePath, float* dst, int size)
    {
        std::ifstream file;
        std::string value;
        file.open(filePath);
        for(int i=0; i<size; i++){
            std::getline(file, value);
            dst[i] = std::stof(value);
        }
        file.close();
    }
    
    inline int divideBoard(int numImages)
    {
        int ret_Div = 1;
        if(numImages < 2) ret_Div = 1;
        else if(numImages < 5) ret_Div = 2;
        else if(numImages < 10) ret_Div = 3;
        else if(numImages < 17) ret_Div = 4;
        else if(numImages < 26) ret_Div = 5;
        else if(numImages < 37) ret_Div = 6;
        else if(numImages < 50) ret_Div = 7;
        return ret_Div;
    }

    template<typename T>
    inline void show(std::vector<T> vec)
    {
        std::cout << "\n[ ";
        for(auto &v:vec)
        {
            std::cout << std::dec << v << ", " ;
        }
        std::cout << " ]" << std::endl;
    }

    inline bool pathValidation(const std::string &path)
    {
        struct stat sb; 
        if(stat(path.c_str(), &sb) == 0)
        {
            return true;
        }
        return false;
    }

    inline std::string getAllPath(const std::string &path)
    {
        if(path[0]=='\\')return path;
#ifdef __linux__
        char* temp = realpath(path.c_str(), NULL);
#elif _WIN32
        char* temp = _fullpath(NULL, path.c_str(), _MAX_PATH);
#endif
        if (temp == nullptr)
        {
            return "";
        }
        std::string absolutePath(temp);
        free(temp);
        return absolutePath;
    }

    inline bool dirValidation(const std::string &path)
    {
#ifdef __linux__
        struct stat sb;
        return (stat(path.c_str(), &sb) == 0) && (sb.st_mode & S_IFDIR);
#elif _WIN32
        DWORD attr = GetFileAttributes(path.c_str());
        return (attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY));
#endif
    }

    inline std::string getFileName(const std::string &path)
    {
        return path.substr(path.find_last_of("/\\") + 1);
    }

    inline std::vector<std::string> loadFilesFromDir(const std::string &path)
    {
        std::vector<std::string> result;
        if(!pathValidation(path))
            return result;
        
#ifdef __linux__
        DIR *dirIter = nullptr;
        struct dirent *entry = nullptr;
        if(pathValidation(path))
        {
            dirIter = opendir(path.c_str());
            if(dirIter != nullptr)
            {
                while((entry = readdir(dirIter)))
                {
                    if(strcmp(entry->d_name, "..") > 0)
                        result.emplace_back(entry->d_name);
                }
            }
        }
        closedir(dirIter);
#elif _WIN32
        std::string searchPath = path + "\\*";
        WIN32_FIND_DATA findData;
        HANDLE hFind = FindFirstFile(searchPath.c_str(), &findData);

        if (hFind != INVALID_HANDLE_VALUE) {
            do {
                std::string fileName = findData.cFileName;
                if (fileName != "." && fileName != "..") {
                    result.emplace_back(fileName);
                }
            } while (FindNextFile(hFind, &findData) != 0);
            FindClose(hFind);
        }
#endif
        return result;
    }
    
    inline std::string getExtension(const std::string& path)
    {
        size_t pos = path.find_last_of(".");
        if(pos == std::string::npos) return "";
        return path.substr(pos+1);
    }

    inline bool checkOrtLinking()
    {
#ifdef __linux__
        std::ostringstream command;
        command << "ldconfig -p | grep dxrt.so";

        FILE* pipe = popen(command.str().c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to run ldconfig command." << std::endl;
            return false;
        }

        char buffer[128];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);

        if(result.empty())
            return false;

        std::string file_path;
        size_t pos = result.find("=>");
        if(pos == std::string::npos) return false;

        file_path = result.substr(pos+3);
        file_path.erase(file_path.find_last_not_of('\n') + 1);

        if(!pathValidation(file_path))
            return false;

        command.str("");
        command << "ldd " << file_path << " | grep libonnxruntime.so";

        pipe = popen(command.str().c_str(), "r");
        if(!pipe) {
            std::cerr << "Failed to run ldd command" << std::endl;
            return false;
        }
        result = "";
        while(fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);

        return !result.empty();
#elif _WIN32
        return ORT_OPTION_DEFAULT;
#endif
    }

    inline std::string getLocalTimeString()
    {
        std::time_t now = std::time(nullptr);
        std::tm local{};

#ifdef __linux__
        localtime_r(&now, &local);
#elif _WIN32
        localtime_s(&local, &now);
#endif
        std::ostringstream oss;
        oss << std::put_time(&local, "%Y-%m-%d_%H-%M-%S");
        return oss.str();
    }

    inline void logThreadFunction(void *args)
    {
        std::vector<std::string> log_messages;
        StatusLog* sl = (StatusLog*)args;
        std::mutex cliCommandLock;
        {
            std::unique_lock<std::mutex> _uniqueLock(cliCommandLock);
            sl->statusCheckCV.wait(_uniqueLock, [&](){return sl->threadStatus.load() > 0;});
        }
        if(sl->threadStatus.load() == 1)
            return;
        std::string fileName = std::string("device_status.") + getLocalTimeString() + ".log";
        std::fstream logFile(fileName, std::ios::app | std::ios::in | std::ios::out);
        while(sl->threadStatus.load() == 2)
        {
            auto status = dxrt::DeviceStatus::GetCurrentStatus(0);
            auto devices = status.GetDeviceCount();
            {
                std::unique_lock<std::mutex> _uniqueLock(cliCommandLock);
                std::string log_message = std::to_string(sl->frameNumber) + ", " +
                                          std::to_string(sl->runningTime) + ", ";
                std::string log_result = 
                             std::string("[Application Status] ") + getLocalTimeString() +
                            " Frame No. " + std::to_string(sl->frameNumber) + 
                            ", running time " + std::to_string(sl->runningTime) + "ms, ";

                for (int i = 0; i < devices * 3; i++)
                {
                    auto ret = status.Temperature(i);
                    log_result += std::to_string(ret) + "\'C,";
                    log_message += std::to_string(ret) + ", ";
                }
                std::cout << log_result << std::endl;
                logFile << log_message << std::endl;
                logFile.flush();
                int fd = fileno(stdout);
                if (fd != -1) fsync(fd);
            }
        }
        logFile.close();
        std::cout << "Logs saved to " << fileName << std::endl;
        std::cout << "logging stopped" << std::endl;
    }

    inline bool isVersionGreaterOrEqual(const std::string& v1, const std::string& v2) 
    {
        // Remove 'v' prefix if present
        std::string version1 = v1;
        std::string version2 = v2;
        
        if (!version1.empty() && version1[0] == 'v') {
            version1 = version1.substr(1);
        }
        if (!version2.empty() && version2[0] == 'v') {
            version2 = version2.substr(1);
        }
        
        std::istringstream s1(version1), s2(version2);
        int num1 = 0, num2 = 0;
        char dot;

        while (s1.good() || s2.good()) {
            if (s1.good()) s1 >> num1;
            if (s2.good()) s2 >> num2;

            if (num1 < num2) return false;
            if (num1 > num2) return true;

            num1 = num2 = 0;
            if (s1.good()) s1 >> dot; 
            if (s2.good()) s2 >> dot;
        }
        return true;
    }

    inline bool minversionforRTandCompiler(dxrt::InferenceEngine* ie)
    {
        std::string rt_version = dxrt::Configuration::GetInstance().GetVersion();
        std::string compiler_version = ie->GetModelVersion();
        
        // Debug output to understand what versions we're comparing
        // std::cout << "[DEBUG] RT version: " << rt_version << std::endl;
        // std::cout << "[DEBUG] Compiler version: " << compiler_version << std::endl;
        
        if(isVersionGreaterOrEqual(rt_version, "3.0.0"))
        {
            if(isVersionGreaterOrEqual(compiler_version, "v7"))
            {
                // std::cout << "[DEBUG] Version check passed" << std::endl;
                return true;
            }
            else{
                std::cerr << "[DXAPP] [ER] Compiler version is too low. (required: >= 7, current: " << compiler_version << ")" << std::endl;
            }
        }
        else{
            std::cerr << "[DXAPP] [ER] DXRT library version is too low. (required: >= 3.0.0, current: " << rt_version << ")" << std::endl;
        }
        std::cout << "[DEBUG] Version check failed" << std::endl;
        return false;
    }

} // namespace common
} // namespace dxapp 

#pragma once
#include <sys/stat.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#if _WIN32
#include <Windows.h>
#include <io.h>
#include <cstdio>
#include <thread>
#define fsync _commit
#define popen _popen
#define pclose _pclose
#define fileno _fileno
#endif

namespace dxapp
{
namespace common
{
    
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
    }

    inline std::vector<std::string> checkNpuTemperature()
    {
        std::vector<std::string> temperature_result; // 11
        std::ostringstream command;
        command << "dxrt-cli -s";

        FILE* pipe = popen(command.str().c_str(), "r");
        if (!pipe) {
            std::cerr << "Failed to run ldconfig command." << std::endl;
            return temperature_result;
        }

        char buffer[1000];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) 
        {
            result += buffer;
        }
        pclose(pipe);

        if(result.empty())
            return temperature_result;
        size_t pos = result.find("temperature ");
        size_t end_pos = result.find("\'C", pos + 12);
        while(pos != std::string::npos)
        {
            std::string str(result.begin() + pos + 12, result.begin() + end_pos);
            temperature_result.push_back(str);
            pos = result.find("temperature ", end_pos);
            end_pos = result.find("\'C", pos + 12);
        }
        return temperature_result;
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
            auto ret = dxapp::common::checkNpuTemperature();
            {
                std::unique_lock<std::mutex> _uniqueLock(cliCommandLock);
                std::string log_message = std::to_string(sl->frameNumber) + ", " +
                                          std::to_string(sl->runningTime) + ", ";
                std::string log_result = 
                             std::string("[Application Status] ") + 
                             "Frame No. " + std::to_string(sl->frameNumber) + 
                             ", running time " + std::to_string(sl->runningTime) + "ms, ";
                for(auto &s:ret)
                {
                    log_result += s + "\'C, ";
                    log_message += s + ", ";
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

    inline bool compareVersions(const std::string& v1, const std::string& v2) 
    {
        std::istringstream s1(v1.substr(1)), s2(v2);
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
#ifdef __linux__
        return false;
#elif _WIN32
        return true;
#endif
    }

} // namespace common
} // namespace dxapp 
#include "common_util.hpp"

namespace dxapp {
namespace common {

template <typename T>
void readBinary(const std::string& filePath, T* dst) {
    std::FILE* fp = NULL;
    fp = std::fopen(filePath.c_str(), "rb");
    if (fp == NULL) {
        std::cout << "Failed to open file: " << filePath << std::endl;
        return;
    }

    std::fseek(fp, 0, SEEK_END);
    auto size = ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    int read_size = fread((void*)dst, sizeof(T), size, fp);
    if (read_size != size)
        std::cout << "file size mismatch(" << read_size << ", " << size
                  << "), fail to read file " << filePath << std::endl;
    fclose(fp);
}

void dumpBinary(void* ptr, int dump_size, const std::string& file_name) {
    std::ofstream outfile(file_name, std::ios::binary);
    if (!outfile.is_open()) {
        std::cout << "can not open file " << file_name << std::endl;
        std::terminate();
    }
    outfile.write((char*)ptr, dump_size);
    outfile.close();
}

void readCSV(const std::string& filePath, float* dst, int size) {
    std::ifstream file;
    std::string value;
    file.open(filePath);
    if (!file.is_open()) {
        std::cout << "Failed to open CSV file: " << filePath << std::endl;
        return;
    }

    for (int i = 0; i < size; i++) {
        if (std::getline(file, value)) {
            dst[i] = std::stof(value);
        } else {
            std::cout << "Unexpected end of file at line " << i << std::endl;
            break;
        }
    }
    file.close();
}

int divideBoard(int numImages) {
    int ret_Div = 1;
    if (numImages < 2)
        ret_Div = 1;
    else if (numImages < 5)
        ret_Div = 2;
    else if (numImages < 10)
        ret_Div = 3;
    else if (numImages < 17)
        ret_Div = 4;
    else if (numImages < 26)
        ret_Div = 5;
    else if (numImages < 37)
        ret_Div = 6;
    else if (numImages < 50)
        ret_Div = 7;
    return ret_Div;
}

bool pathValidation(const std::string& path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0) {
        return true;
    }
    return false;
}

std::string getAllPath(const std::string& path) {
    if (path[0] == '\\') return path;
#ifdef __linux__
    char* temp = realpath(path.c_str(), NULL);
#elif _WIN32
    char* temp = _fullpath(NULL, path.c_str(), _MAX_PATH);
#endif
    if (temp == nullptr) {
        return "";
    }
    std::string absolutePath(temp);
    free(temp);
    return absolutePath;
}

bool dirValidation(const std::string& path) {
#ifdef __linux__
    struct stat sb;
    return (stat(path.c_str(), &sb) == 0) && (sb.st_mode & S_IFDIR);
#elif _WIN32
    DWORD attr = GetFileAttributes(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES &&
            (attr & FILE_ATTRIBUTE_DIRECTORY));
#endif
}

std::vector<std::string> loadFilesFromDir(const std::string& path) {
    std::vector<std::string> result;
    if (!pathValidation(path)) return result;

#ifdef __linux__
    DIR* dirIter = nullptr;
    struct dirent* entry = nullptr;
    if (pathValidation(path)) {
        dirIter = opendir(path.c_str());
        if (dirIter != nullptr) {
            while ((entry = readdir(dirIter))) {
                if (strcmp(entry->d_name, "..") > 0)
                    result.emplace_back(entry->d_name);
            }
            closedir(dirIter);
        }
    }
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

bool checkOrtLinking() {
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

    if (result.empty()) return false;

    std::string file_path;
    size_t pos = result.find("=>");
    if (pos == std::string::npos) return false;

    file_path = result.substr(pos + 3);
    file_path.erase(file_path.find_last_not_of('\n') + 1);

    if (!pathValidation(file_path)) return false;

    command.str("");
    command << "ldd " << file_path << " | grep libonnxruntime.so";

    pipe = popen(command.str().c_str(), "r");
    if (!pipe) {
        std::cerr << "Failed to run ldd command" << std::endl;
        return false;
    }
    result = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);

    return !result.empty();
#elif _WIN32
    return ORT_OPTION_DEFAULT;
#endif
}

std::string getLocalTimeString() {
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

void logThreadFunction(void* args) {
    std::vector<std::string> log_messages;
    StatusLog* sl = (StatusLog*)args;
    std::mutex cliCommandLock;
    {
        std::unique_lock<std::mutex> _uniqueLock(cliCommandLock);
        sl->statusCheckCV.wait(_uniqueLock,
                               [&]() { return sl->threadStatus.load() > 0; });
    }
    if (sl->threadStatus.load() == 1) return;
    std::string fileName =
        std::string("device_status.") + getLocalTimeString() + ".log";
    std::fstream logFile(fileName,
                         std::ios::app | std::ios::in | std::ios::out);
    while (sl->threadStatus.load() == 2) {
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

            for (int i = 0; i < devices * 3; i++) {
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

bool isVersionGreaterOrEqual(const std::string& v1, const std::string& v2) {
    std::istringstream s1(v1), s2(v2);
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

bool minversionforRTandCompiler(dxrt::InferenceEngine* ie) {
    std::string rt_version = dxrt::Configuration::GetInstance().GetVersion();
    std::string compiler_version = ie->GetModelVersion();
    if (isVersionGreaterOrEqual(rt_version, "3.0.0")) {
        if (isVersionGreaterOrEqual(compiler_version, "v7")) {
            return true;
        } else {
            std::cerr << "[DXAPP] [ER] Compiler version is too low. (required: "
                         ">= 7, current: "
                      << compiler_version << ")" << std::endl;
        }
    } else {
        std::cerr << "[DXAPP] [ER] DXRT library version is too low. (required: "
                     ">= 3.0.0, current: "
                  << rt_version << ")" << std::endl;
    }
    return false;
}

std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }

    return tokens;
}

// Explicit template instantiations for common types
template void readBinary<int>(const std::string& filePath, int* dst);
template void readBinary<float>(const std::string& filePath, float* dst);
template void readBinary<double>(const std::string& filePath, double* dst);
template void readBinary<char>(const std::string& filePath, char* dst);
template void readBinary<unsigned char>(const std::string& filePath,
                                        unsigned char* dst);

std::string get_coco_class_name(const int class_id) {
    static const std::vector<std::string> class_names = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush"};
    return class_names.at(class_id);
}

}  // namespace common
}  // namespace dxapp

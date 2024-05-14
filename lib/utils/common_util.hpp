#pragma once
#include <sys/stat.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace dxapp
{
namespace common
{
    int get_align_factor(int length, int based)
    {
        return based - (length - (length & (-based)));
    }
    
    template<typename T>
    void readBinary(std::string filePath, T* dst, int elemSize)
    {
        std::FILE *fp = NULL;
        fp = std::fopen(filePath.c_str(), "rb");
        std::fseek(fp, 0, SEEK_END);
        auto size = ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        int read_size = fread((void*)dst, size, elemSize, fp);
        if(read_size != size)
            std::cout << "file size mismatch, fail to read file " << filePath << std::endl;
        fclose(fp);
    }

    void dumpBinary(void *ptr, int dump_size, std::string file_name)
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
    
    void readCSV(std::string filePath, float* dst, int size)
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
    
    int divideBoard(int numImages)
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
    void show(std::vector<T> vec)
    {
        std::cout << "\n[ ";
        for(auto &v:vec)
        {
            std::cout << std::dec << v << ", " ;
        }
        std::cout << " ]" << std::endl;
    }

    bool pathValidation(const std::string &path)
    {
        struct stat sb; 
        if(stat(path.c_str(), &sb) == 0)
        {
            return true;
        }
        return false;
    }

    bool dirValidation(const std::string &path)
    {
        struct stat sb;
        stat(path.c_str(), &sb);
        if (S_ISDIR(sb.st_mode))
        {
            return true;
        }
        return false;
    }

    std::string getFileName(const std::string &path)
    {
        return path.substr(path.find_last_of("/\\") + 1);
    }

    std::vector<std::string> loadFilesFromDir(const std::string &path)
    {
        DIR *dirIter = nullptr;
        struct dirent *entry = nullptr;
        std::vector<std::string> result;
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
        return result;
    }
    
    std::string getExtension(const std::string& path)
    {
        size_t pos = path.find_last_of(".");
        if(pos == std::string::npos) return "";
        return path.substr(pos+1);
    }

} // namespace common
} // namespace dxapp 
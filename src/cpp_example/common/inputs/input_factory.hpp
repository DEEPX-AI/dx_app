/**
 * @file input_factory.hpp
 * @brief Factory Method for creating input sources
 */

#ifndef DXAPP_INPUT_FACTORY_HPP
#define DXAPP_INPUT_FACTORY_HPP

#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "../base/i_input_source.hpp"
#include "image_source.hpp"
#include "video_source.hpp"
#include "camera_source.hpp"
#include "rtsp_source.hpp"

namespace dxapp {

/**
 * @brief Factory class for creating appropriate input sources
 * 
 * Implements the Factory Method pattern to create the correct
 * input source based on the input type (image, video, camera, RTSP).
 */
class InputFactory {
public:
    /**
     * @brief Create an input source from a file path
     * @param path Path to image or video file
     * @return Unique pointer to the appropriate input source
     * @throws std::runtime_error if file cannot be opened
     */
    static InputSourcePtr createFromFile(const std::string& path) {
        InputType type = detectFileType(path);
        
        InputSourcePtr source;
        switch (type) {
            case InputType::IMAGE:
                source = std::make_unique<ImageSource>(path);
                break;
            case InputType::VIDEO:
                source = std::make_unique<VideoSource>(path);
                break;
            default:
                throw std::runtime_error("Unknown file type: " + path);
        }
        
        if (!source->isOpened()) {
            throw std::runtime_error("Failed to open: " + path);
        }
        
        return source;
    }

    /**
     * @brief Create a camera input source
     * @param device_id Camera device index
     * @param width Desired capture width (0 for default)
     * @param height Desired capture height (0 for default)
     * @param fps Desired capture FPS (0 for default)
     * @return Unique pointer to camera source
     * @throws std::runtime_error if camera cannot be opened
     */
    static InputSourcePtr createFromCamera(int device_id, 
                                           int width = 0, 
                                           int height = 0, 
                                           double fps = 0.0) {
        auto source = std::make_unique<CameraSource>(device_id, width, height, fps);
        
        if (!source->isOpened()) {
            throw std::runtime_error("Failed to open camera: " + std::to_string(device_id));
        }
        
        return source;
    }

    /**
     * @brief Create an RTSP stream input source
     * @param url RTSP stream URL
     * @return Unique pointer to RTSP source
     * @throws std::runtime_error if stream cannot be opened
     */
    static InputSourcePtr createFromRTSP(const std::string& url) {
        auto source = std::make_unique<RTSPSource>(url);
        
        if (!source->isOpened()) {
            throw std::runtime_error("Failed to open RTSP stream: " + url);
        }
        
        return source;
    }

    /**
     * @brief Create input source based on command line arguments
     * @param image_path Path to image file (empty if not used)
     * @param video_path Path to video file (empty if not used)
     * @param camera_index Camera index (-1 if not used)
     * @param rtsp_url RTSP URL (empty if not used)
     * @return Unique pointer to the appropriate input source
     * @throws std::runtime_error if no valid input or input cannot be opened
     */
    static InputSourcePtr create(const std::string& image_path,
                                 const std::string& video_path,
                                 int camera_index,
                                 const std::string& rtsp_url) {
        // Validate exactly one input source
        int source_count = 0;
        if (!image_path.empty()) source_count++;
        if (!video_path.empty()) source_count++;
        if (camera_index >= 0) source_count++;
        if (!rtsp_url.empty()) source_count++;

        if (source_count == 0) {
            throw std::runtime_error("No input source specified");
        }
        if (source_count > 1) {
            throw std::runtime_error("Multiple input sources specified. Please specify only one.");
        }

        // Create appropriate source
        if (!image_path.empty()) {
            return createFromFile(image_path);
        }
        if (!video_path.empty()) {
            return createFromFile(video_path);
        }
        if (camera_index >= 0) {
            return createFromCamera(camera_index);
        }
        if (!rtsp_url.empty()) {
            return createFromRTSP(rtsp_url);
        }

        throw std::runtime_error("Invalid input configuration");
    }

    /**
     * @brief Detect file type based on extension
     * @param path File path
     * @return InputType (IMAGE or VIDEO)
     */
    static InputType detectFileType(const std::string& path) {
        // Get extension
        size_t dot_pos = path.rfind('.');
        if (dot_pos == std::string::npos) {
            return InputType::UNKNOWN;
        }
        
        std::string ext = path.substr(dot_pos + 1);
        // Convert to lowercase
        std::transform(ext.begin(), ext.end(), ext.begin(), 
                       [](unsigned char c) { return std::tolower(c); });

        // Image extensions
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || 
            ext == "bmp" || ext == "tiff" || ext == "webp") {
            return InputType::IMAGE;
        }
        
        // Video extensions
        if (ext == "mp4" || ext == "avi" || ext == "mov" || 
            ext == "mkv" || ext == "wmv" || ext == "flv" || ext == "webm") {
            return InputType::VIDEO;
        }

        return InputType::UNKNOWN;
    }
};

}  // namespace dxapp

#endif  // DXAPP_INPUT_FACTORY_HPP

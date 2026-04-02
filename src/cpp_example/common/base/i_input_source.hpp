/**
 * @file i_input_source.hpp
 * @brief Abstract interface for input sources (Factory Method pattern)
 * 
 * This interface defines the contract for all input sources (image, video, camera, RTSP).
 * Part of DX-APP v3.0.0 refactoring for independent build capability.
 */

#ifndef DXAPP_I_INPUT_SOURCE_HPP
#define DXAPP_I_INPUT_SOURCE_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>

namespace dxapp {

/**
 * @brief Input source type enumeration
 */
enum class InputType {
    IMAGE,
    VIDEO,
    CAMERA,
    RTSP,
    UNKNOWN
};

/**
 * @brief Abstract interface for input sources
 * 
 * All input sources (image, video, camera, RTSP) must implement this interface.
 * This enables the Factory Method pattern for creating appropriate input handlers.
 */
class IInputSource {
public:
    virtual ~IInputSource() = default;

    /**
     * @brief Get the next frame from the input source
     * @param frame Output frame (BGR format)
     * @return true if frame was successfully read, false otherwise
     */
    virtual bool getFrame(cv::Mat& frame) = 0;

    /**
     * @brief Check if the input source is opened/available
     * @return true if source is ready to provide frames
     */
    virtual bool isOpened() const = 0;

    /**
     * @brief Release the input source resources
     */
    virtual void release() = 0;

    /**
     * @brief Get the type of input source
     * @return InputType enum value
     */
    virtual InputType getType() const = 0;

    /**
     * @brief Get the width of frames
     * @return Frame width in pixels
     */
    virtual int getWidth() const = 0;

    /**
     * @brief Get the height of frames
     * @return Frame height in pixels
     */
    virtual int getHeight() const = 0;

    /**
     * @brief Get the FPS of the source (if applicable)
     * @return FPS value, 0 for static images
     */
    virtual double getFPS() const = 0;

    /**
     * @brief Get total frame count (for video files)
     * @return Total frames, -1 for streams/cameras, 1 for images
     */
    virtual int getTotalFrames() const = 0;

    /**
     * @brief Get source description for logging
     * @return Human-readable source description
     */
    virtual std::string getDescription() const = 0;

    /**
     * @brief Check if this is a live source (camera/RTSP)
     * @return true for live sources, false for files
     */
    virtual bool isLiveSource() const = 0;
};

// Alias for smart pointer
using InputSourcePtr = std::unique_ptr<IInputSource>;

}  // namespace dxapp

#endif  // DXAPP_I_INPUT_SOURCE_HPP

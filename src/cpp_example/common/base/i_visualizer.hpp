/**
 * @file i_visualizer.hpp
 * @brief Abstract interface for result visualization
 * 
 * This interface defines the contract for visualizing inference results.
 */

#ifndef DXAPP_I_VISUALIZER_HPP
#define DXAPP_I_VISUALIZER_HPP

#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include <string>

#include "i_processor.hpp"

namespace dxapp {

/**
 * @brief Abstract interface for visualizers
 * 
 * Visualizers draw inference results on images for display/saving.
 * This is a template interface to support different result types.
 */
template <typename ResultType>
class IVisualizer {
public:
    virtual ~IVisualizer() = default;

    /**
     * @brief Draw results on the image
     * @param frame Original image (will be modified)
     * @param results Vector of results to visualize
     * @param ctx Preprocessing context for coordinate transformation
     * @return Visualized image
     */
    virtual cv::Mat draw(const cv::Mat& frame, 
                         const std::vector<ResultType>& results,
                         const PreprocessContext& ctx) = 0;

    /**
     * @brief Set visualization parameters
     * @param line_thickness Thickness of bounding box lines
     * @param font_scale Scale of text labels
     * @param alpha Transparency for overlays (0.0 to 1.0)
     */
    virtual void setParameters(int line_thickness = 2, 
                               double font_scale = 0.5, 
                               float alpha = 0.6f) = 0;
};

// Smart pointer aliases
template <typename T>
using VisualizerPtr = std::unique_ptr<IVisualizer<T>>;

}  // namespace dxapp

#endif  // DXAPP_I_VISUALIZER_HPP

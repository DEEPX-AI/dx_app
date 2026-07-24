/**
 * @file sfa3d_visualizer.hpp
 * @brief SFA3D horizontal multi-view visualization (BEV | Image | Cam+Box)
 */

#ifndef SFA3D_VISUALIZER_HPP
#define SFA3D_VISUALIZER_HPP

#include "common/base/i_visualizer.hpp"
#include "common/utility/kitti_calib.hpp"
#include "common/utility/lidar_util.hpp"
#include "common/utility/sfa3d_geometry.hpp"
#include <opencv2/opencv.hpp>

namespace dxapp {

class SFA3DVisualizer : public IVisualizer<Detection3DResult> {
public:
    SFA3DVisualizer() = default;

    void setSourcePath(const std::string& path) { source_path_ = path; }

    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<Detection3DResult>& results,
                 const PreprocessContext& /*ctx*/) override {
        cv::Mat bev = prepareBevFrame(frame);
        // Display-only mirror: model BEV keeps +y on the right; flip raster before overlays.
        cv::flip(bev, bev, 1);
        bev = drawBevBoxes(bev, results, panel_size_, panel_size_, class_names_);

        if (source_path_.empty() || !isLidarInputPath(source_path_)) {
            return composeRowPanels({bev}, {"BEV"}, panel_size_);
        }

        cv::Mat points = loadKittiPointCloud(source_path_);
        std::string calib_path;
        if (!findCalibPath(source_path_, calib_path)) {
            cv::Mat placeholder(panel_size_, panel_size_, CV_8UC3, cv::Scalar(24, 24, 28));
            cv::putText(placeholder, "No calib", cv::Point(10, panel_size_ / 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
            return composeRowPanels({bev, placeholder, placeholder},
                                    {"BEV", "Image", "Cam+Box"}, panel_size_);
        }

        KittiCalib calib;
        loadKittiCalib(calib_path, calib);
        cv::Mat raw_img;
        std::string image_path;
        if (findImagePath(source_path_, image_path)) {
            raw_img = cv::imread(image_path, cv::IMREAD_COLOR);
        }
        if (raw_img.empty()) {
            raw_img = renderSyntheticCameraImage(points, calib);
        }
        cv::Mat cam_overlay = renderCameraOverlay(raw_img.clone(), results, calib);
        return composeRowPanels({bev, raw_img, cam_overlay},
                                {"BEV", "Image", "Cam+Box"}, panel_size_);
    }

    void setParameters(int line_thickness = 2, double font_scale = 0.5, float alpha = 0.6f) override {
        line_thickness_ = line_thickness;
        font_scale_ = font_scale;
        alpha_ = alpha;
    }

private:
    cv::Mat prepareBevFrame(const cv::Mat& frame) {
        cv::Mat output = frame.clone();
        if (output.type() != CV_8UC3) {
            cv::Mat converted;
            if (output.type() == CV_32FC3) output.convertTo(converted, CV_8UC3, 255.0);
            else cv::cvtColor(output, converted, cv::COLOR_GRAY2BGR);
            output = converted;
        }
        return output;
    }

    std::string source_path_;
    int panel_size_{608};
    int line_thickness_{2};
    double font_scale_{0.5};
    float alpha_{0.6f};
    std::vector<std::string> class_names_{"Pedestrian", "Car", "Cyclist"};
};

}  // namespace dxapp

#endif  // SFA3D_VISUALIZER_HPP

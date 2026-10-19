/**
 * @file detection_runner_ops.hpp
 * @brief What the detection runners need to know about their per-frame result.
 *
 * DetectionResult: the factory's createPostprocessor()/createVisualizer().
 * PanopticResult (YOLOPv2): ONE result per frame carrying that frame's boxes
 * and masks, from createPanopticPostprocessor()/createPanopticVisualizer().
 * boxes() is what the runners print as [DET] lines.
 */
#ifndef DXAPP_DETECTION_RUNNER_OPS_HPP
#define DXAPP_DETECTION_RUNNER_OPS_HPP

#include <vector>

#include "common/base/i_factory.hpp"
#include "common/base/i_processor.hpp"
#include "common/base/i_visualizer.hpp"

namespace dxapp {

template <typename ResultT>
struct DetectionRunnerOps;

template <>
struct DetectionRunnerOps<DetectionResult> {
    template <typename FactoryT>
    static PostprocessorPtr<DetectionResult> createPostprocessor(FactoryT& factory, int w, int h, bool ort) {
        return factory.createPostprocessor(w, h, ort);
    }
    template <typename FactoryT>
    static VisualizerPtr<DetectionResult> createVisualizer(FactoryT& factory) {
        return factory.createVisualizer();
    }
    static const std::vector<DetectionResult>& boxes(const std::vector<DetectionResult>& results) {
        return results;
    }
};

template <>
struct DetectionRunnerOps<PanopticResult> {
    template <typename FactoryT>
    static PostprocessorPtr<PanopticResult> createPostprocessor(FactoryT& factory, int w, int h, bool /*ort*/) {
        return factory.createPanopticPostprocessor(w, h);
    }
    template <typename FactoryT>
    static VisualizerPtr<PanopticResult> createVisualizer(FactoryT& factory) {
        return factory.createPanopticVisualizer();
    }
    static const std::vector<DetectionResult>& boxes(const std::vector<PanopticResult>& results) {
        static const std::vector<DetectionResult> kNone;
        return results.empty() ? kNone : results.front().detections;
    }
};

}  // namespace dxapp

#endif  // DXAPP_DETECTION_RUNNER_OPS_HPP

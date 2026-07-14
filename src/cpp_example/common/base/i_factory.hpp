/**
 * @file i_factory.hpp
 * @brief Abstract Factory interface for model component creation
 * 
 * This interface defines the Abstract Factory pattern for creating
 * matching sets of preprocessor, postprocessor, and visualizer components.
 */

#ifndef DXAPP_I_FACTORY_HPP
#define DXAPP_I_FACTORY_HPP

#include <array>
#include <memory>
#include <string>

#include "i_processor.hpp"
#include "i_visualizer.hpp"

namespace dxapp {

// Forward declaration for config loading
class ModelConfig;

/**
 * @brief Input normalization parameters for float-input models.
 *
 * When @c apply_mean_std is true, the runner feeds the model
 * (pixel/255 - mean) / std per channel. mean/std are in the preprocessor's
 * output channel order (RGB when BGR->RGB conversion is applied).
 * When false, float-input models receive plain /255 normalization and
 * uint8-input models are unaffected.
 */
struct InputNormalizationParams {
    bool apply_mean_std = false;
    std::array<float, 3> mean{0.f, 0.f, 0.f};
    std::array<float, 3> std{1.f, 1.f, 1.f};
};

/**
 * @brief Abstract Factory interface for object detection models
 * 
 * Creates matching sets of components for object detection models.
 * Each concrete factory (e.g., YOLOv5Factory) creates components
 * that are guaranteed to work together correctly.
 */
class IDetectionFactory {
public:
    virtual ~IDetectionFactory() = default;

    /**
     * @brief Create a preprocessor for this model
     * @param input_width Model input width
     * @param input_height Model input height
     * @return Unique pointer to preprocessor
     */
    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;

    /**
     * @brief Create a postprocessor for this model
     * @param input_width Model input width
     * @param input_height Model input height
     * @param is_ort_configured Whether ORT inference is configured
     * @return Unique pointer to detection postprocessor
     */
    virtual PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;

    /**
     * @brief Create a visualizer for this model
     * @return Unique pointer to detection visualizer
     */
    virtual VisualizerPtr<DetectionResult> createVisualizer() = 0;

    /**
     * @brief Get the model name this factory is for
     * @return Model name string (e.g., "YOLOv5", "YOLOv8")
     */
    virtual std::string getModelName() const = 0;

    /**
     * @brief Get the task type this factory is for
     * @return Task type string (e.g., "object_detection")
     */
    virtual std::string getTaskType() const = 0;

    /**
     * @brief Load configuration from an external JSON file
     * @param config Parsed ModelConfig instance
     *
     * Override in concrete factories to apply runtime parameters.
     * Default implementation is a no-op (uses constructor defaults).
     */
    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for semantic segmentation models
 */
class ISegmentationFactory {
public:
    virtual ~ISegmentationFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<SegmentationResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<SegmentationResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for classification models
 */
class IClassificationFactory {
public:
    virtual ~IClassificationFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<ClassificationResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<ClassificationResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for face detection models
 */
class IFaceDetectionFactory {
public:
    virtual ~IFaceDetectionFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<FaceDetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    
    virtual VisualizerPtr<FaceDetectionResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for pose estimation models
 */
class IPoseFactory {
public:
    virtual ~IPoseFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    
    virtual VisualizerPtr<PoseResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for instance segmentation models
 */
class IInstanceSegmentationFactory {
public:
    virtual ~IInstanceSegmentationFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<InstanceSegmentationResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    
    virtual VisualizerPtr<InstanceSegmentationResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for OBB (Oriented Bounding Box) detection models
 */
class IOBBFactory {
public:
    virtual ~IOBBFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<OBBResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    
    virtual VisualizerPtr<OBBResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for 3D LiDAR detection models (SFA3D)
 */
class I3DDetectionFactory {
public:
    virtual ~I3DDetectionFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;

    virtual PostprocessorPtr<Detection3DResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;

    virtual VisualizerPtr<Detection3DResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for depth estimation models
 */
class IDepthEstimationFactory {
public:
    virtual ~IDepthEstimationFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<DepthResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<DepthResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }

    /**
     * @brief Input normalization for float-input depth models.
     *
     * Default: no mean/std (runner uses plain /255 for float inputs, or raw
     * uint8 for uint8 inputs). Override for models such as Depth Anything V2
     * that require ImageNet mean/std normalization.
     */
    virtual InputNormalizationParams getInputNormalization() const { return {}; }
};

/**
 * @brief Abstract Factory interface for image restoration models
 */
class IRestorationFactory {
public:
    virtual ~IRestorationFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<RestorationResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<RestorationResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for embedding/feature extraction models
 */
class IEmbeddingFactory {
public:
    virtual ~IEmbeddingFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<EmbeddingResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<EmbeddingResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for face alignment / 3D face reconstruction models
 * 
 * Creates matching sets of components for models that output
 * 3DMM parameters and facial landmarks (3DDFA v2, etc.).
 */
class IFaceAlignmentFactory {
public:
    virtual ~IFaceAlignmentFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<FaceAlignmentResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    
    virtual VisualizerPtr<FaceAlignmentResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for hand landmark detection models
 * 
 * Creates matching sets of components for models that output
 * hand keypoints (MediaPipe Hands, etc.).
 */
class IHandLandmarkFactory {
public:
    virtual ~IHandLandmarkFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    
    virtual PostprocessorPtr<HandLandmarkResult> createPostprocessor(
        int input_width, int input_height) = 0;
    
    virtual VisualizerPtr<HandLandmarkResult> createVisualizer() = 0;

    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;

    virtual void loadConfig(const ModelConfig& /*config*/) { /* No-op: subclasses override to apply runtime parameters */ }
};

/**
 * @brief Abstract Factory interface for object pose estimation models (e.g. DOPE).
 */
class IObjectPoseFactory {
public:
    virtual ~IObjectPoseFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    virtual PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height) = 0;
    virtual VisualizerPtr<PoseResult> createVisualizer() = 0;
    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;
    virtual void loadConfig(const ModelConfig& /*config*/) {}
};

/**
 * @brief Abstract Factory interface for keypoint detection models (e.g. SuperPoint).
 */
class IKeypointDetectionFactory {
public:
    virtual ~IKeypointDetectionFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    virtual PostprocessorPtr<PoseResult> createPostprocessor(
        int input_width, int input_height) = 0;
    virtual VisualizerPtr<PoseResult> createVisualizer() = 0;
    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;
    virtual void loadConfig(const ModelConfig& /*config*/) {}
};

/**
 * @brief Abstract Factory interface for panoptic driving perception models (e.g. YOLOPv2).
 */
class IPanopticDrivingFactory {
public:
    virtual ~IPanopticDrivingFactory() = default;

    virtual PreprocessorPtr createPreprocessor(int input_width, int input_height) = 0;
    virtual PostprocessorPtr<DetectionResult> createPostprocessor(
        int input_width, int input_height, bool is_ort_configured = false) = 0;
    virtual VisualizerPtr<DetectionResult> createVisualizer() = 0;
    virtual std::string getModelName() const = 0;
    virtual std::string getTaskType() const = 0;
    virtual void loadConfig(const ModelConfig& /*config*/) {}
};

// Smart pointer aliases for factories
using DetectionFactoryPtr = std::unique_ptr<IDetectionFactory>;
using SegmentationFactoryPtr = std::unique_ptr<ISegmentationFactory>;
using ClassificationFactoryPtr = std::unique_ptr<IClassificationFactory>;
using FaceDetectionFactoryPtr = std::unique_ptr<IFaceDetectionFactory>;
using PoseFactoryPtr = std::unique_ptr<IPoseFactory>;
using InstanceSegmentationFactoryPtr = std::unique_ptr<IInstanceSegmentationFactory>;
using OBBFactoryPtr = std::unique_ptr<IOBBFactory>;
using DepthEstimationFactoryPtr = std::unique_ptr<IDepthEstimationFactory>;
using RestorationFactoryPtr = std::unique_ptr<IRestorationFactory>;
using EmbeddingFactoryPtr = std::unique_ptr<IEmbeddingFactory>;
using FaceAlignmentFactoryPtr = std::unique_ptr<IFaceAlignmentFactory>;
using HandLandmarkFactoryPtr = std::unique_ptr<IHandLandmarkFactory>;
using ObjectPoseFactoryPtr = std::unique_ptr<IObjectPoseFactory>;
using KeypointDetectionFactoryPtr = std::unique_ptr<IKeypointDetectionFactory>;
using PanopticDrivingFactoryPtr = std::unique_ptr<IPanopticDrivingFactory>;

}  // namespace dxapp

#endif  // DXAPP_I_FACTORY_HPP

/**
 * @file i_processor.hpp
 * @brief Abstract interfaces for pre-processor and post-processor
 * 
 * These interfaces define the contract for all preprocessing and postprocessing operations.
 */

#ifndef DXAPP_I_PROCESSOR_HPP
#define DXAPP_I_PROCESSOR_HPP

#include <dxrt/dxrt_api.h>
#include <opencv2/opencv.hpp>
#if defined(__has_include)
#if __has_include(<opencv2/dnn.hpp>)
#include <opencv2/dnn.hpp>
#elif __has_include(<opencv2/dnn/dnn.hpp>)
#include <opencv2/dnn/dnn.hpp>
#endif
#endif
#include <cmath>
#include <vector>
#include <memory>
#include <string>

namespace dxapp {

/**
 * @brief Preprocessing context containing metadata needed for postprocessing
 * 
 * This structure holds information about the preprocessing transformations
 * that need to be reversed during postprocessing (e.g., letterbox padding).
 */
struct PreprocessContext {
    int pad_x{0};
    int pad_y{0};
    float scale{1.0f};
    float scale_x{0.0f};
    float scale_y{0.0f};

    int original_width{0};
    int original_height{0};

    int input_width{0};
    int input_height{0};

    cv::Mat source_image;

    PreprocessContext() = default;
    ~PreprocessContext() = default;
    PreprocessContext(const PreprocessContext&) = default;
    PreprocessContext& operator=(const PreprocessContext&) = default;
    PreprocessContext(PreprocessContext&&) noexcept = default;
    PreprocessContext& operator=(PreprocessContext&&) noexcept = default;

    PreprocessContext(int px, int py, float s, int ow, int oh, int iw, int ih)
        : pad_x(px), pad_y(py), scale(s), 
          original_width(ow), original_height(oh),
          input_width(iw), input_height(ih) {}
};

/**
 * @brief Abstract interface for preprocessors
 * 
 * Preprocessors transform input images into the format expected by the model.
 */
class IPreprocessor {
public:
    virtual ~IPreprocessor() = default;

    /**
     * @brief Preprocess an input image for model inference
     * @param input Original input image (BGR format)
     * @param output Preprocessed image ready for inference
     * @param ctx Output preprocessing context for postprocessing
     */
    virtual void process(const cv::Mat& input, cv::Mat& output, PreprocessContext& ctx) = 0;

    /**
     * @brief Get the expected input width for the model
     * @return Model input width
     */
    virtual int getInputWidth() const = 0;

    /**
     * @brief Get the expected input height for the model
     * @return Model input height
     */
    virtual int getInputHeight() const = 0;

    /**
     * @brief Get the expected color space conversion code
     * @return OpenCV color conversion code (e.g., cv::COLOR_BGR2RGB)
     */
    virtual int getColorConversion() const = 0;
};

/**
 * @brief Base detection result structure
 * 
 * This is a common base for all detection-based results.
 * Specific models may extend this with additional fields.
 */
struct DetectionResult {
    std::vector<float> box;  // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::string class_name;

    DetectionResult() = default;
    
    DetectionResult(std::vector<float> b, float conf, int cls_id, const std::string& cls_name)
        : box(std::move(b)), confidence(conf), class_id(cls_id), class_name(cls_name) {}

    float area() const {
        if (box.size() < 4) return 0.0f;
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    float iou(const DetectionResult& other) const {
        if (box.size() < 4 || other.box.size() < 4) return 0.0f;
        
        float x_left = std::max(box[0], other.box[0]);
        float y_top = std::max(box[1], other.box[1]);
        float x_right = std::min(box[2], other.box[2]);
        float y_bottom = std::min(box[3], other.box[3]);

        if (x_right < x_left || y_bottom < y_top) return 0.0f;

        float intersection = (x_right - x_left) * (y_bottom - y_top);
        float union_area = area() + other.area() - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
};

/**
 * @brief Panoptic driving perception result (YOLOPv2): ONE per frame.
 *
 * The frame's vehicle boxes and its own drivable/lane masks travel together,
 * so an async runner can decode frame N+1 while it still draws frame N.
 */
struct PanopticResult {
    std::vector<DetectionResult> detections;
    cv::Mat drivable;  // CV_8UC1 at the original frame size; 1 = drivable area
    cv::Mat lane;      // CV_8UC1 at the original frame size; 1 = lane line

    PanopticResult() = default;
};

/**
 * @brief Segmentation result structure
 */
struct SegmentationResult {
    std::vector<int> mask;  // Flattened H*W mask with class IDs
    int width{0};
    int height{0};
    std::vector<int> class_ids;
    std::vector<std::string> class_names;
    // Soft alpha matte in [0,1], for models whose output is a continuous foreground
    // opacity rather than a class map (PP-Matting). `mask` still carries the
    // thresholded 0/1 class map so SemanticSegmentationVisualizer -- which reads
    // `mask` as class IDs -- renders it unchanged, while a consumer that wants the
    // real matte has it here instead of having to recover it from a binary image.
    // Empty for every class-map model.
    cv::Mat alpha;

    SegmentationResult() = default;
};

/**
 * @brief Classification result structure
 */
struct ClassificationResult {
    int class_id{0};
    std::string class_name;
    float confidence{0.0f};
    std::vector<std::pair<int, float>> top_k;  // top-k predictions

    ClassificationResult() = default;
};

/**
 * @brief Keypoint structure for pose/face landmarks
 */
struct Keypoint {
    float x{0.0f};
    float y{0.0f};
    float confidence{0.0f};

    Keypoint() = default;
    Keypoint(float x_, float y_, float conf = 1.0f) : x(x_), y(y_), confidence(conf) {}
};

/**
 * @brief Face detection result with landmarks
 */
struct FaceDetectionResult {
    std::vector<float> box;  // x1, y1, x2, y2
    float confidence{0.0f};
    std::vector<Keypoint> landmarks;  // 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
    std::string class_name{"face"};   // what the box holds: "hand" for the MediaPipe palm detector

    FaceDetectionResult() = default;
    
    float area() const {
        if (box.size() < 4) return 0.0f;
        return (box[2] - box[0]) * (box[3] - box[1]);
    }

    float iou(const FaceDetectionResult& other) const {
        if (box.size() < 4 || other.box.size() < 4) return 0.0f;
        
        float x_left = std::max(box[0], other.box[0]);
        float y_top = std::max(box[1], other.box[1]);
        float x_right = std::min(box[2], other.box[2]);
        float y_bottom = std::min(box[3], other.box[3]);

        if (x_right < x_left || y_bottom < y_top) return 0.0f;

        float intersection = (x_right - x_left) * (y_bottom - y_top);
        float union_area = area() + other.area() - intersection;
        
        return union_area > 0 ? intersection / union_area : 0.0f;
    }
};

/**
 * @brief Pose estimation result with keypoints
 */
struct PoseResult {
    std::vector<float> box;  // x1, y1, x2, y2 (optional bounding box)
    float confidence{0.0f};
    std::vector<Keypoint> keypoints;  // 17 keypoints for COCO pose
    std::shared_ptr<const std::vector<std::vector<float>>> descriptors;

    PoseResult() = default;
    
    float area() const {
        if (box.size() < 4) return 0.0f;
        return (box[2] - box[0]) * (box[3] - box[1]);
    }
};

/**
 * @brief Instance segmentation result
 */
struct InstanceSegmentationResult {
    std::vector<float> box;  // x1, y1, x2, y2
    float confidence{0.0f};
    int class_id{0};
    std::string class_name;
    cv::Mat mask;  // Instance mask
    int track_id{-1};  // stable multi-frame id assigned by a tracker (-1 = untracked)

    InstanceSegmentationResult() = default;
    ~InstanceSegmentationResult() = default;
    InstanceSegmentationResult(const InstanceSegmentationResult&) = default;
    InstanceSegmentationResult& operator=(const InstanceSegmentationResult&) = default;
    InstanceSegmentationResult(InstanceSegmentationResult&&) noexcept = default;
    InstanceSegmentationResult& operator=(InstanceSegmentationResult&&) noexcept = default;

    float area() const {
        if (box.size() < 4) return 0.0f;
        return (box[2] - box[0]) * (box[3] - box[1]);
    }
};

/**
 * @brief Depth estimation result
 */
struct DepthResult {
    cv::Mat depth_map;    // Float depth map (H x W)
    int width{0};
    int height{0};
    float min_depth{0.0f};
    float max_depth{0.0f};

    DepthResult() = default;
};

/**
 * @brief Anomaly-detection response
 *
 * Mirrors the Python AnomalyResult. Two producers, two meanings:
 * EfficientADPostprocessor runs all three networks and returns EfficientAD's own
 * combination of their disagreements, with `score` read off the RAW combination so it
 * ranks frames; AnomalyFeaturePostprocessor sees one network and can only report a
 * feature magnitude, with `score` off the normalised map. `heatmap` is always
 * normalised against its own frame, so a hot region stays visible whatever the
 * absolute magnitudes -- it is not comparable between frames.
 */
struct AnomalyResult {
    cv::Mat heatmap;      // Float H x W, 0..1, frame-relative
    float score{0.0f};
    int channels{0};

    AnomalyResult() = default;
};

/**
 * @brief Image restoration result
 */
struct RestorationResult {
    cv::Mat restored_image;  // Restored image (H x W x C)
    int width{0};
    int height{0};

    RestorationResult() = default;
};

/**
 * @brief Embedding/feature extraction result
 * 
 * Stores a normalized feature vector from encoder models (e.g., CLIP, ArcFace).
 */
struct GalleryMatch {
    int rank{0};
    float score{0.0f};        // cosine similarity in [-1, 1], higher is nearer
    std::string path;         // gallery image, repo-relative, for the visualizer
    std::string label;        // gallery folder name, empty when unlabelled
};

struct EmbeddingResult {
    std::vector<float> embedding;  // Feature vector
    int dimension{0};              // Embedding dimension

    // Retrieval fields, filled only by GalleryRetrievalPostprocessor and left empty
    // by every pure-embedding model. They live HERE, rather than in a RetrievalResult
    // of their own, because IEmbeddingFactory and IVisualizer are templated on this
    // exact type: a derived struct would be sliced by the runner's
    // std::vector<EmbeddingResult>, and a separate type would need a parallel factory
    // interface plus its own sync and async runner for no behavioural gain.
    // (The Python tree can afford a distinct RetrievalResult because its runner is
    // duck-typed and its verify serializer dispatches on the class name; the JSON both
    // trees emit is the same either way.)
    std::vector<GalleryMatch> matches;   // ranked best-first
    int gallery_size{0};
    std::string gallery_name;

    EmbeddingResult() = default;
};

/**
 * @brief Face alignment / 3D face reconstruction result
 */
struct FaceAlignmentResult {
    std::vector<Keypoint> landmarks_2d;  // [68] projected 2D landmarks
    std::vector<float> pose;             // [yaw, pitch, roll] in degrees
    std::vector<float> params;           // raw 3DMM parameters (optional)

    FaceAlignmentResult() = default;
};

/**
 * @brief Hand landmark detection result
 */
struct HandLandmarkResult {
    std::vector<Keypoint> landmarks;  // [21] hand landmarks
    float confidence{0.0f};           // hand presence confidence
    std::string handedness;           // "Left" or "Right"

    HandLandmarkResult() = default;
};

/**
 * @brief Oriented Bounding Box detection result
 * 
 * Uses center-based representation: [cx, cy, w, h, angle]
 * where angle is the rotation angle in radians.
 * 4 corners can be computed from these values.
 */
struct OBBResult {
    float cx{0.0f};          // Center x coordinate
    float cy{0.0f};          // Center y coordinate
    float width{0.0f};       // Box width (before rotation)
    float height{0.0f};      // Box height (before rotation)
    float angle{0.0f};       // Rotation angle in radians
    float confidence{0.0f};
    int class_id{0};
    std::string class_name;

    OBBResult() = default;

    /**
     * @brief Compute 4 corner points of the rotated bounding box
     * @return Vector of 4 corner points (p1, p2, p3, p4)
     */
    std::vector<cv::Point2f> getCorners() const {
        float cos_v = std::cos(angle);
        float sin_v = std::sin(angle);
        cv::Point2f ctr(cx, cy);
        cv::Point2f vec1(width * 0.5f * cos_v, width * 0.5f * sin_v);
        cv::Point2f vec2(-height * 0.5f * sin_v, height * 0.5f * cos_v);
        return {ctr + vec1 + vec2, ctr + vec1 - vec2,
                ctr - vec1 - vec2, ctr - vec1 + vec2};
    }
};

/**
 * @brief 3D object detection result (SFA3D / KITTI-like)
 */
struct Detection3DResult {
    int class_id{0};
    std::string class_name;
    float confidence{0.0f};
    float bev_x{0.0f};
    float bev_y{0.0f};
    float bev_w{0.0f};
    float bev_h{0.0f};
    float x3d{0.0f};
    float y3d{0.0f};
    float z3d{0.0f};
    float dim_h{0.0f};
    float dim_w{0.0f};
    float dim_l{0.0f};
    float yaw{0.0f};
};

/**
 * @brief Abstract interface for postprocessors
 *
 * Postprocessors transform model outputs into usable detection/segmentation results.
 * This is a template interface to support different result types.
 */
template <typename ResultType>
class IPostprocessor {
public:
    virtual ~IPostprocessor() = default;

    /**
     * @brief Process model outputs into results
     * @param outputs Model output tensors
     * @param ctx Preprocessing context for coordinate transformation
     * @return Vector of processed results
     */
    virtual std::vector<ResultType> process(const dxrt::TensorPtrs& outputs, 
                                            const PreprocessContext& ctx) = 0;

    /**
     * @brief Get the model name this postprocessor is designed for
     * @return Model name string
     */
    virtual std::string getModelName() const = 0;
};

// Type aliases for common postprocessor types
using DetectionPostprocessor = IPostprocessor<DetectionResult>;
using SegmentationPostprocessor = IPostprocessor<SegmentationResult>;
using ClassificationPostprocessor = IPostprocessor<ClassificationResult>;
using FaceDetectionPostprocessor = IPostprocessor<FaceDetectionResult>;
using PosePostprocessor = IPostprocessor<PoseResult>;
using InstanceSegmentationPostprocessor = IPostprocessor<InstanceSegmentationResult>;
using OBBPostprocessor = IPostprocessor<OBBResult>;
using Detection3DPostprocessor = IPostprocessor<Detection3DResult>;
using DepthPostprocessor = IPostprocessor<DepthResult>;
using RestorationPostprocessor = IPostprocessor<RestorationResult>;
using EmbeddingPostprocessorBase = IPostprocessor<EmbeddingResult>;
using FaceAlignmentPostprocessorBase = IPostprocessor<FaceAlignmentResult>;
using HandLandmarkPostprocessorBase = IPostprocessor<HandLandmarkResult>;

// Smart pointer aliases
using PreprocessorPtr = std::unique_ptr<IPreprocessor>;
template <typename T>
using PostprocessorPtr = std::unique_ptr<IPostprocessor<T>>;

}  // namespace dxapp

#endif  // DXAPP_I_PROCESSOR_HPP

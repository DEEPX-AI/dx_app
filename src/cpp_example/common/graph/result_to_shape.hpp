/**
 * @file result_to_shape.hpp
 * @brief Convert the framework's 13 result types into engine payloads.
 *
 * Interface -> output shape is one-to-one, so these overloads are mechanical.
 * They are the only place that knows a concrete result type.
 *
 * ToStagePorts(results) builds a model's extra output ports (U-08), one
 * overload per result type that has any. Each returns every port it knows,
 * always non-null, even when nothing was decoded (the PORTS CONTRACT);
 * TypedStage keeps only the names the model declares.
 *
 * AnomalyResult has no conversion on purpose: anomaly models are registered
 * not-ready this release (R9, scripts/gen_model_registry.py), so no stage is
 * ever built for one.
 */
#ifndef DXAPP_GRAPH_RESULT_TO_SHAPE_HPP
#define DXAPP_GRAPH_RESULT_TO_SHAPE_HPP

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "common/graph/shape.hpp"

namespace dxapp {
namespace graph {

namespace detail {

/// [x1,y1,x2,y2] -> cv::Rect2f. Returns an empty rect for a malformed box.
inline cv::Rect2f ToRect(const std::vector<float>& box) {
    if (box.size() < 4) return cv::Rect2f();
    return cv::Rect2f(box[0], box[1], box[2] - box[0], box[3] - box[1]);
}

}  // namespace detail

inline StageDataPtr ToStageData(const std::vector<DetectionResult>& results) {
    std::shared_ptr<BoxesData> data(new BoxesData(Shape::kBoxes));
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        BoxItem item;
        item.box = detail::ToRect(results[i].box);
        item.score = results[i].confidence;
        item.class_id = results[i].class_id;
        item.class_name = results[i].class_name;
        data->items.push_back(item);
    }
    return data;
}

inline StageDataPtr ToStageData(const std::vector<FaceDetectionResult>& results) {
    std::shared_ptr<BoxesData> data(new BoxesData(Shape::kBoxes));
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        BoxItem item;
        item.box = detail::ToRect(results[i].box);
        item.score = results[i].confidence;
        item.class_id = 0;
        item.class_name = results[i].class_name;
        item.landmarks = results[i].landmarks;
        data->items.push_back(item);
    }
    return data;
}

inline StageDataPtr ToStageData(const std::vector<OBBResult>& results) {
    std::shared_ptr<BoxesData> data(new BoxesData(Shape::kObBoxes));
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        const OBBResult& obb = results[i];
        BoxItem item;
        item.box = cv::Rect2f(obb.cx - obb.width * 0.5f,
                              obb.cy - obb.height * 0.5f,
                              obb.width, obb.height);
        item.angle = obb.angle;
        item.score = obb.confidence;
        item.class_id = obb.class_id;
        item.class_name = obb.class_name;
        data->items.push_back(item);
    }
    return data;
}

inline StageDataPtr ToStageData(const std::vector<InstanceSegmentationResult>& results) {
    std::shared_ptr<BoxesData> data(new BoxesData(Shape::kInstances));
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        BoxItem item;
        item.box = detail::ToRect(results[i].box);
        item.score = results[i].confidence;
        item.class_id = results[i].class_id;
        item.class_name = results[i].class_name;
        item.mask = results[i].mask;
        data->items.push_back(item);
    }
    return data;
}

/// YOLOPv2 (SP3): one PanopticResult per frame. The primary payload is its
/// boxes, converted exactly like a DetectionResult; the masks are ports.
inline StageDataPtr ToStageData(const std::vector<PanopticResult>& results) {
    return ToStageData(results.empty() ? std::vector<DetectionResult>() : results[0].detections);
}

inline StagePorts ToStagePorts(const std::vector<PanopticResult>& results) {
    // BGR colours of the YOLOPv2 runner's visualizer.
    std::shared_ptr<LabelMapData> drivable(new LabelMapData());
    drivable->binary_mask = true;
    drivable->mask_color = cv::Vec3b(0, 180, 0);
    std::shared_ptr<LabelMapData> lane(new LabelMapData());
    lane->binary_mask = true;
    lane->mask_color = cv::Vec3b(0, 0, 200);
    if (!results.empty()) {
        drivable->labels = results[0].drivable;
        lane->labels = results[0].lane;
    }
    StagePorts ports;
    ports["drivable"] = drivable;
    ports["lane"] = lane;
    return ports;
}

inline StageDataPtr ToStageData(const std::vector<PoseResult>& results) {
    std::shared_ptr<KeypointsData> data(new KeypointsData());
    data->items = results;
    // SuperPoint's descriptors travel in the "descriptors" port; the
    // keypoints payload does not hold on to them. Keypoints that come with
    // descriptors are SuperPoint's: unordered, whatever their count.
    for (std::size_t i = 0; i < data->items.size(); ++i) {
        if (data->items[i].descriptors) data->unordered = true;
        data->items[i].descriptors.reset();
    }
    return data;
}

/// SuperPoint: row r = the r-th keypoint across the frame's items, in order.
/// PoseResult::descriptors is a shared, immutable set (8d0b748's design: the
/// visualizer's tracker reads the same set); an item without one - every
/// pose model but SuperPoint - adds no rows, so such a model yields an empty
/// matrix.
inline StagePorts ToStagePorts(const std::vector<PoseResult>& results) {
    static const std::vector<std::vector<float> > kNone;
    std::size_t rows = 0;
    std::size_t cols = 0;
    bool sized = false;
    for (std::size_t i = 0; i < results.size(); ++i) {
        const std::vector<std::vector<float> >& set =
            results[i].descriptors ? *results[i].descriptors : kNone;
        for (std::size_t d = 0; d < set.size(); ++d) {
            const std::size_t length = set[d].size();
            if (!sized) {
                cols = length;
                sized = true;
            } else if (length != cols) {
                throw std::runtime_error("SuperPoint descriptors have different lengths");
            }
            ++rows;
        }
    }
    std::shared_ptr<DenseMapData> data(new DenseMapData());
    data->spatial = false;
    if (rows > 0 && cols > 0) {
        data->values.create(static_cast<int>(rows), static_cast<int>(cols), CV_32F);
        int row = 0;
        for (std::size_t i = 0; i < results.size(); ++i) {
            if (!results[i].descriptors) continue;
            const std::vector<std::vector<float> >& set = *results[i].descriptors;
            for (std::size_t d = 0; d < set.size(); ++d, ++row) {
                const std::vector<float>& values = set[d];
                std::copy(values.begin(), values.end(), data->values.ptr<float>(row));
            }
        }
    }
    StagePorts ports;
    ports["descriptors"] = data;
    return ports;
}

inline StageDataPtr ToStageData(const std::vector<HandLandmarkResult>& results) {
    std::shared_ptr<KeypointsData> data(new KeypointsData());
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        PoseResult pose;
        pose.confidence = results[i].confidence;
        pose.keypoints = results[i].landmarks;
        data->items.push_back(pose);
    }
    return data;
}

/// One score per hand: Left 0, Right 1, anything else (Unknown) -1.
inline StagePorts ToStagePorts(const std::vector<HandLandmarkResult>& results) {
    std::shared_ptr<ScoresData> data(new ScoresData());
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        ClassificationResult item;
        item.class_name = results[i].handedness;
        item.class_id = results[i].handedness == "Right" ? 1
                        : results[i].handedness == "Left" ? 0 : -1;
        item.confidence = results[i].confidence;
        data->items.push_back(item);
    }
    StagePorts ports;
    ports["handedness"] = data;
    return ports;
}

inline StageDataPtr ToStageData(const std::vector<FaceAlignmentResult>& results) {
    std::shared_ptr<KeypointsData> data(new KeypointsData());
    data->items.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
        PoseResult pose;
        pose.keypoints = results[i].landmarks_2d;
        data->items.push_back(pose);
    }
    return data;
}

/// Every face's [yaw, pitch, roll] (degrees), concatenated in item order.
inline StagePorts ToStagePorts(const std::vector<FaceAlignmentResult>& results) {
    std::shared_ptr<VectorData> data(new VectorData());
    for (std::size_t i = 0; i < results.size(); ++i) {
        data->values.insert(data->values.end(), results[i].pose.begin(), results[i].pose.end());
    }
    StagePorts ports;
    ports["pose"] = data;
    return ports;
}

inline StageDataPtr ToStageData(const SegmentationResult& result) {
    std::shared_ptr<LabelMapData> data(new LabelMapData());
    // SegmentationResult::mask is a flattened vector<int> of H*W class ids
    // (not a cv::Mat) — reassemble it into the engine's dense payload.
    if (result.width > 0 && result.height > 0 &&
        result.mask.size() ==
            static_cast<std::size_t>(result.width) * static_cast<std::size_t>(result.height)) {
        cv::Mat labels(result.height, result.width, CV_32S,
                       const_cast<int*>(result.mask.data()));
        data->labels = labels.clone();
    }
    return data;
}

/// PP-Matting: the soft alpha matte in [0, 1], as a spatial CV_32F map (a
/// copy). A class-map model leaves SegmentationResult::alpha empty, and so
/// is the port.
inline StagePorts ToStagePorts(const std::vector<SegmentationResult>& results) {
    std::shared_ptr<DenseMapData> data(new DenseMapData());
    if (!results.empty() && !results[0].alpha.empty()) {
        results[0].alpha.convertTo(data->values, CV_32F);
    }
    StagePorts ports;
    ports["alpha"] = data;
    return ports;
}

/// A copy: the payload outlives the stage's output tensors, and a
/// postprocessor's depth_map may still point into one (8d0b748's
/// FastDepthPostprocessor normalizes in place, inside the dxrt buffer).
inline StageDataPtr ToStageData(const DepthResult& result) {
    std::shared_ptr<DenseMapData> data(new DenseMapData());
    data->values = result.depth_map.clone();
    return data;
}

inline StageDataPtr ToStageData(const RestorationResult& result) {
    std::shared_ptr<ImageData> data(new ImageData());
    data->image = result.restored_image;
    return data;
}

inline StageDataPtr ToStageData(const std::vector<ClassificationResult>& results) {
    std::shared_ptr<ScoresData> data(new ScoresData());
    data->items = results;
    return data;
}

inline StageDataPtr ToStageData(const EmbeddingResult& result) {
    std::shared_ptr<VectorData> data(new VectorData());
    data->values = result.embedding;
    return data;
}

/// A gallery model's ranking (GalleryRetrievalPostprocessor), best first:
/// one score per match, class_id = its rank, confidence = its cosine score,
/// class_name = its gallery label, or its image path when unlabelled.
inline StagePorts ToStagePorts(const std::vector<EmbeddingResult>& results) {
    std::shared_ptr<ScoresData> data(new ScoresData());
    if (!results.empty()) {
        const std::vector<GalleryMatch>& matches = results[0].matches;
        data->items.reserve(matches.size());
        for (std::size_t i = 0; i < matches.size(); ++i) {
            ClassificationResult item;
            item.class_id = matches[i].rank;
            item.confidence = matches[i].score;
            item.class_name = matches[i].label.empty() ? matches[i].path : matches[i].label;
            data->items.push_back(item);
        }
    }
    StagePorts ports;
    ports["matches"] = data;
    return ports;
}

inline StageDataPtr ToStageData(const std::vector<Detection3DResult>& results) {
    std::shared_ptr<Boxes3dData> data(new Boxes3dData());
    data->items = results;
    return data;
}

}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_GRAPH_RESULT_TO_SHAPE_HPP

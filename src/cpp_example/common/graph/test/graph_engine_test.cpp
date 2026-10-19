/**
 * @file graph_engine_test.cpp
 * @brief Plain-assert test binary for the graph engine.
 *
 * This repository has no gtest and no CTest; C++ behaviour is exercised by
 * pytest driving built binaries. This binary prints one line per case and
 * returns non-zero when any case fails.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "common/graph/graph_config.hpp"
#include "common/graph/graph_error.hpp"
#include "common/graph/graph_runner_async.hpp"
#include "common/graph/graph_runner_sync.hpp"
#include "common/graph/graph_visualizer.hpp"
#include "common/graph/logic_ops.hpp"
#include "common/graph/roi_router.hpp"
#include "common/graph/shape.hpp"
#include "common/graph/result_to_shape.hpp"
#include "common/graph/stage_graph.hpp"
#include "common/graph/test/fake_model_registry.hpp"
#include "common/processors/hand_landmark_postprocessor.hpp"
#include "common/processors/mediapipe_hand_postprocessor.hpp"
#include "common/utility/visualization.hpp"
#include "common/visualizers/segmentation_visualizer.hpp"

// Task 11. The concrete registry and the factory bridge live OUTSIDE
// common/graph/ (see their own headers for why); this fixture directory
// is the one place under common/graph/ the boundary guard exempts, and
// it is exempt precisely so a test may name a concrete registry.
#include "common/registry/static_model_registry.hpp"
#include "common/registry/typed_stage.hpp"
#include "visual_place_recognition/eigenplaces/eigenplaces-resnet18_512x512/factory/eigenplaces-resnet18_512x512_factory.hpp"
#include "common/utility/dxnn_container.hpp"
#include "common/utility/repo_path.hpp"
#include "multi_model_graph/graph_cli_output.hpp"
#include "multi_model_graph/graph_consumer.hpp"
#include "multi_model_graph/graph_report_writer.hpp"
#include "common/graph/cpu_reduce.hpp"
#include "common/graph/user_logic.hpp"
#include "multi_model_graph/graph_cli.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL %s:%d  %s\n", file, line, expr);
    }
}

}  // namespace

#define GRAPH_CHECK(cond) Check((cond), #cond, __FILE__, __LINE__)

namespace dxapp {
namespace graph {

void TestShapeVocabulary() {
    GRAPH_CHECK(std::string(ToString(Shape::kBoxes)) == "boxes");
    GRAPH_CHECK(std::string(ToString(Shape::kObBoxes)) == "obboxes");
    GRAPH_CHECK(std::string(ToString(Shape::kLabelMap)) == "labelmap");

    GRAPH_CHECK(ProducesRoi(Shape::kBoxes));
    GRAPH_CHECK(ProducesRoi(Shape::kObBoxes));
    GRAPH_CHECK(ProducesRoi(Shape::kInstances));
    GRAPH_CHECK(!ProducesRoi(Shape::kLabelMap));
    GRAPH_CHECK(!ProducesRoi(Shape::kDenseMap));
    GRAPH_CHECK(!ProducesRoi(Shape::kKeypoints));
    GRAPH_CHECK(!ProducesRoi(Shape::kFrame));

    GRAPH_CHECK(AcceptsRoi(InputContract::kRoi));
    GRAPH_CHECK(AcceptsRoi(InputContract::kEither));
    GRAPH_CHECK(!AcceptsRoi(InputContract::kFullFrame));
}

void TestDetectionConversion() {
    std::vector<DetectionResult> detections;
    std::vector<float> box;
    box.push_back(10.f); box.push_back(20.f); box.push_back(110.f); box.push_back(140.f);
    detections.push_back(DetectionResult(box, 0.9f, 3, "person"));

    StageDataPtr data = ToStageData(detections);
    GRAPH_CHECK(data->shape() == Shape::kBoxes);

    const BoxesData* boxes = dynamic_cast<const BoxesData*>(data.get());
    GRAPH_CHECK(boxes != NULL);
    GRAPH_CHECK(boxes->items.size() == 1);
    GRAPH_CHECK(boxes->items[0].class_name == "person");
    GRAPH_CHECK(boxes->items[0].class_id == 3);
    GRAPH_CHECK(boxes->items[0].track_id == -1);
    GRAPH_CHECK(boxes->items[0].box.x == 10.f);
    GRAPH_CHECK(boxes->items[0].box.width == 100.f);
    GRAPH_CHECK(boxes->items[0].box.height == 120.f);
    GRAPH_CHECK(boxes->items[0].landmarks.empty());
}

void TestFaceConversionKeepsLandmarks() {
    std::vector<FaceDetectionResult> faces;
    FaceDetectionResult face;
    face.box.push_back(0.f); face.box.push_back(0.f);
    face.box.push_back(64.f); face.box.push_back(64.f);
    face.confidence = 0.8f;
    for (int i = 0; i < 5; ++i) {
        Keypoint point;
        point.x = static_cast<float>(i);
        point.y = static_cast<float>(i * 2);
        face.landmarks.push_back(point);
    }
    faces.push_back(face);

    StageDataPtr data = ToStageData(faces);
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(data.get());
    GRAPH_CHECK(boxes != NULL);
    GRAPH_CHECK(data->shape() == Shape::kBoxes);
    GRAPH_CHECK(boxes->items[0].landmarks.size() == 5);
    GRAPH_CHECK(boxes->items[0].landmarks[4].y == 8.f);
}

void TestObbConversionKeepsAngle() {
    std::vector<OBBResult> obbs;
    OBBResult obb;
    obb.cx = 100.f; obb.cy = 200.f;
    obb.width = 40.f; obb.height = 20.f;
    obb.angle = 0.5f; obb.confidence = 0.7f;
    obb.class_id = 1; obb.class_name = "ship";
    obbs.push_back(obb);

    StageDataPtr data = ToStageData(obbs);
    GRAPH_CHECK(data->shape() == Shape::kObBoxes);
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(data.get());
    GRAPH_CHECK(boxes != NULL);
    GRAPH_CHECK(boxes->items[0].angle == 0.5f);
    // Axis-aligned bounds of the un-rotated rect, centred on (cx, cy).
    GRAPH_CHECK(boxes->items[0].box.x == 80.f);
    GRAPH_CHECK(boxes->items[0].box.y == 190.f);
}

void TestEmbeddingConversion() {
    EmbeddingResult embedding;
    embedding.embedding.push_back(0.1f);
    embedding.embedding.push_back(0.2f);

    StageDataPtr data = ToStageData(embedding);
    GRAPH_CHECK(data->shape() == Shape::kVector);
    const VectorData* vector = dynamic_cast<const VectorData*>(data.get());
    GRAPH_CHECK(vector != NULL);
    GRAPH_CHECK(vector->values.size() == 2);
}

void TestRoiRefDefaultsToIdentity() {
    RoiRef ref;
    GRAPH_CHECK(!ref.from_roi);
    GRAPH_CHECK(ref.track_id == -1);
    GRAPH_CHECK(ref.inv_align(0, 0) == 1.f);
    GRAPH_CHECK(ref.inv_align(0, 1) == 0.f);
    GRAPH_CHECK(ref.inv_align(1, 1) == 1.f);
    GRAPH_CHECK(ref.inv_align(0, 2) == 0.f);
}

void TestInstanceSegmentationConversion() {
    std::vector<InstanceSegmentationResult> results;
    InstanceSegmentationResult seg;
    seg.box.push_back(1.f); seg.box.push_back(2.f);
    seg.box.push_back(11.f); seg.box.push_back(22.f);
    seg.confidence = 0.6f;
    seg.class_id = 5;
    seg.class_name = "cat";
    seg.mask = cv::Mat::ones(2, 2, CV_8U) * 9;
    seg.track_id = 7;  // model-internal id space; must not leak into the graph's

    results.push_back(seg);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kInstances);
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(data.get());
    GRAPH_CHECK(boxes != NULL);
    GRAPH_CHECK(boxes->items.size() == 1);
    GRAPH_CHECK(boxes->items[0].class_id == 5);
    GRAPH_CHECK(boxes->items[0].class_name == "cat");
    // The graph owns tracking as a node attribute (a later task assigns ids
    // over BoxItem); a model's own internal track_id is a different id space
    // and must stay out of BoxItem::track_id here.
    GRAPH_CHECK(boxes->items[0].track_id == -1);
    GRAPH_CHECK(boxes->items[0].mask.rows == 2);
    GRAPH_CHECK(boxes->items[0].mask.cols == 2);
    GRAPH_CHECK(boxes->items[0].mask.at<uchar>(0, 0) == 9);
}

void TestPoseConversionIsDirectCopy() {
    std::vector<PoseResult> results;
    PoseResult pose;
    pose.confidence = 0.55f;
    Keypoint kp1(1.f, 2.f, 0.9f);
    Keypoint kp2(3.f, 4.f, 0.8f);
    pose.keypoints.push_back(kp1);
    pose.keypoints.push_back(kp2);
    results.push_back(pose);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kKeypoints);
    const KeypointsData* kps = dynamic_cast<const KeypointsData*>(data.get());
    GRAPH_CHECK(kps != NULL);
    GRAPH_CHECK(kps->items.size() == 1);
    GRAPH_CHECK(kps->items[0].keypoints.size() == 2);
    GRAPH_CHECK(kps->items[0].keypoints[1].x == 3.f);
    GRAPH_CHECK(kps->items[0].confidence == 0.55f);
}

void TestHandLandmarkConversion() {
    std::vector<HandLandmarkResult> results;
    HandLandmarkResult hand;
    hand.confidence = 0.77f;
    hand.handedness = "Left";
    for (int i = 0; i < 3; ++i) {
        Keypoint kp;
        kp.x = static_cast<float>(i);
        kp.y = static_cast<float>(i + 1);
        hand.landmarks.push_back(kp);
    }
    results.push_back(hand);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kKeypoints);
    const KeypointsData* kps = dynamic_cast<const KeypointsData*>(data.get());
    GRAPH_CHECK(kps != NULL);
    GRAPH_CHECK(kps->items.size() == 1);
    GRAPH_CHECK(kps->items[0].confidence == 0.77f);
    GRAPH_CHECK(kps->items[0].keypoints.size() == 3);
    GRAPH_CHECK(kps->items[0].keypoints[2].y == 3.f);
}

/// A float tensor over `values`, shaped [1, n] or [1, n, k] - the layouts
/// dxrt hands the MediaPipe hand models' postprocessors.
dxrt::TensorPtr FloatTensor(const char* name, std::vector<float>& values,
                            const std::vector<int64_t>& shape) {
    return std::make_shared<dxrt::Tensor>(name, shape, dxrt::DataType::FLOAT,
                                          values.data());
}

/// The palm detector decodes into FaceDetectionResult, but what it finds is
/// a hand: the graph's boxes (report class_name, drawn label) say so.
void TestPalmDetectorBoxesAreNamedHand() {
    const int kAnchors = 2016;
    std::vector<float> reg(static_cast<std::size_t>(kAnchors) * 18, 0.f);
    std::vector<float> scores(kAnchors, -10.f);
    const int centre = (12 * 24 + 12) * 2;  // stride-8 cell (12, 12)
    reg[centre * 18 + 2] = 40.f;
    reg[centre * 18 + 3] = 40.f;
    reg[centre * 18 + 4 + 5] = -10.f;  // middle-finger MCP above the wrist
    scores[centre] = 5.f;
    std::vector<int64_t> reg_shape;
    reg_shape.push_back(1); reg_shape.push_back(kAnchors); reg_shape.push_back(18);
    std::vector<int64_t> score_shape;
    score_shape.push_back(1); score_shape.push_back(kAnchors); score_shape.push_back(1);
    dxrt::TensorPtrs outputs;
    outputs.push_back(FloatTensor("regressors", reg, reg_shape));
    outputs.push_back(FloatTensor("classificators", scores, score_shape));

    PreprocessContext ctx;
    ctx.original_width = 192;
    ctx.original_height = 192;
    ctx.input_width = 192;
    ctx.input_height = 192;
    MediaPipeHandPostprocessor palm(192, 0.5f, 0.3f);
    const std::vector<FaceDetectionResult> hands = palm.process(outputs, ctx);
    GRAPH_CHECK(hands.size() == 1);

    StageDataPtr data = ToStageData(hands);
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(data.get());
    GRAPH_CHECK(boxes != NULL);
    if (boxes == NULL || boxes->items.size() != 1) return;
    GRAPH_CHECK(boxes->items[0].class_name == "hand");

    // A real face detector's boxes are still faces.
    std::vector<FaceDetectionResult> faces(1);
    faces[0].box.push_back(0.f); faces[0].box.push_back(0.f);
    faces[0].box.push_back(10.f); faces[0].box.push_back(10.f);
    StageDataPtr face_data = ToStageData(faces);
    const BoxesData* face_boxes = dynamic_cast<const BoxesData*>(face_data.get());
    GRAPH_CHECK(face_boxes != NULL && face_boxes->items.size() == 1 &&
                face_boxes->items[0].class_name == "face");
}

/// Brightest channel in the 5x5 window around `point`: lines are drawn
/// anti-aliased, so one exact pixel is too strict a probe.
int PeakAround(const cv::Mat& canvas, const cv::Point& point) {
    const cv::Rect window = cv::Rect(point.x - 2, point.y - 2, 5, 5) &
                            cv::Rect(0, 0, canvas.cols, canvas.rows);
    double peak = 0.0;
    cv::minMaxLoc(canvas(window).reshape(1), NULL, &peak);
    return static_cast<int>(peak);
}

/// hand_cascade's landmark node, end to end: HandLandmarkLite's decoded
/// output (z negative, as on the NPU) through the graph payload onto the
/// canvas. All 21 points and the hand's own bones are drawn; COCO body limbs
/// between hand indices are not.
void TestHandLandmarksAreDrawnWithTheHandSkeleton() {
    // Wrist at the bottom, five vertical fingers: MCP, PIP, DIP, TIP.
    std::vector<cv::Point2f> points;
    points.push_back(cv::Point2f(112.f, 200.f));
    points.push_back(cv::Point2f(70.f, 180.f));
    points.push_back(cv::Point2f(50.f, 160.f));
    points.push_back(cv::Point2f(35.f, 140.f));
    points.push_back(cv::Point2f(25.f, 120.f));
    const float finger_x[] = {76.f, 112.f, 148.f, 184.f};
    const float joint_y[] = {150.f, 115.f, 85.f, 55.f};
    for (int f = 0; f < 4; ++f) {
        for (int j = 0; j < 4; ++j) points.push_back(cv::Point2f(finger_x[f], joint_y[j]));
    }
    std::vector<float> landmarks;
    for (std::size_t i = 0; i < points.size(); ++i) {
        landmarks.push_back(points[i].x);
        landmarks.push_back(points[i].y);
        landmarks.push_back(-3.f - static_cast<float>(i));  // relative depth
    }
    std::vector<float> presence(1, 0.98f);
    std::vector<float> handedness(1, 0.2f);
    std::vector<float> world(63, 0.01f);
    std::vector<int64_t> flat63;
    flat63.push_back(1); flat63.push_back(63);
    std::vector<int64_t> flat1;
    flat1.push_back(1); flat1.push_back(1);
    dxrt::TensorPtrs outputs;
    outputs.push_back(FloatTensor("Identity", landmarks, flat63));
    outputs.push_back(FloatTensor("Identity_1", presence, flat1));
    outputs.push_back(FloatTensor("Identity_2", handedness, flat1));
    outputs.push_back(FloatTensor("Identity_3", world, flat63));

    PreprocessContext ctx;
    ctx.original_width = 224;
    ctx.original_height = 224;
    HandLandmarkPostprocessor post(224, 224);
    const std::vector<HandLandmarkResult> hands = post.process(outputs, ctx);
    GRAPH_CHECK(hands.size() == 1);

    FrameReport report;
    StageResult landmark;
    landmark.data = ToStageData(hands);
    report.node_results["landmark"] = landmark;
    const cv::Mat source = cv::Mat::zeros(224, 224, CV_8UC3);
    const cv::Mat canvas = RenderReport(source, report);

    int drawn_points = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (PeakAround(canvas, cv::Point(cvRound(points[i].x), cvRound(points[i].y))) > 100) {
            ++drawn_points;
        }
    }
    GRAPH_CHECK(drawn_points == 21);
    // Hand bones: wrist -> index MCP, index MCP -> middle MCP (palm).
    GRAPH_CHECK(PeakAround(canvas, cv::Point(94, 175)) > 100);
    GRAPH_CHECK(PeakAround(canvas, cv::Point(94, 150)) > 100);
    // COCO's left shoulder -> left hip (5 -> 11) is not a hand bone.
    GRAPH_CHECK(PeakAround(canvas, cv::Point(94, 118)) == 0);
}

/// SuperPoint's keypoints come with descriptors: in a graph they are drawn
/// as points only - all of them, however low their score - even when there
/// happen to be 21 (the hand's count) or 17 (COCO's).
void TestSuperPointKeypointsAreDrawnAsPointsOnly() {
    const std::size_t counts[] = {17, 21};
    for (std::size_t c = 0; c < 2; ++c) {
        PoseResult pose;
        pose.confidence = 1.f;
        std::shared_ptr<std::vector<std::vector<float> > > descriptors(
            new std::vector<std::vector<float> >());
        for (std::size_t i = 0; i < counts[c]; ++i) {
            // A 5 x 5 grid, 40 px apart: no two points touch.
            pose.keypoints.push_back(Keypoint(20.f + 40.f * static_cast<float>(i % 5),
                                              20.f + 40.f * static_cast<float>(i / 5), 0.05f));
            descriptors->push_back(std::vector<float>(4, 0.5f));
        }
        pose.descriptors = descriptors;
        StageDataPtr data = ToStageData(std::vector<PoseResult>(1, pose));
        const KeypointsData* kps = dynamic_cast<const KeypointsData*>(data.get());
        GRAPH_CHECK(kps != NULL && kps->unordered);

        FrameReport report;
        StageResult points;
        points.data = data;
        report.node_results["superpoint"] = points;
        const cv::Mat canvas = RenderReport(cv::Mat::zeros(224, 224, CV_8UC3), report);
        int drawn = 0;
        for (std::size_t i = 0; i < pose.keypoints.size(); ++i) {
            const cv::Point at(cvRound(pose.keypoints[i].x), cvRound(pose.keypoints[i].y));
            if (PeakAround(canvas, at) > 100) ++drawn;
        }
        GRAPH_CHECK(drawn == static_cast<int>(counts[c]));
        GRAPH_CHECK(PeakAround(canvas, cv::Point(40, 20)) == 0);  // 0 -> 1: COCO and hand bone
        GRAPH_CHECK(PeakAround(canvas, cv::Point(20, 40)) == 0);  // 0 -> 5: hand bone
    }
    // A pose model's keypoints (no descriptors) stay ordered.
    PoseResult body;
    body.keypoints.push_back(Keypoint(1.f, 1.f, 0.9f));
    StageDataPtr body_data = ToStageData(std::vector<PoseResult>(1, body));
    const KeypointsData* body_kps = dynamic_cast<const KeypointsData*>(body_data.get());
    GRAPH_CHECK(body_kps != NULL && !body_kps->unordered);
}

/// A face alignment node's 68 landmarks are drawn with the face's outline
/// (jaw, brows, nose, eyes, lips), not with COCO body limbs.
void TestFaceAlignmentLandmarksAreDrawnWithTheFaceOutline() {
    std::vector<FaceAlignmentResult> results(1);
    for (int i = 0; i < 68; ++i) {
        // 68 points, 20 px apart on a circle of radius 220 around (240, 240).
        const double angle = 2.0 * 3.14159265358979 * i / 68.0;
        results[0].landmarks_2d.push_back(Keypoint(static_cast<float>(240.0 + 220.0 * std::cos(angle)),
                                                   static_cast<float>(240.0 + 220.0 * std::sin(angle))));
    }
    const std::vector<Keypoint>& lm = results[0].landmarks_2d;
    FrameReport report;
    StageResult face;
    face.data = ToStageData(results);
    report.node_results["align"] = face;
    const cv::Mat canvas = RenderReport(cv::Mat::zeros(480, 480, CV_8UC3), report);
    const cv::Point lips((cvRound(lm[59].x) + cvRound(lm[48].x)) / 2,
                         (cvRound(lm[59].y) + cvRound(lm[48].y)) / 2);
    const cv::Point limb((cvRound(lm[5].x) + cvRound(lm[11].x)) / 2,
                         (cvRound(lm[5].y) + cvRound(lm[11].y)) / 2);
    GRAPH_CHECK(PeakAround(canvas, lips) > 100);  // outer lip closes 59 -> 48
    GRAPH_CHECK(PeakAround(canvas, limb) == 0);   // COCO's 5 -> 11 is not drawn
}

void TestFaceAlignmentConversion() {
    std::vector<FaceAlignmentResult> results;
    FaceAlignmentResult align;
    for (int i = 0; i < 4; ++i) {
        Keypoint kp;
        kp.x = static_cast<float>(i * 2);
        kp.y = static_cast<float>(i * 3);
        align.landmarks_2d.push_back(kp);
    }
    align.pose.push_back(1.f); align.pose.push_back(2.f); align.pose.push_back(3.f);
    results.push_back(align);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kKeypoints);
    const KeypointsData* kps = dynamic_cast<const KeypointsData*>(data.get());
    GRAPH_CHECK(kps != NULL);
    GRAPH_CHECK(kps->items.size() == 1);
    GRAPH_CHECK(kps->items[0].keypoints.size() == 4);
    GRAPH_CHECK(kps->items[0].keypoints[3].x == 6.f);
}

void TestSegmentationConversionCatchesTransposition() {
    // Non-square (width != height) so a row/column swap would be caught: a
    // transposed implementation would produce a 4x3 Mat instead of 3x4, and
    // element (row, col) would read the wrong source value.
    SegmentationResult result;
    result.width = 4;
    result.height = 3;
    for (int i = 0; i < 12; ++i) {
        result.mask.push_back(i);  // row-major: row = i / width, col = i % width
    }

    StageDataPtr data = ToStageData(result);
    GRAPH_CHECK(data->shape() == Shape::kLabelMap);
    const LabelMapData* label_map = dynamic_cast<const LabelMapData*>(data.get());
    GRAPH_CHECK(label_map != NULL);
    GRAPH_CHECK(label_map->labels.rows == 3);
    GRAPH_CHECK(label_map->labels.cols == 4);
    // row 1, col 2 -> flattened index 1*4 + 2 = 6
    GRAPH_CHECK(label_map->labels.at<int>(1, 2) == 6);
    // row 2, col 0 -> flattened index 2*4 + 0 = 8
    GRAPH_CHECK(label_map->labels.at<int>(2, 0) == 8);
    // row 0, col 3 -> flattened index 3
    GRAPH_CHECK(label_map->labels.at<int>(0, 3) == 3);
}

void TestSegmentationConversionDefensiveSizeCheck() {
    // mask.size() != width * height: the malformed-input path must not crash
    // and must not fabricate a Mat from a mismatched buffer.
    SegmentationResult result;
    result.width = 4;
    result.height = 3;
    result.mask.push_back(1);
    result.mask.push_back(2);  // only 2 elements, not 12

    StageDataPtr data = ToStageData(result);
    const LabelMapData* label_map = dynamic_cast<const LabelMapData*>(data.get());
    GRAPH_CHECK(label_map != NULL);
    GRAPH_CHECK(label_map->labels.empty());
}

void TestDepthConversion() {
    DepthResult result;
    result.depth_map = cv::Mat(2, 2, CV_32F);
    result.depth_map.at<float>(0, 0) = 1.5f;
    result.depth_map.at<float>(1, 1) = 4.5f;

    StageDataPtr data = ToStageData(result);
    GRAPH_CHECK(data->shape() == Shape::kDenseMap);
    const DenseMapData* dense = dynamic_cast<const DenseMapData*>(data.get());
    GRAPH_CHECK(dense != NULL);
    GRAPH_CHECK(dense->values.rows == 2);
    GRAPH_CHECK(dense->values.cols == 2);
    GRAPH_CHECK(dense->values.at<float>(1, 1) == 4.5f);
}

void TestRestorationConversion() {
    RestorationResult result;
    result.restored_image = cv::Mat(2, 3, CV_8UC1);
    result.restored_image.at<uchar>(0, 2) = 42;

    StageDataPtr data = ToStageData(result);
    GRAPH_CHECK(data->shape() == Shape::kImage);
    const ImageData* image = dynamic_cast<const ImageData*>(data.get());
    GRAPH_CHECK(image != NULL);
    GRAPH_CHECK(image->image.rows == 2);
    GRAPH_CHECK(image->image.cols == 3);
    GRAPH_CHECK(image->image.at<uchar>(0, 2) == 42);
}

void TestClassificationConversion() {
    std::vector<ClassificationResult> results;
    ClassificationResult cls;
    cls.class_id = 9;
    cls.class_name = "dog";
    cls.confidence = 0.42f;
    results.push_back(cls);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kScores);
    const ScoresData* scores = dynamic_cast<const ScoresData*>(data.get());
    GRAPH_CHECK(scores != NULL);
    GRAPH_CHECK(scores->items.size() == 1);
    GRAPH_CHECK(scores->items[0].class_id == 9);
    GRAPH_CHECK(scores->items[0].class_name == "dog");
}

void TestDetection3DConversion() {
    std::vector<Detection3DResult> results;
    Detection3DResult det3d;
    det3d.class_id = 2;
    det3d.class_name = "car";
    det3d.confidence = 0.65f;
    det3d.x3d = 10.f; det3d.y3d = 20.f; det3d.z3d = 1.f;
    results.push_back(det3d);

    StageDataPtr data = ToStageData(results);
    GRAPH_CHECK(data->shape() == Shape::kBoxes3d);
    const Boxes3dData* boxes3d = dynamic_cast<const Boxes3dData*>(data.get());
    GRAPH_CHECK(boxes3d != NULL);
    GRAPH_CHECK(boxes3d->items.size() == 1);
    GRAPH_CHECK(boxes3d->items[0].class_name == "car");
    GRAPH_CHECK(boxes3d->items[0].x3d == 10.f);
}

void TestFakeRegistryServesScriptedResults() {
    FakeModelRegistry registry;

    ModelInfo detector;
    detector.model_name = "fake_detector";
    detector.task = "object_detection";
    detector.dxnn_file = "fake_detector_640x640.dxnn";
    detector.input_width = 640;
    detector.input_height = 640;
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.produces_landmarks = false;
    detector.ready = true;

    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem item;
    item.box = cv::Rect2f(10.f, 10.f, 50.f, 80.f);
    item.score = 0.9f;
    item.class_name = "person";
    boxes->items.push_back(item);

    registry.AddModel(detector, boxes);

    GRAPH_CHECK(registry.find("fake_detector") != NULL);
    GRAPH_CHECK(registry.find("nope") == NULL);
    GRAPH_CHECK(registry.list().size() == 1);

    StageParams params;
    std::unique_ptr<IStage> stage =
        registry.createStage("fake_detector", "/dev/null", params);
    GRAPH_CHECK(stage.get() != NULL);
    GRAPH_CHECK(stage->outputShape() == Shape::kBoxes);
    GRAPH_CHECK(stage->inputContract() == InputContract::kFullFrame);

    StageInput input;
    input.image = cv::Mat::zeros(480, 640, CV_8UC3);
    StageResult result = stage->run(input);
    const BoxesData* got = dynamic_cast<const BoxesData*>(result.data.get());
    GRAPH_CHECK(got != NULL);
    GRAPH_CHECK(got->items.size() == 1);
    GRAPH_CHECK(got->items[0].class_name == "person");
}

void TestFakeRegistryReportsNotReady() {
    FakeModelRegistry registry;
    ModelInfo pending;
    pending.model_name = "rtdetr_r50";
    pending.task = "object_detection";
    pending.output_shape = Shape::kBoxes;
    pending.input_contract = InputContract::kFullFrame;
    pending.ready = false;
    pending.not_ready_reason = "no postprocessor";
    registry.AddModel(pending, StageDataPtr());

    const ModelInfo* info = registry.find("rtdetr_r50");
    GRAPH_CHECK(info != NULL);
    GRAPH_CHECK(!info->ready);
    GRAPH_CHECK(info->not_ready_reason == "no postprocessor");
}

void TestFakeStageSubmitDeliversScriptedResultAndHonoursOrder() {
    FakeModelRegistry registry;

    ModelInfo detector;
    detector.model_name = "fake_detector_async";
    detector.task = "object_detection";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.ready = true;

    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem item;
    item.class_name = "cat";
    boxes->items.push_back(item);
    registry.AddModel(detector, boxes);

    StageParams params;
    std::unique_ptr<IStage> stage =
        registry.createStage("fake_detector_async", "/dev/null", params);
    FakeStage* fake = dynamic_cast<FakeStage*>(stage.get());
    GRAPH_CHECK(fake != NULL);

    std::vector<int> arrival_order;
    StageInput input_a;
    input_a.origin.roi_index = 0;
    StageInput input_b;
    input_b.origin.roi_index = 1;
    StageInput input_c;
    input_c.origin.roi_index = 2;

    stage->submit(input_a, [&arrival_order](const StageResult& r, const std::string& err) {
        GRAPH_CHECK(err.empty());
        arrival_order.push_back(r.origin.roi_index);
    });
    stage->submit(input_b, [&arrival_order](const StageResult& r, const std::string& err) {
        GRAPH_CHECK(err.empty());
        arrival_order.push_back(r.origin.roi_index);
    });
    stage->submit(input_c, [&arrival_order](const StageResult& r, const std::string& err) {
        GRAPH_CHECK(err.empty());
        arrival_order.push_back(r.origin.roi_index);
    });

    GRAPH_CHECK(fake->pending() == 3u);
    fake->SetDeliveryOrder(FakeStage::kReverse);
    fake->flush();
    GRAPH_CHECK(fake->pending() == 0u);
    GRAPH_CHECK(arrival_order.size() == 3u);
    GRAPH_CHECK(arrival_order[0] == 2);
    GRAPH_CHECK(arrival_order[1] == 1);
    GRAPH_CHECK(arrival_order[2] == 0);
}

void TestFakeStageForwardFlushOrder() {
    FakeModelRegistry registry;
    ModelInfo detector;
    detector.model_name = "fake_detector_fwd";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.ready = true;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    registry.AddModel(detector, boxes);

    StageParams params;
    std::unique_ptr<IStage> stage =
        registry.createStage("fake_detector_fwd", "/dev/null", params);
    FakeStage* fake = dynamic_cast<FakeStage*>(stage.get());

    std::vector<int> arrival_order;
    for (int i = 0; i < 3; ++i) {
        StageInput input;
        input.origin.roi_index = i;
        stage->submit(input, [&arrival_order](const StageResult& r, const std::string&) {
            arrival_order.push_back(r.origin.roi_index);
        });
    }
    // Default mode is forward; flush() must honour it.
    fake->flush();
    GRAPH_CHECK(arrival_order.size() == 3u);
    GRAPH_CHECK(arrival_order[0] == 0);
    GRAPH_CHECK(arrival_order[1] == 1);
    GRAPH_CHECK(arrival_order[2] == 2);
}

void TestFakeStageFailurePropagatesToCallback() {
    FakeModelRegistry registry;
    ModelInfo detector;
    detector.model_name = "fake_detector_fail";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.ready = true;
    registry.AddModel(detector, StageDataPtr());
    registry.SetFailure("fake_detector_fail", "device busy");

    StageParams params;
    std::unique_ptr<IStage> stage =
        registry.createStage("fake_detector_fail", "/dev/null", params);

    bool run_threw = false;
    try {
        StageInput input;
        stage->run(input);
    } catch (const std::runtime_error& e) {
        run_threw = true;
        GRAPH_CHECK(std::string(e.what()) == "device busy");
    }
    GRAPH_CHECK(run_threw);

    bool callback_saw_error = false;
    StageInput input;
    stage->submit(input, [&callback_saw_error](const StageResult& r, const std::string& err) {
        callback_saw_error = !err.empty();
        GRAPH_CHECK(r.data.get() == NULL);
    });
    stage->flush();
    GRAPH_CHECK(callback_saw_error);
}

void TestRegistryRejectsUnknownAndNotReadyStage() {
    FakeModelRegistry registry;
    StageParams params;

    bool threw_unknown = false;
    try {
        registry.createStage("nonexistent_model", "/dev/null", params);
    } catch (const std::runtime_error&) {
        threw_unknown = true;
    }
    GRAPH_CHECK(threw_unknown);

    ModelInfo pending;
    pending.model_name = "not_ready_model";
    pending.output_shape = Shape::kBoxes;
    pending.input_contract = InputContract::kFullFrame;
    pending.ready = false;
    pending.not_ready_reason = "no factory";
    registry.AddModel(pending, StageDataPtr());

    bool threw_not_ready = false;
    try {
        registry.createStage("not_ready_model", "/dev/null", params);
    } catch (const std::runtime_error&) {
        threw_not_ready = true;
    }
    GRAPH_CHECK(threw_not_ready);
}

namespace {

bool ThrowsWithCode(const std::string& json, GraphErrorCode expected,
                    std::string* message) {
    try {
        ParseGraphText(json, "test.json");
    } catch (const GraphError& error) {
        *message = error.what();
        return error.code() == expected;
    } catch (...) {
        *message = "non-GraphError exception";
        return false;
    }
    *message = "no exception";
    return false;
}

}  // namespace

void TestParseMinimalGraph() {
    const std::string json =
        "{\n"
        "  \"version\": 1,\n"
        "  \"name\": \"person-reid\",\n"
        "  \"nodes\": [\n"
        "    {\"id\": \"cam\",  \"type\": \"source\", \"uri\": \"a.mp4\"},\n"
        "    {\"id\": \"od\",   \"model\": \"yolov8n\",\n"
        "     \"params\": {\"score_threshold\": 0.35},\n"
        "     \"track\": {\"algo\": \"iou\", \"max_age\": 30}},\n"
        "    {\"id\": \"reid\", \"model\": \"casvit_t\"}\n"
        "  ],\n"
        "  \"edges\": [\n"
        "    {\"from\": \"cam\", \"to\": \"od\"},\n"
        "    {\"from\": \"od\",  \"to\": \"reid\",\n"
        "     \"roi\": {\"classes\": [\"person\"], \"pad\": 0.05, \"max\": 16}}\n"
        "  ]\n"
        "}";

    GraphSpec spec = ParseGraphText(json, "test.json");
    GRAPH_CHECK(spec.version == 1);
    GRAPH_CHECK(spec.name == "person-reid");
    GRAPH_CHECK(spec.nodes.size() == 3);
    GRAPH_CHECK(spec.edges.size() == 2);

    GRAPH_CHECK(spec.nodes[0].is_source);
    GRAPH_CHECK(spec.nodes[0].uri == "a.mp4");
    GRAPH_CHECK(!spec.nodes[1].is_source);
    GRAPH_CHECK(spec.nodes[1].model == "yolov8n");
    GRAPH_CHECK(spec.nodes[1].params.numeric["score_threshold"] == 0.35);
    GRAPH_CHECK(spec.nodes[1].track.present);
    GRAPH_CHECK(spec.nodes[1].track.algo == "iou");
    GRAPH_CHECK(spec.nodes[1].track.max_age == 30);
    GRAPH_CHECK(!spec.nodes[2].track.present);

    GRAPH_CHECK(!spec.edges[0].roi.present);
    GRAPH_CHECK(spec.edges[1].roi.present);
    GRAPH_CHECK(spec.edges[1].roi.classes.size() == 1);
    GRAPH_CHECK(spec.edges[1].roi.classes[0] == "person");
    GRAPH_CHECK(spec.edges[1].roi.pad == 0.05f);
    GRAPH_CHECK(spec.edges[1].roi.max == 16);
    GRAPH_CHECK(spec.edges[1].roi.align.empty());
}

void TestParseRejectsMalformedJson() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode("{ not json", GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("test.json") != std::string::npos);
}

// Review Focus #2 — well-formed JSON, wrong types.
void TestParseRejectsWrongTypes() {
    std::string message;

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":{},\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("nodes") != std::string::npos);
    GRAPH_CHECK(message.find("array") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("edges") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"a\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"a\",\"to\":\"a\",\"roi\":{\"pad\":\"0.05\"}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("pad") != std::string::npos);
    GRAPH_CHECK(message.find("number") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"model\":\"m\"}],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("id") != std::string::npos);
}

void TestParseRejectsUnsupportedVersion() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":2,\"nodes\":[],\"edges\":[]}",
        GraphErrorCode::kGraphVersion, &message));
    GRAPH_CHECK(message.find("supports: 1") != std::string::npos);
}

void TestParseRejectsReservedKeys() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"c\",\"model\":\"clip\","
        "\"prompt\":[\"person\"]}],\"edges\":[]}",
        GraphErrorCode::kGraphReserved, &message));
    GRAPH_CHECK(message.find("prompt") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"s\",\"type\":\"fuse\","
        "\"inputs\":[\"a\",\"b\"]}],\"edges\":[]}",
        GraphErrorCode::kGraphReserved, &message));
    GRAPH_CHECK(message.find("fuse") != std::string::npos);
}

// ---------------------------------------------------------------------
// SP1 Task 1: schema numbers, names and the tracker algo (U-09, U-10)
// ---------------------------------------------------------------------
namespace {

/// cam -> od (with "track": `track` unless empty) -> reid over an ROI edge
/// whose "roi" is `roi`. `top` is spliced in after "version":1, e.g.
/// "\"name\":5,".
std::string SchemaProbeJson(const std::string& top, const std::string& track,
                            const std::string& roi) {
    return "{\"version\":1," + top + "\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"" +
           (track.empty() ? std::string() : ",\"track\":" + track) + "},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"reid\",\"roi\":" + roi + "}]}";
}

/// The what() of the GRAPH_SCHEMA error ParseGraphText throws for `json`.
std::string SchemaError(const std::string& json) {
    std::string message;
    if (!ThrowsWithCode(json, GraphErrorCode::kGraphSchema, &message)) {
        return "not a GRAPH_SCHEMA error: " + message;
    }
    return message;
}

bool Parses(const std::string& json) {
    try {
        ParseGraphText(json, "test.json");
    } catch (const GraphError&) {
        return false;
    }
    return true;
}

}  // namespace

/// --output-layers names a frame's files <frame>_<node id>.png beside
/// <frame>_base.png (the clean frame) and <frame>_events.png (the event
/// panel), so a node may not be called "base" or "events".
void TestParseRejectsTheLayerFileNamesAsNodeIds() {
    const char* const reserved[] = {"base", "events"};
    for (const char* id : reserved) {
        std::string message;
        GRAPH_CHECK(ThrowsWithCode(
            std::string("{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\","
                        "\"uri\":\"a.jpg\"},{\"id\":\"") + id +
                "\",\"model\":\"m\"}],\"edges\":[{\"from\":\"cam\",\"to\":\"" + id + "\"}]}",
            GraphErrorCode::kGraphSchema, &message));
        GRAPH_CHECK(message.find(std::string("node id \"") + id + "\" is reserved") != std::string::npos);
        GRAPH_CHECK(message.find("node[1]") != std::string::npos);
        GRAPH_CHECK(message.find("-> ") != std::string::npos);
    }
    // only the exact names: other ids that start with them are fine
    GRAPH_CHECK(Parses("{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\","
                       "\"uri\":\"a.jpg\"},{\"id\":\"base_od\",\"model\":\"m\"}],"
                       "\"edges\":[{\"from\":\"cam\",\"to\":\"base_od\"}]}"));
}

namespace {

const std::string kRoiWhere =
    "ERROR [GRAPH_SCHEMA] test.json edge[1] \"od\"->\"reid\": ";
const std::string kTrackWhere = "ERROR [GRAPH_SCHEMA] test.json node[1]: ";
const std::string kMaxHow =
    "\n  -> \"max\" keeps at most this many boxes; leave it out for no limit";
const std::string kPadHow =
    "\n  -> \"pad\" grows each box by this fraction of its size on every side "
    "(0.05 = 5 %)";
const std::string kScoreHow =
    "\n  -> \"min_score\" is a confidence as the producer reports it "
    "(0.35, not 35)";
const std::string kIouHow =
    "\n  -> \"iou\" is the overlap a box needs to continue a track "
    "(0.3 is typical)";
const std::string kAgeHow =
    "\n  -> \"max_age\" is how many frames an unmatched track survives";

void CheckSchemaMessage(const std::string& json, const std::string& expected) {
    const std::string got = SchemaError(json);
    GRAPH_CHECK(got == expected);
    if (got != expected) {
        std::printf("      got:      %s\n      expected: %s\n", got.c_str(),
                    expected.c_str());
    }
}

}  // namespace

void TestSchemaRejectsAFractionalVersion() {
    CheckSchemaMessage("{\"version\":1.5,\"nodes\":[],\"edges\":[]}",
                       "ERROR [GRAPH_SCHEMA] test.json: \"version\" must be a "
                       "whole number, got 1.5\n  -> add \"version\": 1");
    GraphSpec spec =
        ParseGraphText("{\"version\":1.0,\"nodes\":[],\"edges\":[]}", "test.json");
    GRAPH_CHECK(spec.version == 1);
    std::string message;
    GRAPH_CHECK(ThrowsWithCode("{\"version\":2,\"nodes\":[],\"edges\":[]}",
                               GraphErrorCode::kGraphVersion, &message));
    GRAPH_CHECK(message.find("graph version 2 is not supported") != std::string::npos);
}

void TestSchemaRejectsANonStringName() {
    CheckSchemaMessage(SchemaProbeJson("\"name\":5,", "", "{}"),
                       "ERROR [GRAPH_SCHEMA] test.json: \"name\" must be a "
                       "string, got 5\n  -> \"name\" is free text: put it in quotes");
    GRAPH_CHECK(ParseGraphText(SchemaProbeJson("\"name\":\"n\",", "", "{}"),
                               "test.json").name == "n");
}

void TestSchemaChecksEachRoiNumber() {
    struct Case {
        const char* roi;
        std::string message;
    };
    const Case bad[] = {
        {"{\"max\":0}", "\"max\" must be a whole number of at least 1, got 0" + kMaxHow},
        {"{\"max\":-1}", "\"max\" must be a whole number of at least 1, got -1" + kMaxHow},
        {"{\"max\":2.5}", "\"max\" must be a whole number of at least 1, got 2.5" + kMaxHow},
        {"{\"max\":3000000000}",
         "\"max\" must be at most 2147483647, got 3000000000" + kMaxHow},
        {"{\"pad\":-0.1}", "\"pad\" must be between 0 and 1, got -0.1" + kPadHow},
        {"{\"pad\":5}", "\"pad\" must be between 0 and 1, got 5" + kPadHow},
        {"{\"min_score\":35}", "\"min_score\" must be between 0 and 1, got 35" + kScoreHow},
        {"{\"min_score\":-0.5}",
         "\"min_score\" must be between 0 and 1, got -0.5" + kScoreHow},
        {"{\"min_area\":-5}",
         "\"min_area\" must be at least 0, got -5\n  -> \"min_area\" is an area in pixels"},
        {"{\"min_area\":1e39}",
         "\"min_area\" must be at most 3.4028234663852886e+38, got 1e+39\n"
         "  -> \"min_area\" is an area in pixels"},
    };
    for (std::size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        CheckSchemaMessage(SchemaProbeJson("", "", bad[i].roi), kRoiWhere + bad[i].message);
    }
    const char* good[] = {"{\"max\":1}", "{\"max\":16.0}", "{\"pad\":0}", "{\"pad\":1}",
                          "{\"min_score\":0}", "{\"min_score\":1}", "{\"min_area\":0}"};
    for (std::size_t i = 0; i < sizeof(good) / sizeof(good[0]); ++i) {
        GRAPH_CHECK(Parses(SchemaProbeJson("", "", good[i])));
    }
    GRAPH_CHECK(ParseGraphText(SchemaProbeJson("", "", "{\"max\":16.0}"), "test.json")
                    .edges[1].roi.max == 16);
}

void TestSchemaReportsEveryRoiViolationAtOnce() {
    CheckSchemaMessage(
        SchemaProbeJson("", "", "{\"max\":2.5,\"pad\":-0.1,\"min_score\":35}"),
        kRoiWhere +
            "3 invalid values: \"min_score\" must be between 0 and 1, got 35; "
            "\"pad\" must be between 0 and 1, got -0.1; "
            "\"max\" must be a whole number of at least 1, got 2.5\n"
            "  -> \"min_score\" is a confidence as the producer reports it "
            "(0.35, not 35); \"pad\" grows each box by this fraction of its "
            "size on every side (0.05 = 5 %); \"max\" keeps at most this many "
            "boxes; leave it out for no limit");
}

void TestSchemaChecksTrackFields() {
    CheckSchemaMessage(SchemaProbeJson("", "{\"algo\":\"sort\"}", "{}"),
                       kTrackWhere + "unknown tracker \"sort\"\n"
                                     "  -> accepted \"algo\" values: iou");
    CheckSchemaMessage(SchemaProbeJson("", "{\"iou\":0}", "{}"),
                       kTrackWhere + "\"iou\" must be greater than 0 and at most 1, got 0" +
                           kIouHow);
    CheckSchemaMessage(SchemaProbeJson("", "{\"iou\":1.5}", "{}"),
                       kTrackWhere + "\"iou\" must be greater than 0 and at most 1, got 1.5" +
                           kIouHow);
    CheckSchemaMessage(SchemaProbeJson("", "{\"max_age\":-1}", "{}"),
                       kTrackWhere + "\"max_age\" must be a whole number of at least 0, got -1" +
                           kAgeHow);
    CheckSchemaMessage(SchemaProbeJson("", "{\"max_age\":1.5}", "{}"),
                       kTrackWhere + "\"max_age\" must be a whole number of at least 0, got 1.5" +
                           kAgeHow);
    CheckSchemaMessage(
        SchemaProbeJson("", "{\"algo\":\"sort\",\"iou\":0,\"max_age\":1.5}", "{}"),
        kTrackWhere +
            "3 invalid values: unknown tracker \"sort\"; \"iou\" must be greater "
            "than 0 and at most 1, got 0; \"max_age\" must be a whole number of "
            "at least 0, got 1.5\n  -> accepted \"algo\" values: iou; \"iou\" is "
            "the overlap a box needs to continue a track (0.3 is typical); "
            "\"max_age\" is how many frames an unmatched track survives");
    // Spec R1: 0 is "drop on the first miss", used on purpose by the parity
    // stress tests; iou 1 is the strictest legal match.
    const GraphSpec spec = ParseGraphText(
        SchemaProbeJson("", "{\"algo\":\"iou\",\"iou\":1,\"max_age\":0}", "{}"), "test.json");
    GRAPH_CHECK(spec.nodes[1].track.max_age == 0);
    GRAPH_CHECK(spec.nodes[1].track.iou == 1.f);
}

/// Passes before AND after this task: today's single-error texts must not move.
void TestSchemaKeepsTodaysSingleTypeErrors() {
    CheckSchemaMessage(SchemaProbeJson("", "", "{\"pad\":\"0.05\"}"),
                       kRoiWhere + "\"pad\" must be a number");
    CheckSchemaMessage(SchemaProbeJson("", "{\"algo\":5}", "{}"),
                       kTrackWhere + "\"algo\" must be a string");
    CheckSchemaMessage(SchemaProbeJson("", "", "{\"align\":\"face6\"}"),
                       kRoiWhere + "unknown align mode \"face6\"\n"
                                   "  -> the only supported value is \"face5\"");
    CheckSchemaMessage(SchemaProbeJson("", "", "{\"classes\":\"person\"}"),
                       kRoiWhere + "\"classes\" must be an array");
}

namespace {

FakeModelRegistry BuildValidationRegistry() {
    FakeModelRegistry registry;

    ModelInfo detector;
    detector.model_name = "yolov8n";
    detector.task = "object_detection";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.ready = true;
    registry.AddModel(detector, StageDataPtr(new BoxesData(Shape::kBoxes)));

    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;
    registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));

    ModelInfo reid;
    reid.model_name = "casvit_t";
    reid.task = "reid";
    reid.output_shape = Shape::kVector;
    reid.input_contract = InputContract::kRoi;
    reid.ready = true;
    registry.AddModel(reid, StageDataPtr(new VectorData()));

    ModelInfo face;
    face.model_name = "scrfd10g";
    face.task = "face_detection";
    face.output_shape = Shape::kBoxes;
    face.input_contract = InputContract::kFullFrame;
    face.produces_landmarks = true;
    face.ready = true;
    registry.AddModel(face, StageDataPtr(new BoxesData(Shape::kBoxes)));

    ModelInfo pending;
    pending.model_name = "rtdetr_r50";
    pending.task = "object_detection";
    pending.output_shape = Shape::kBoxes;
    pending.input_contract = InputContract::kFullFrame;
    pending.ready = false;
    pending.not_ready_reason = "no postprocessor";
    registry.AddModel(pending, StageDataPtr());

    return registry;
}

bool ValidationThrows(const std::string& json, GraphErrorCode expected,
                      std::string* message) {
    FakeModelRegistry registry = BuildValidationRegistry();
    try {
        GraphSpec spec = ParseGraphText(json, "test.json");
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        *message = error.what();
        return error.code() == expected;
    } catch (const std::exception& error) {
        *message = std::string("non-GraphError: ") + error.what();
        return false;
    }
    *message = "no exception";
    return false;
}

std::string ValidCascade() {
    return "{\"version\":1,\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":["
           "{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

}  // namespace

void TestValidateAcceptsCascade() {
    FakeModelRegistry registry = BuildValidationRegistry();
    GraphSpec spec = ParseGraphText(ValidCascade(), "test.json");
    bool threw = false;
    try {
        ValidateGraph(spec, registry);
    } catch (const std::exception&) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
}

void TestValidateRejectsUnknownModel() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov99\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kModelUnknown, &message));
    GRAPH_CHECK(message.find("yolov99") != std::string::npos);
    GRAPH_CHECK(message.find("model_registry.json") != std::string::npos);
}

void TestValidateRejectsNotReadyModel() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"det\",\"model\":\"rtdetr_r50\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"det\"}]}",
        GraphErrorCode::kModelNotReady, &message));
    GRAPH_CHECK(message.find("no postprocessor") != std::string::npos);
    GRAPH_CHECK(message.find("graph_models.md") != std::string::npos);
}

void TestValidateRejectsRoiFromNonProducer() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"seg\",\"to\":\"reid\",\"roi\":{}}]}",
        GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("labelmap") != std::string::npos);
    GRAPH_CHECK(message.find("not boxes") != std::string::npos);
}

void TestValidateRejectsRoiIntoFullFrameConsumer() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"seg\",\"roi\":{}}]}",
        GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("full frame") != std::string::npos);
}

void TestValidateRejectsFrameIntoRoiOnlyConsumer() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"reid\"}]}",
        GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("requires a cropped region") != std::string::npos);
}

void TestValidateRejectsAlignWithoutLandmarks() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"align\":\"face5\"}}]}",
        GraphErrorCode::kGraphAlign, &message));
    GRAPH_CHECK(message.find("yolov8n") != std::string::npos);
    GRAPH_CHECK(message.find("landmark") != std::string::npos);
}

void TestValidateAcceptsAlignFromFaceDetector() {
    FakeModelRegistry registry = BuildValidationRegistry();
    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"fd\",\"model\":\"scrfd10g\"},"
        "{\"id\":\"emb\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"fd\"},"
        "{\"from\":\"fd\",\"to\":\"emb\",\"roi\":{\"align\":\"face5\"}}]}",
        "test.json");
    bool threw = false;
    try {
        ValidateGraph(spec, registry);
    } catch (const std::exception&) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
}

void TestValidateRejectsCycle() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"a\",\"model\":\"yolov8n\"},"
        "{\"id\":\"b\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"a\"},"
        "{\"from\":\"a\",\"to\":\"b\",\"roi\":{}},"
        "{\"from\":\"b\",\"to\":\"a\",\"roi\":{}}]}",
        GraphErrorCode::kGraphCycle, &message));
    GRAPH_CHECK(message.find("cycle") != std::string::npos);
}

void TestValidateRejectsOrphanNode() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"lonely\",\"model\":\"bisenetv2\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphOrphan, &message));
    GRAPH_CHECK(message.find("lonely") != std::string::npos);
}

void TestValidateRejectsUnknownEndpoint() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"ghost\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("ghost") != std::string::npos);
}

void TestValidateRejectsSourceWithIncomingEdge() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"cam2\"},"
        "{\"from\":\"cam2\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("cam2") != std::string::npos);
}

// Review Focus #5 — a duplicate edge would double-count the async completion
// counter and the frame would never finalize.
void TestValidateRejectsDuplicateEdge() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{}},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("duplicate edge") != std::string::npos);
    GRAPH_CHECK(message.find("od") != std::string::npos);
    GRAPH_CHECK(message.find("reid") != std::string::npos);
}

void TestValidateRejectsNoSource() {
    std::string message;
    GRAPH_CHECK(ValidationThrows(
        "{\"version\":1,\"nodes\":[{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[]}",
        GraphErrorCode::kGraphOrphan, &message));
    GRAPH_CHECK(message.find("no source node") != std::string::npos);
}

void TestValidateRejectsBoxes3d() {
    FakeModelRegistry registry = BuildValidationRegistry();
    ModelInfo lidar;
    lidar.model_name = "sfa3d_608x608";
    lidar.task = "3d_object_detection";
    lidar.output_shape = Shape::kBoxes3d;
    lidar.input_contract = InputContract::kFullFrame;
    lidar.ready = true;
    registry.AddModel(lidar, StageDataPtr(new Boxes3dData()));

    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"l\",\"model\":\"sfa3d_608x608\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"l\"}]}",
        "test.json");

    bool correct = false;
    std::string message;
    try {
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        correct = error.code() == GraphErrorCode::kGraphEdge;
        message = error.what();
    }
    GRAPH_CHECK(correct);
    GRAPH_CHECK(message.find("LiDAR") != std::string::npos);
}

namespace {

BoxesData MakeBoxes(Shape shape) { return BoxesData(shape); }

BoxItem MakeItem(float x, float y, float w, float h, float score,
                 const std::string& name) {
    BoxItem item;
    item.box = cv::Rect2f(x, y, w, h);
    item.score = score;
    item.class_name = name;
    return item;
}

}  // namespace

void TestRouteFiltersByClassAndScore() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(10, 10, 40, 60, 0.90f, "person"));
    boxes.items.push_back(MakeItem(80, 10, 40, 60, 0.85f, "car"));
    boxes.items.push_back(MakeItem(10, 90, 40, 60, 0.20f, "person"));

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;
    spec.classes.push_back("person");
    spec.min_score = 0.5f;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].ref.from_roi);
    GRAPH_CHECK(crops[0].ref.parent_node == "od");
    GRAPH_CHECK(crops[0].ref.parent_index == 0);
    GRAPH_CHECK(crops[0].ref.roi_index == 0);
    GRAPH_CHECK(crops[0].image.cols == 40);
    GRAPH_CHECK(crops[0].image.rows == 60);
    GRAPH_CHECK(stats.filtered == 2);
}

void TestRouteAppliesMaxCutByScore() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(0, 0, 20, 20, 0.50f, "person"));
    boxes.items.push_back(MakeItem(30, 0, 20, 20, 0.95f, "person"));
    boxes.items.push_back(MakeItem(60, 0, 20, 20, 0.70f, "person"));

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;
    spec.max = 2;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 2);
    if (crops.size() != 2) return;
    // Highest score first, and roi_index follows emission order.
    GRAPH_CHECK(crops[0].ref.parent_index == 1);
    GRAPH_CHECK(crops[1].ref.parent_index == 2);
    GRAPH_CHECK(crops[0].ref.roi_index == 0);
    GRAPH_CHECK(crops[1].ref.roi_index == 1);
}

void TestRoutePadExpandsAndClips() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(0, 0, 100, 100, 0.9f, "person"));

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.10f;  // 10 px each side, clipped at the top-left corner

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].ref.src_box.x == 0.f);
    GRAPH_CHECK(crops[0].ref.src_box.y == 0.f);
    GRAPH_CHECK(crops[0].ref.src_box.width == 110.f);
    GRAPH_CHECK(crops[0].ref.src_box.height == 110.f);
}

// Review Focus #4 — a box entirely outside the frame must be dropped and
// counted, never handed to cv::resize.
void TestRouteDropsFullyOutOfBoundsBox() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(700, 500, 40, 40, 0.9f, "person"));  // outside
    boxes.items.push_back(MakeItem(-60, -60, 40, 40, 0.9f, "person"));  // outside
    boxes.items.push_back(MakeItem(10, 10, 40, 40, 0.9f, "person"));    // inside

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    GRAPH_CHECK(stats.clipped_away == 2);
    for (std::size_t i = 0; i < crops.size(); ++i) {
        GRAPH_CHECK(!crops[i].image.empty());
    }
}

void TestRouteDropsBoxBelowMinArea() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(10, 10, 4, 4, 0.9f, "person"));
    boxes.items.push_back(MakeItem(50, 50, 40, 40, 0.9f, "person"));

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;
    spec.min_area = 100.f;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    GRAPH_CHECK(stats.filtered == 1);
}

void TestRouteEmptyProducerYieldsNoCrops() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.empty());
    GRAPH_CHECK(stats.filtered == 0);
    GRAPH_CHECK(stats.clipped_away == 0);
}

// Fix round 3: ref.inv_align alone now maps crop-local to source (RouteRois
// folds the crop's own translation into it at construction time), so a
// hand-built RoiRef for a plain crop must set inv_align to that translation
// itself — leaving it at the default identity while setting src_box, as
// this test did before round 3, now describes an inconsistent RoiRef no
// real RouteRois output would ever produce.
void TestRestoreBoxMapsRoiLocalToSource() {
    RoiRef ref;
    ref.from_roi = true;
    ref.src_box = cv::Rect2f(100.f, 50.f, 80.f, 160.f);
    ref.inv_align = cv::Matx23f(1.f, 0.f, 100.f, 0.f, 1.f, 50.f);

    const cv::Rect2f restored = RestoreBox(cv::Rect2f(10.f, 20.f, 30.f, 40.f), ref);
    GRAPH_CHECK(restored.x == 110.f);
    GRAPH_CHECK(restored.y == 70.f);
    GRAPH_CHECK(restored.width == 30.f);
    GRAPH_CHECK(restored.height == 40.f);
}

void TestRestoreBoxIsIdentityForFullFrame() {
    RoiRef ref;  // from_roi == false
    const cv::Rect2f restored = RestoreBox(cv::Rect2f(5.f, 6.f, 7.f, 8.f), ref);
    GRAPH_CHECK(restored.x == 5.f);
    GRAPH_CHECK(restored.y == 6.f);
}

void TestTrackIdsAreStableAcrossFrames() {
    IouTracker tracker;

    BoxesData frame1 = MakeBoxes(Shape::kBoxes);
    frame1.items.push_back(MakeItem(10, 10, 40, 60, 0.9f, "person"));
    frame1.items.push_back(MakeItem(200, 10, 40, 60, 0.9f, "person"));
    ApplyTrackIds(&frame1, &tracker);
    GRAPH_CHECK(frame1.items[0].track_id >= 0);
    GRAPH_CHECK(frame1.items[1].track_id >= 0);
    GRAPH_CHECK(frame1.items[0].track_id != frame1.items[1].track_id);

    const int first_id = frame1.items[0].track_id;

    BoxesData frame2 = MakeBoxes(Shape::kBoxes);
    frame2.items.push_back(MakeItem(12, 11, 40, 60, 0.9f, "person"));  // moved
    frame2.items.push_back(MakeItem(202, 11, 40, 60, 0.9f, "person"));
    ApplyTrackIds(&frame2, &tracker);
    GRAPH_CHECK(frame2.items[0].track_id == first_id);
}

void TestRouteCarriesTrackIdIntoRoiRef() {
    IouTracker tracker;
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(10, 10, 40, 60, 0.9f, "person"));
    ApplyTrackIds(&boxes, &tracker);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].ref.track_id == boxes.items[0].track_id);
    GRAPH_CHECK(crops[0].ref.track_id >= 0);
}

// Fix round 1, finding #1 (medium): Review Focus #4's guarantee — a ROI
// that clips to nothing is dropped and counted, never handed to a Mat
// operation that throws on an empty region — was only crash-tested for the
// plain axis-aligned crop path. The OBB un-rotate branch has its own
// (shared, post-refactor) empty-window guard that had never been exercised.
void TestUnrotateCropDropsFullyOutOfBoundsObb() {
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    BoxItem obb = MakeItem(700, 500, 40, 40, 0.9f, "obj");  // outside
    obb.angle = 0.3f;  // radians; non-zero so RouteRois takes the OBB branch
    boxes.items.push_back(obb);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.empty());
    GRAPH_CHECK(stats.clipped_away == 1);
}

// Fix round 1, finding #1 (medium): face5's own fallback (BuildFace5Transform
// returning false for <5 landmarks) had never been exercised either. The
// documented behaviour is "fall back to the plain crop rather than dropping
// a detection the user asked to process" — assert that fallback fires, and
// (fix round 3) that inv_align is the crop's plain translation, matching
// every other plain-crop path: "no warp ran" means no rotation/scale
// component, NOT identity inv_align — src_box's translation still has to be
// in there, or RestoreBox/RestorePointWarped would return this crop's
// points unmoved instead of shifted back to the source frame.
void TestFace5FallsBackToPlainCropWithTooFewLandmarks() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    BoxItem face = MakeItem(50, 50, 80, 100, 0.9f, "face");
    face.landmarks.push_back(Keypoint(60.f, 70.f));
    face.landmarks.push_back(Keypoint(100.f, 70.f));  // only 2 of 5 landmarks
    boxes.items.push_back(face);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;
    spec.align = "face5";

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].image.cols == 80);
    GRAPH_CHECK(crops[0].image.rows == 100);
    GRAPH_CHECK(crops[0].ref.inv_align(0, 0) == 1.f);
    GRAPH_CHECK(crops[0].ref.inv_align(0, 1) == 0.f);
    GRAPH_CHECK(crops[0].ref.inv_align(0, 2) == 50.f);  // src_box.x, not 0
    GRAPH_CHECK(crops[0].ref.inv_align(1, 0) == 0.f);
    GRAPH_CHECK(crops[0].ref.inv_align(1, 1) == 1.f);
    GRAPH_CHECK(crops[0].ref.inv_align(1, 2) == 50.f);  // src_box.y, not 0
}

// Positive OBB case, requested alongside the OOB test: without this, a
// regression that silently skipped un-rotation (e.g. the shape/angle guard
// on the OBB branch always failing) would still pass every other test —
// the plain axis-aligned `else` branch would quietly produce a non-empty
// crop too, but with an identity inv_align instead of a real rotation.
void TestUnrotateCropProducesNonEmptyCropForInBoundsObb() {
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    BoxItem obb = MakeItem(250, 150, 100, 60, 0.9f, "obj");  // fully inside
    obb.angle = 0.4f;  // radians (~22.9 degrees)
    boxes.items.push_back(obb);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(!crops[0].image.empty());
    // Identity would be (1,0,0, 0,1,0); a real rotation moves the
    // off-diagonal terms well away from 0 (sin(0.4) ~= 0.39).
    GRAPH_CHECK(std::fabs(crops[0].ref.inv_align(0, 1)) > 0.01f);
    GRAPH_CHECK(std::fabs(crops[0].ref.inv_align(1, 0)) > 0.01f);
}

// Fix round 2 pins the invariant the round-1 dedup fix actually protects:
// crop.image's pixel size and ref.src_box's recorded size must describe the
// same window, because RestoreBox/RestorePoint and (later) Task 12's
// inv_align composition both trust src_box without re-deriving it from the
// pixels. A partially-clipped OBB straddling a frame edge with fractional
// coordinates is exactly the case where an independent recompute of the
// extraction window (clip-in-float-then-truncate vs truncate-then-intersect)
// disagrees by a pixel from ClipBoxToFrame's own result — not a crash, a
// silent mismatch. Pin the property (crop size == src_box size), not the
// specific truncated pixel count, so a future, still-correct change to how
// truncation works does not spuriously break this test.
void TestUnrotateCropWindowMatchesSrcBoxForPartiallyClippedObb() {
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    // Straddles the right frame edge (640 wide) with fractional coordinates:
    // x=634.7, width=8.5 -> box spans [634.7, 643.2), 3.2px past the edge.
    BoxItem obb = MakeItem(634.7f, 200.f, 8.5f, 20.f, 0.9f, "obj");
    obb.angle = 0.2f;  // radians; non-zero so RouteRois takes the OBB branch
    boxes.items.push_back(obb);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    std::vector<RoiCrop> crops =
        RouteRois(boxes, source, spec, "od", NULL, &stats);

    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].image.cols == static_cast<int>(crops[0].ref.src_box.width));
    GRAPH_CHECK(crops[0].image.rows == static_cast<int>(crops[0].ref.src_box.height));
}

// =============================================================================
// Task 9 — StageGraph and the synchronous executor.
//
// Build() is called with require_artifacts=false in these tests: the fakes
// point at "/models", a directory that holds no real .dxnn stub files, and
// these tests are about graph shape / execution semantics, not the artifact
// check. TestStageGraphBuildThrowsModelMissingWhenArtifactAbsent below is
// the one test that turns require_artifacts back on, to pin the kModelMissing
// producer.
// =============================================================================

namespace {

/// cam -> od -(roi)-> reid, with the detector scripted to emit two boxes.
FakeModelRegistry BuildCascadeRegistry() {
    FakeModelRegistry registry;

    ModelInfo detector;
    detector.model_name = "yolov8n";
    detector.task = "object_detection";
    detector.dxnn_file = "yolov8n.dxnn";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.input_width = 640;
    detector.input_height = 640;
    detector.ready = true;
    detector.published = true;  // the MODEL_MISSING tests expect a download line

    // Scores deliberately reversed relative to parent_index (item 0 scores
    // lower than item 1): RouteRois's max-cut step sorts by descending
    // score, so if TestSyncExecutorRunsCascade's parent_index assertions
    // passed only because score order happened to equal index order, this
    // fixture makes them fail instead of passing by coincidence.
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10, 10, 40, 60, 0.8f, "person"));
    boxes->items.push_back(MakeItem(200, 20, 40, 60, 0.9f, "person"));
    registry.AddModel(detector, boxes);

    ModelInfo reid;
    reid.model_name = "casvit_t";
    reid.task = "reid";
    reid.dxnn_file = "casvit_t.dxnn";
    reid.output_shape = Shape::kVector;
    reid.input_contract = InputContract::kRoi;
    reid.input_width = 224;
    reid.input_height = 224;
    reid.ready = true;

    std::shared_ptr<VectorData> vector(new VectorData());
    vector->values.push_back(0.5f);
    registry.AddModel(reid, vector);

    return registry;
}

std::string CascadeJson() {
    return "{\"version\":1,\"name\":\"t\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"track\":{\"algo\":\"iou\"}},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

}  // namespace

void TestStageGraphBuildsTopologicalOrder() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    GRAPH_CHECK(graph.nodes().size() == 3);
    GRAPH_CHECK(graph.edges().size() == 2);
    GRAPH_CHECK(graph.source_indices().size() == 1);

    const std::vector<std::size_t>& order = graph.topological_order();
    GRAPH_CHECK(order.size() == 3);
    if (order.size() != 3) return;
    GRAPH_CHECK(graph.nodes()[order[0]].id == "cam");
    GRAPH_CHECK(graph.nodes()[order[1]].id == "od");
    GRAPH_CHECK(graph.nodes()[order[2]].id == "reid");

    GRAPH_CHECK(graph.nodes()[order[1]].tracked);
    GRAPH_CHECK(!graph.nodes()[order[2]].tracked);
}

void TestSyncExecutorRunsCascade() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport report = executor.RunFrame(graph, frame, 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(report.roi_results.count("reid") == 1);
    GRAPH_CHECK(report.roi_results["reid"].size() == 2);
    if (report.roi_results["reid"].size() != 2) return;

    // ROI results arrive sorted by (parent index, roi index).
    GRAPH_CHECK(report.roi_results["reid"][0].origin.parent_index == 0);
    GRAPH_CHECK(report.roi_results["reid"][1].origin.parent_index == 1);

    // Tracking was declared on "od", so ids reached the crops.
    GRAPH_CHECK(report.roi_results["reid"][0].origin.track_id >= 0);
}

void TestSyncExecutorHandlesEmptyDetection() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    // Rescript the detector to emit nothing.
    ModelInfo detector = *registry.find("yolov8n");
    registry.AddModel(detector, StageDataPtr(new BoxesData(Shape::kBoxes)));

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(report.roi_results["reid"].empty());
}

// Review Focus #3 — an empty frame must be refused with a message, not
// silently produce a partial report.
void TestSyncExecutorRejectsEmptyFrame() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report = executor.RunFrame(graph, cv::Mat(), 0);

    GRAPH_CHECK(!report.error.empty());
    GRAPH_CHECK(report.error.find("empty") != std::string::npos);
    GRAPH_CHECK(report.node_results.empty());
}

void TestSyncExecutorIsolatesStageFailure() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFailure("casvit_t", "simulated inference failure");

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(!report.error.empty());
    GRAPH_CHECK(report.error.find("simulated") != std::string::npos);
    // The detector result is still present: failure is per frame, not fatal.
    GRAPH_CHECK(report.node_results.count("od") == 1);
}

// StageGraph::Build is the sole producer of kModelMissing in the tree (Task 7
// review finding): when require_artifacts is true (the production default)
// and a node's resolved .dxnn file is not on disk, Build must raise it.
void TestStageGraphBuildThrowsModelMissingWhenArtifactAbsent() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    bool threw = false;
    try {
        graph.Build(spec, registry, "/definitely/not/a/real/model/dir");
    } catch (const GraphError& error) {
        threw = true;
        GRAPH_CHECK(error.code() == GraphErrorCode::kModelMissing);
        GRAPH_CHECK(std::string(error.what()).find("MODEL_MISSING") !=
                    std::string::npos);
    }
    GRAPH_CHECK(threw);
}

// SP1 U-23: Build's own MODEL_MISSING (direct Build() callers) names the
// model as the model zoo spells it, the translation PrepareGraph prints.
void TestBuildFallbackPrintsTheModelZooName() {
    FakeModelRegistry registry;
    ModelInfo obb;
    obb.model_name = "yolo26l_obb";
    obb.task = "obb_detection";
    obb.dxnn_file = "yolo26-l-obb_1024x1024.dxnn";
    obb.download_name = "yolo26l-obb";
    obb.published = true;  // an unpublished model gets no download line (M1)
    obb.output_shape = Shape::kObBoxes;
    obb.input_contract = InputContract::kFullFrame;
    obb.ready = true;
    registry.AddModel(obb, StageDataPtr(new BoxesData(Shape::kObBoxes)));
    const GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"name\":\"z\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"obb\",\"model\":\"yolo26l_obb\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"obb\"}]}", "z.json");
    ValidateGraph(spec, registry);
    std::string message;
    try {
        StageGraph graph;
        graph.Build(spec, registry, "/definitely/not/a/real/model/dir");
    } catch (const GraphError& error) {
        message = error.what();
    }
    // download_models.py into the model dir, never setup.sh: setup.sh also
    // re-fetches the sample videos and can rm -rf an assets/videos symlink.
    GRAPH_CHECK(message.find("-> python3 scripts/download_models.py --models yolo26l-obb "
                             "--output /definitely/not/a/real/model/dir") != std::string::npos);
    GRAPH_CHECK(message.find("setup.sh") == std::string::npos);
    GRAPH_CHECK(message.find("--models yolo26l_obb") == std::string::npos);

    // No download name: the registry name, as before.
    FakeModelRegistry cascade = BuildCascadeRegistry();
    const GraphSpec plain = ParseGraphText(CascadeJson(), "t.json");
    message.clear();
    try {
        StageGraph graph;
        graph.Build(plain, cascade, "/definitely/not/a/real/model/dir");
    } catch (const GraphError& error) {
        message = error.what();
    }
    GRAPH_CHECK(message.find("-> python3 scripts/download_models.py --models yolov8n --output "
                             "/definitely/not/a/real/model/dir") != std::string::npos);
}

namespace {

std::string BuildError(const std::string& json, const FakeModelRegistry& registry,
                       GraphErrorCode* code) {
    const GraphSpec spec = ParseGraphText(json, "t.json");
    ValidateGraph(spec, registry);
    try {
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
    } catch (const GraphError& error) {
        *code = error.code();
        return error.what();
    } catch (const std::exception& error) {
        return std::string("untyped: ") + error.what();
    }
    return "no exception";
}

std::string TwoDetectorsJson() {
    return "{\"version\":1,\"name\":\"two\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},"
           "{\"id\":\"od2\",\"model\":\"yolov8n\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},{\"from\":\"cam\",\"to\":\"od2\"}]}";
}

}  // namespace

// SP1 U-18: an engine the runtime cannot load is MODEL_LOAD, naming the node,
// with a hint that depends on whether engines were loaded before it.
void TestBuildReportsAnEngineThatWillNotLoadAsModelLoad() {
    GRAPH_CHECK(std::string(ToString(GraphErrorCode::kModelLoad)) == "MODEL_LOAD");

    // A later engine fails: the memory hint, counting what was loaded first.
    FakeModelRegistry later = BuildCascadeRegistry();
    later.SetCreateFailure("casvit_t", "Failed to register memory cache");
    GraphErrorCode code = GraphErrorCode::kGraphSchema;
    std::string message = BuildError(CascadeJson(), later, &code);
    GRAPH_CHECK(code == GraphErrorCode::kModelLoad);
    GRAPH_CHECK(message.find("ERROR [MODEL_LOAD] node \"reid\": model \"casvit_t\" "
                             "(casvit_t.dxnn) could not be loaded: Failed to register "
                             "memory cache") == 0);
    GRAPH_CHECK(message.find("device memory may be full") != std::string::npos);
    GRAPH_CHECK(message.find("loaded 1 model(s) before this one") != std::string::npos);
    // M2: the example names the models as variants, as the READMEs do.
    GRAPH_CHECK(message.find("DX-M1 cannot hold a second realesrgan-x2_192x192 next to "
                             "yolov8-n_640x640 and resnet50_224x224") != std::string::npos);
    if (code != GraphErrorCode::kModelLoad) std::printf("      %s\n", message.c_str());

    // The second copy of one model: DX-M1's realesrgan_x2 case, in miniature.
    FakeModelRegistry twice = BuildCascadeRegistry();
    twice.SetCreateFailure("yolov8n", "Failed to register memory cache", 1);
    code = GraphErrorCode::kGraphSchema;
    message = BuildError(TwoDetectorsJson(), twice, &code);
    GRAPH_CHECK(code == GraphErrorCode::kModelLoad);
    GRAPH_CHECK(message.find("node \"od2\"") != std::string::npos);
    GRAPH_CHECK(message.find("even of the same model") != std::string::npos);

    // The first engine fails: nothing else is loaded, so the device hint.
    FakeModelRegistry first = BuildCascadeRegistry();
    first.SetCreateFailure("yolov8n", "device is not ready");
    code = GraphErrorCode::kGraphSchema;
    message = BuildError(CascadeJson(), first, &code);
    GRAPH_CHECK(code == GraphErrorCode::kModelLoad);
    GRAPH_CHECK(message.find("node \"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("dxrt-cli -s") != std::string::npos);
    GRAPH_CHECK(message.find("python3 scripts/download_models.py --models yolov8n --output /models") !=
                std::string::npos);
    GRAPH_CHECK(message.find("setup.sh") == std::string::npos);
    GRAPH_CHECK(message.find("device memory") == std::string::npos);
}

void TestStaticRegistryCarriesModelZooNames() {
    StaticModelRegistry registry;
    const ModelInfo* obb = registry.find("yolo26l_obb");
    GRAPH_CHECK(obb != NULL && obb->download_name == "yolo26l-obb");
    const ModelInfo* od = registry.find("yolov8n");
    GRAPH_CHECK(od != NULL && od->download_name == "YoloV8N");
    const ModelInfo* sr = registry.find("realesrgan_x2");
    GRAPH_CHECK(sr != NULL && sr->download_name == "RealESRGAN_x2-1");
    // An alias_of row is its variant, download name included (R6). Every
    // .dxnn of 8d0b748's registry is in the manifest, so the no-entry case
    // (FOLLOWUPS U-57) is pinned by tests/scripts/test_gen_model_registry.py.
    const ModelInfo* alias = registry.find("deit_base384_distilled");
    GRAPH_CHECK(alias != NULL && alias->download_name == "DeiTBaseDistilled-2");
}

namespace {

/// Two detector branches of ONE source, both routing ROI crops into one
/// shared "reid" node. Nothing in ValidateGraph forbids two ROI parents on
/// one node, and free composition through JSON is the point of this
/// project, so this is a graph a user can write today. (Before SP2 this had
/// two sources, which only worked because every source was given the same
/// frame - U-01. Two sources are two streams now: see TwoStreamFanInJson.)
std::string FanInJson() {
    return "{\"version\":1,\"name\":\"fanin\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"odA\",\"model\":\"yolov8n\"},"
           "{\"id\":\"odB\",\"model\":\"yolov8n\"},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"odA\"},"
           "{\"from\":\"cam\",\"to\":\"odB\"},"
           "{\"from\":\"odA\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}},"
           "{\"from\":\"odB\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

}  // namespace

// Review round 1, fix 2 — a node with two ROI parents must see both
// parents' crops. graph_runner_sync.cpp used to do
// roi_inbox[edge.to] = crops, which overwrites: the second parent processed
// silently discarded the first parent's contribution. Each detector in
// BuildCascadeRegistry's registry emits 2 boxes for "person", so a correct
// merge delivers 4 crops to "reid", not 2.
void TestSyncExecutorMergesRoisFromTwoParents() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(FanInJson(), "fanin.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.roi_results.count("reid") == 1);
    GRAPH_CHECK(report.roi_results["reid"].size() == 4);
    if (report.roi_results["reid"].size() != 4) return;

    int from_a = 0, from_b = 0;
    for (std::size_t i = 0; i < report.roi_results["reid"].size(); ++i) {
        const std::string& parent =
            report.roi_results["reid"][i].origin.parent_node;
        if (parent == "odA") ++from_a;
        else if (parent == "odB") ++from_b;
    }
    GRAPH_CHECK(from_a == 2);
    GRAPH_CHECK(from_b == 2);

    // Canonical order (ByOrigin): grouped by parent_node first.
    GRAPH_CHECK(report.roi_results["reid"][0].origin.parent_node == "odA");
    GRAPH_CHECK(report.roi_results["reid"][1].origin.parent_node == "odA");
    GRAPH_CHECK(report.roi_results["reid"][2].origin.parent_node == "odB");
    GRAPH_CHECK(report.roi_results["reid"][3].origin.parent_node == "odB");
}

// Review round 1, fix 4 — pin the defensive kModelUnknown throw added in
// StageGraph::Build for when the registry lookup misses. This path is never
// reached through ValidateGraph (which already rejects an unknown model),
// so it needs its own hand-built, unvalidated GraphSpec to exercise it.
void TestStageGraphBuildThrowsModelUnknownWhenRegistryLookupMisses() {
    GraphSpec spec;
    spec.version = 1;
    spec.name = "t";

    NodeSpec source;
    source.id = "cam";
    source.is_source = true;
    source.uri = "a.jpg";
    spec.nodes.push_back(source);

    NodeSpec model;
    model.id = "od";
    model.model = "not_in_the_registry";
    spec.nodes.push_back(model);

    EdgeSpec edge;
    edge.from = "cam";
    edge.to = "od";
    spec.edges.push_back(edge);

    FakeModelRegistry registry;  // empty: "not_in_the_registry" is unknown.

    StageGraph graph;
    bool threw = false;
    try {
        graph.Build(spec, registry, "/models", false);  // no ValidateGraph
    } catch (const GraphError& error) {
        threw = true;
        GRAPH_CHECK(error.code() == GraphErrorCode::kModelUnknown);
    }
    GRAPH_CHECK(threw);
}

// =============================================================================
// Review round 2 — direct tests for operator==(FrameReport, FrameReport),
// specifically its payload comparator SamePayload(). Nothing in Task 9 itself
// calls this operator (it exists for Task 10's parity test), so these build
// FrameReports by hand and call operator== directly rather than going through
// either executor — exactly what a unit test for a comparison operator should
// do, and it needs nothing from Task 10.
// =============================================================================

namespace {

BoxItem MakeBoxItemForEquality(float x, float score, int class_id,
                               const std::string& class_name) {
    BoxItem item;
    item.box = cv::Rect2f(x, 10.f, 40.f, 60.f);
    item.score = score;
    item.class_id = class_id;
    item.class_name = class_name;
    return item;
}

StageDataPtr MakeFramePayloadForEquality(int fill_value) {
    std::shared_ptr<FrameData> data(new FrameData());
    data->image = cv::Mat(4, 4, CV_8UC1, cv::Scalar(fill_value));
    return data;
}

StageDataPtr MakeVectorPayloadForEquality(float value) {
    std::shared_ptr<VectorData> data(new VectorData());
    data->values.push_back(value);
    return data;
}

/// A FrameReport with exactly one node_results entry, all other fields at
/// their defaults, so two reports built this way differ only in `data`.
FrameReport MakeNodeReport(const std::string& node_id, StageDataPtr data) {
    FrameReport report;
    StageResult result;
    result.data = data;
    report.node_results[node_id] = result;
    return report;
}

/// A FrameReport with exactly one roi_results entry (one crop), all other
/// fields at their defaults.
FrameReport MakeRoiReport(const std::string& node_id, const RoiRef& origin,
                          StageDataPtr data) {
    FrameReport report;
    StageResult result;
    result.data = data;
    result.origin = origin;
    report.roi_results[node_id].push_back(result);
    return report;
}

}  // namespace

// Case 1a (kFrame) — same shape, different payload must not be equal.
void TestOperatorEqualityDetectsFrameDivergence() {
    FrameReport a = MakeNodeReport("cam", MakeFramePayloadForEquality(10));
    FrameReport b = MakeNodeReport("cam", MakeFramePayloadForEquality(20));
    GRAPH_CHECK(!(a == b));
    GRAPH_CHECK(a != b);
}

// Case 1b (kVector) — same shape, different payload must not be equal.
void TestOperatorEqualityDetectsVectorDivergence() {
    FrameReport a = MakeNodeReport("reid", MakeVectorPayloadForEquality(0.5f));
    FrameReport b = MakeNodeReport("reid", MakeVectorPayloadForEquality(0.6f));
    GRAPH_CHECK(!(a == b));
    GRAPH_CHECK(a != b);
}

// Case 1c (kBoxes) — vary each field SameBoxItem compares in turn: a
// coordinate, a score, the class, and the item count. A comparator that
// silently skips any one of these must fail exactly that sub-check.
void TestOperatorEqualityDetectsBoxesFieldDivergence() {
    const BoxItem base = MakeBoxItemForEquality(10.f, 0.9f, 1, "person");

    // Coordinate differs.
    {
        BoxItem moved = base;
        moved.box.x += 5.f;
        std::shared_ptr<BoxesData> pa(new BoxesData(Shape::kBoxes));
        pa->items.push_back(base);
        std::shared_ptr<BoxesData> pb(new BoxesData(Shape::kBoxes));
        pb->items.push_back(moved);
        FrameReport a = MakeNodeReport("od", pa);
        FrameReport b = MakeNodeReport("od", pb);
        GRAPH_CHECK(!(a == b));
    }

    // Score differs.
    {
        BoxItem rescored = base;
        rescored.score = 0.1f;
        std::shared_ptr<BoxesData> pa(new BoxesData(Shape::kBoxes));
        pa->items.push_back(base);
        std::shared_ptr<BoxesData> pb(new BoxesData(Shape::kBoxes));
        pb->items.push_back(rescored);
        FrameReport a = MakeNodeReport("od", pa);
        FrameReport b = MakeNodeReport("od", pb);
        GRAPH_CHECK(!(a == b));
    }

    // Class differs.
    {
        BoxItem reclassed = base;
        reclassed.class_id = 2;
        reclassed.class_name = "car";
        std::shared_ptr<BoxesData> pa(new BoxesData(Shape::kBoxes));
        pa->items.push_back(base);
        std::shared_ptr<BoxesData> pb(new BoxesData(Shape::kBoxes));
        pb->items.push_back(reclassed);
        FrameReport a = MakeNodeReport("od", pa);
        FrameReport b = MakeNodeReport("od", pb);
        GRAPH_CHECK(!(a == b));
    }

    // Count differs.
    {
        std::shared_ptr<BoxesData> pa(new BoxesData(Shape::kBoxes));
        pa->items.push_back(base);
        std::shared_ptr<BoxesData> pb(new BoxesData(Shape::kBoxes));
        pb->items.push_back(base);
        pb->items.push_back(MakeBoxItemForEquality(200.f, 0.5f, 3, "bike"));
        FrameReport a = MakeNodeReport("od", pa);
        FrameReport b = MakeNodeReport("od", pb);
        GRAPH_CHECK(!(a == b));
    }
}

// Case 2 — identical payload, two separately-allocated StageData objects,
// must compare equal. Without this, a comparator that returns false for
// everything would still pass every check in case 1.
void TestOperatorEqualitySamePayloadDistinctObjectsEqual() {
    // kFrame
    {
        std::shared_ptr<FrameData> pa(new FrameData());
        pa->image = cv::Mat(4, 4, CV_8UC1, cv::Scalar(42));
        std::shared_ptr<FrameData> pb(new FrameData());
        pb->image = cv::Mat(4, 4, CV_8UC1, cv::Scalar(42));
        GRAPH_CHECK(static_cast<const void*>(pa.get()) !=
                    static_cast<const void*>(pb.get()));
        FrameReport a = MakeNodeReport("cam", pa);
        FrameReport b = MakeNodeReport("cam", pb);
        GRAPH_CHECK(a == b);
    }
    // kBoxes
    {
        std::shared_ptr<BoxesData> pa(new BoxesData(Shape::kBoxes));
        pa->items.push_back(MakeBoxItemForEquality(10.f, 0.9f, 1, "person"));
        std::shared_ptr<BoxesData> pb(new BoxesData(Shape::kBoxes));
        pb->items.push_back(MakeBoxItemForEquality(10.f, 0.9f, 1, "person"));
        GRAPH_CHECK(static_cast<const void*>(pa.get()) !=
                    static_cast<const void*>(pb.get()));
        FrameReport a = MakeNodeReport("od", pa);
        FrameReport b = MakeNodeReport("od", pb);
        GRAPH_CHECK(a == b);
    }
    // kVector
    {
        FrameReport a = MakeNodeReport("reid", MakeVectorPayloadForEquality(0.5f));
        FrameReport b = MakeNodeReport("reid", MakeVectorPayloadForEquality(0.5f));
        GRAPH_CHECK(a == b);
    }
}

// Case 3 (spec C1) — every shape has an exact comparator. These replace
// the old fail-closed pin (TestOperatorEqualityUnimplementedShapeNeverEqual)
// now that kObBoxes, kInstances and kBoxes3d are compared for real.

namespace {

/// Two BoxesData of `shape` holding `x` and `y`, compared as one node.
bool SameBoxesPayload(Shape shape, const BoxItem& x, const BoxItem& y) {
    std::shared_ptr<BoxesData> pa(new BoxesData(shape));
    pa->items.push_back(x);
    std::shared_ptr<BoxesData> pb(new BoxesData(shape));
    pb->items.push_back(y);
    return MakeNodeReport("od", pa) == MakeNodeReport("od", pb);
}

/// An 8x6 CV_8UC1 instance mask with a small filled block, freshly
/// allocated, so two calls give equal pixels in distinct buffers.
cv::Mat MakeInstanceMaskForEquality() {
    cv::Mat mask = cv::Mat::zeros(6, 8, CV_8UC1);
    mask(cv::Rect(2, 1, 3, 3)).setTo(cv::Scalar(255));
    return mask;
}

Detection3DResult MakeDetection3DForEquality() {
    Detection3DResult item;
    item.class_id = 2;
    item.class_name = "car";
    item.confidence = 0.8f;
    item.bev_x = 10.f;
    item.bev_y = 20.f;
    item.bev_w = 30.f;
    item.bev_h = 40.f;
    item.x3d = 1.5f;
    item.y3d = 2.5f;
    item.z3d = 3.5f;
    item.dim_h = 1.25f;
    item.dim_w = 1.75f;
    item.dim_l = 4.5f;
    item.yaw = 0.3f;
    return item;
}

bool SameBoxes3dPayload(const Detection3DResult& x, const Detection3DResult& y) {
    std::shared_ptr<Boxes3dData> pa(new Boxes3dData());
    pa->items.push_back(x);
    std::shared_ptr<Boxes3dData> pb(new Boxes3dData());
    pb->items.push_back(y);
    return MakeNodeReport("bev", pa) == MakeNodeReport("bev", pb);
}

/// A fully populated payload of `shape`, freshly allocated on every call,
/// so two calls give identical contents in distinct objects. No default:
/// a Shape added without a case here is a build error (-Werror=switch).
/// True only for the last Shape enumerator. Exhaustive, no default: a
/// Shape appended after kRecords is a build error here (-Werror=switch)
/// until it is given a case, and moving the "true" to it is what makes
/// the every-shape loop below visit it.
bool IsLastShape(Shape shape) {
    switch (shape) {
        case Shape::kFrame:
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances:
        case Shape::kKeypoints:
        case Shape::kLabelMap:
        case Shape::kDenseMap:
        case Shape::kImage:
        case Shape::kScores:
        case Shape::kVector:
            return false;
        case Shape::kBoxes3d:
            return false;
        case Shape::kRecords:
            return true;
    }
    return true;  // outside the enum: stop the loop
}

StageDataPtr MakeIdenticalPayloadForShape(Shape shape) {
    switch (shape) {
        case Shape::kFrame: {
            std::shared_ptr<FrameData> data(new FrameData());
            data->image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(1, 2, 3));
            return data;
        }
        case Shape::kBoxes:
        case Shape::kObBoxes:
        case Shape::kInstances: {
            std::shared_ptr<BoxesData> data(new BoxesData(shape));
            BoxItem item = MakeBoxItemForEquality(10.f, 0.9f, 1, "person");
            item.track_id = 4;
            item.landmarks.push_back(Keypoint(1.f, 2.f, 0.5f));
            if (shape == Shape::kObBoxes) item.angle = 0.25f;
            if (shape == Shape::kInstances) item.mask = MakeInstanceMaskForEquality();
            data->items.push_back(item);
            return data;
        }
        case Shape::kKeypoints: {
            std::shared_ptr<KeypointsData> data(new KeypointsData());
            PoseResult pose;
            pose.confidence = 0.7f;
            pose.box.push_back(1.f); pose.box.push_back(2.f);
            pose.box.push_back(30.f); pose.box.push_back(40.f);
            pose.keypoints.push_back(Keypoint(5.f, 6.f, 0.9f));
            data->items.push_back(pose);
            return data;
        }
        case Shape::kLabelMap: {
            std::shared_ptr<LabelMapData> data(new LabelMapData());
            data->labels = cv::Mat(4, 4, CV_8UC1, cv::Scalar(3));
            return data;
        }
        case Shape::kDenseMap: {
            std::shared_ptr<DenseMapData> data(new DenseMapData());
            data->values = cv::Mat(4, 4, CV_32FC1, cv::Scalar(0.5));
            return data;
        }
        case Shape::kImage: {
            std::shared_ptr<ImageData> data(new ImageData());
            data->image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(7, 8, 9));
            return data;
        }
        case Shape::kScores: {
            std::shared_ptr<ScoresData> data(new ScoresData());
            ClassificationResult top;
            top.class_id = 5;
            top.class_name = "tabby";
            top.confidence = 0.6f;
            top.top_k.push_back(std::make_pair(5, 0.6f));
            data->items.push_back(top);
            return data;
        }
        case Shape::kVector: {
            std::shared_ptr<VectorData> data(new VectorData());
            data->values.push_back(0.5f);
            data->values.push_back(-0.25f);
            return data;
        }
        case Shape::kBoxes3d: {
            std::shared_ptr<Boxes3dData> data(new Boxes3dData());
            data->items.push_back(MakeDetection3DForEquality());
            return data;
        }
        case Shape::kRecords: {
            std::shared_ptr<RecordsData> data(new RecordsData());
            RecordItem item;
            item.numbers.push_back(std::make_pair(std::string("pitch"), 1.5));
            item.text.push_back(std::make_pair(std::string("class_name"), std::string("box")));
            data->items.push_back(item);
            return data;
        }
    }
    return StageDataPtr();
}

}  // namespace

// kObBoxes: an identical pair is equal; the angle alone tells two apart.
void TestOperatorEqualityComparesObBoxes() {
    BoxItem base = MakeBoxItemForEquality(10.f, 0.9f, 1, "ship");
    base.angle = 0.5f;
    GRAPH_CHECK(SameBoxesPayload(Shape::kObBoxes, base, base));

    BoxItem turned = base;
    turned.angle = 0.75f;
    GRAPH_CHECK(!SameBoxesPayload(Shape::kObBoxes, base, turned));
}

// kInstances: masks are compared pixel for pixel, not by emptiness.
void TestOperatorEqualityComparesInstanceMasks() {
    BoxItem base = MakeBoxItemForEquality(10.f, 0.9f, 1, "person");
    base.mask = MakeInstanceMaskForEquality();
    BoxItem same = base;
    same.mask = MakeInstanceMaskForEquality();  // equal pixels, own buffer
    GRAPH_CHECK(base.mask.data != same.mask.data);
    GRAPH_CHECK(SameBoxesPayload(Shape::kInstances, base, same));

    // One pixel differs.
    BoxItem one_pixel = base;
    one_pixel.mask = MakeInstanceMaskForEquality();
    one_pixel.mask.at<unsigned char>(5, 7) = 1;
    GRAPH_CHECK(!SameBoxesPayload(Shape::kInstances, base, one_pixel));

    // One has a mask, the other does not.
    BoxItem no_mask = base;
    no_mask.mask = cv::Mat();
    GRAPH_CHECK(!SameBoxesPayload(Shape::kInstances, base, no_mask));

    // Same pixels, different size.
    BoxItem wider = base;
    wider.mask = cv::Mat::zeros(6, 9, CV_8UC1);
    MakeInstanceMaskForEquality().copyTo(wider.mask(cv::Rect(0, 0, 8, 6)));
    GRAPH_CHECK(!SameBoxesPayload(Shape::kInstances, base, wider));

    // Same size and pixel values, different type.
    BoxItem deeper = base;
    MakeInstanceMaskForEquality().convertTo(deeper.mask, CV_16UC1);
    GRAPH_CHECK(deeper.mask.size() == base.mask.size());
    GRAPH_CHECK(!SameBoxesPayload(Shape::kInstances, base, deeper));
}

// kBoxes3d: an identical pair is equal; each of Detection3DResult's 14
// fields alone tells two apart (yaw is the spec's named case).
void TestOperatorEqualityComparesBoxes3d() {
    const Detection3DResult base = MakeDetection3DForEquality();
    GRAPH_CHECK(SameBoxes3dPayload(base, MakeDetection3DForEquality()));

    Detection3DResult yawed = base;
    yawed.yaw = 0.4f;
    GRAPH_CHECK(!SameBoxes3dPayload(base, yawed));

    Detection3DResult other = base;
    other.class_id = 3;
    GRAPH_CHECK(!SameBoxes3dPayload(base, other));
    other = base;
    other.class_name = "truck";
    GRAPH_CHECK(!SameBoxes3dPayload(base, other));

    float Detection3DResult::* const floats[] = {
        &Detection3DResult::confidence, &Detection3DResult::bev_x,
        &Detection3DResult::bev_y,      &Detection3DResult::bev_w,
        &Detection3DResult::bev_h,      &Detection3DResult::x3d,
        &Detection3DResult::y3d,        &Detection3DResult::z3d,
        &Detection3DResult::dim_h,      &Detection3DResult::dim_w,
        &Detection3DResult::dim_l,      &Detection3DResult::yaw};
    for (std::size_t f = 0; f < sizeof(floats) / sizeof(floats[0]); ++f) {
        Detection3DResult varied = base;
        varied.*floats[f] += 1.f;
        const bool equal = SameBoxes3dPayload(base, varied);
        if (equal) std::printf("      boxes3d float field %d not compared\n", static_cast<int>(f));
        GRAPH_CHECK(!equal);
    }

    // Item count differs.
    std::shared_ptr<Boxes3dData> pa(new Boxes3dData());
    pa->items.push_back(base);
    std::shared_ptr<Boxes3dData> pb(new Boxes3dData());
    pb->items.push_back(base);
    pb->items.push_back(base);
    GRAPH_CHECK(!(MakeNodeReport("bev", pa) == MakeNodeReport("bev", pb)));
}

// Every Shape value: two identical payloads in distinct objects compare
// equal. A shape that still falls back to the fail-closed default fails
// here. The loop runs from the first enumerator until IsLastShape; both
// IsLastShape and MakeIdenticalPayloadForShape switch with no default, so
// -Werror=switch forces a new Shape into both, and SamePayload's own
// switch (dxapp_graph_obj, also -Werror=switch) forces its comparator.
void TestOperatorEqualityEveryShapeComparesIdenticalPayloadsEqual() {
    int shapes = 0;
    int visited = 0;
    for (int s = static_cast<int>(Shape::kFrame);; ++s) {
        const Shape shape = static_cast<Shape>(s);
        ++visited;
        const StageDataPtr pa = MakeIdenticalPayloadForShape(shape);
        const StageDataPtr pb = MakeIdenticalPayloadForShape(shape);
        GRAPH_CHECK(pa && pb);
        if (pa && pb) {
            ++shapes;
            GRAPH_CHECK(pa.get() != pb.get());
            GRAPH_CHECK(pa->shape() == shape);
            const bool equal = MakeNodeReport("n", pa) == MakeNodeReport("n", pb);
            if (!equal) {
                std::printf("      shape %d (%s): identical payloads unequal\n", s,
                            ToString(shape));
            }
            GRAPH_CHECK(equal);
        }
        if (IsLastShape(shape)) break;
    }
    // Every enumerator the loop visited had a payload builder; the count
    // comes from the loop, not a constant that a new Shape would outgrow.
    GRAPH_CHECK(shapes == visited);
    GRAPH_CHECK(shapes >= 11);
}

// Case 4 — the roi_results payload path (the gap found in round 1): two
// reports whose RoiRef fields all agree but whose .data payloads differ
// must compare unequal.
void TestOperatorEqualityDetectsRoiResultsPayloadDivergence() {
    RoiRef origin;
    origin.from_roi = true;
    origin.parent_node = "od";
    origin.parent_index = 0;
    origin.roi_index = 0;
    origin.track_id = 3;
    origin.src_box = cv::Rect2f(1.f, 2.f, 3.f, 4.f);

    FrameReport a = MakeRoiReport("reid", origin, MakeVectorPayloadForEquality(0.5f));
    FrameReport b = MakeRoiReport("reid", origin, MakeVectorPayloadForEquality(0.9f));

    // Sanity: every RoiRef field really is identical between a and b (both
    // built from the same `origin`); only the payload differs, so a false
    // result below can only come from the payload comparison.
    GRAPH_CHECK(a.roi_results["reid"][0].origin.parent_index ==
                b.roi_results["reid"][0].origin.parent_index);
    GRAPH_CHECK(a.roi_results["reid"][0].origin.roi_index ==
                b.roi_results["reid"][0].origin.roi_index);
    GRAPH_CHECK(a.roi_results["reid"][0].origin.track_id ==
                b.roi_results["reid"][0].origin.track_id);
    GRAPH_CHECK(a.roi_results["reid"][0].origin.parent_node ==
                b.roi_results["reid"][0].origin.parent_node);
    GRAPH_CHECK(a.roi_results["reid"][0].origin.src_box ==
                b.roi_results["reid"][0].origin.src_box);

    GRAPH_CHECK(!(a == b));
}

// Review round 3, item 2 — close the other half of the origin comparison:
// inv_align and from_roi were still uncompared. Vary each alone against an
// otherwise-identical origin and identical payload, mirroring the round-2
// per-field style (TestOperatorEqualityDetectsBoxesFieldDivergence).
void TestOperatorEqualityDetectsRoiOriginFieldDivergence() {
    RoiRef base;
    base.from_roi = true;
    base.parent_node = "od";
    base.parent_index = 0;
    base.roi_index = 0;
    base.track_id = 3;
    base.src_box = cv::Rect2f(1.f, 2.f, 3.f, 4.f);
    // base.inv_align defaults to identity (RoiRef's own constructor).

    // inv_align differs.
    {
        RoiRef varied = base;
        varied.inv_align = cv::Matx23f(1.f, 0.f, 5.f, 0.f, 1.f, 0.f);  // translated
        FrameReport a = MakeRoiReport("reid", base, MakeVectorPayloadForEquality(0.5f));
        FrameReport b = MakeRoiReport("reid", varied, MakeVectorPayloadForEquality(0.5f));
        GRAPH_CHECK(!(a == b));
    }

    // from_roi differs.
    {
        RoiRef varied = base;
        varied.from_roi = false;
        FrameReport a = MakeRoiReport("reid", base, MakeVectorPayloadForEquality(0.5f));
        FrameReport b = MakeRoiReport("reid", varied, MakeVectorPayloadForEquality(0.5f));
        GRAPH_CHECK(!(a == b));
    }
}

// =============================================================================
// Task 10 — the asynchronous executor, and its parity with the synchronous
// one. SyncExecutor is the oracle: every "matches sync" case below builds the
// SAME GraphSpec twice against two independent registries, runs one report
// through each executor, and requires operator==(FrameReport, FrameReport) to
// hold. (The brief calls that comparison SameReport(); Task 9 shipped it as
// operator==, so these call the operator directly rather than adding a
// one-line alias that would have to be kept in step with it.)
//
// Fixture shapes matter here. operator=='s payload comparator fails closed on
// any shape it does not implement (stage_graph.cpp SamePayload: kFrame,
// kBoxes and kVector only), so a parity fixture must not carry a kLabelMap or
// kDenseMap payload in node_results and expect equality — the multi-failure
// case below relies on all three of its models failing, which is exactly why
// no unimplemented payload ever reaches the comparator there.
// =============================================================================

namespace {

/// cam -> od, cam -> seg, cam -> dep, but with the EDGES declared in reverse
/// node order. StageGraph::Build's Kahn pass is lowest-node-index-first, so
/// the synchronous executor still visits od, seg, dep in that order, while
/// the asynchronous executor's ready queue is fed in edge-declaration order
/// and therefore visits dep, seg, od. Any report field whose value depends on
/// "which node was processed last" diverges on this fixture and is caught.
std::string ReversedEdgeFanOutJson() {
    return "{\"version\":1,\"name\":\"fanout\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},"
           "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
           "{\"id\":\"dep\",\"model\":\"fastdepth_1\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"dep\"},"
           "{\"from\":\"cam\",\"to\":\"seg\"},"
           "{\"from\":\"cam\",\"to\":\"od\"}]}";
}

FakeModelRegistry BuildFanOutRegistry() {
    FakeModelRegistry registry;

    ModelInfo detector;
    detector.model_name = "yolov8n";
    detector.task = "object_detection";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.ready = true;
    registry.AddModel(detector, StageDataPtr(new BoxesData(Shape::kBoxes)));

    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;
    registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));

    ModelInfo depth;
    depth.model_name = "fastdepth_1";
    depth.task = "depth_estimation";
    depth.output_shape = Shape::kDenseMap;
    depth.input_contract = InputContract::kFullFrame;
    depth.ready = true;
    registry.AddModel(depth, StageDataPtr(new DenseMapData()));

    return registry;
}

}  // namespace

namespace {
/// The previous meaning of the old one-number constructor, spelled out.
AsyncOptions StageJobs(std::size_t jobs) {
    AsyncOptions options;
    options.max_jobs_per_stage = jobs;
    return options;
}
}  // namespace

// Reusing a one-number constructor for a new meaning would reinterpret every
// existing call site silently - the exact failure that turned --max-inflight
// from "frames" into "jobs per stage" without anyone noticing.
static_assert(!std::is_constructible<AsyncExecutor, std::size_t>::value,
              "AsyncExecutor must not be constructible from a bare number");

void TestAsyncMatchesSyncOnCascade() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    GRAPH_CHECK(actual.roi_results["reid"].size() == 2);
}

// Parity rule 1 — deliver the two ROI completions in reverse order and the
// report must still be identical. This is the case that fails the moment the
// ByOrigin sort is removed: the crops are collected in completion order, so
// reverse delivery puts parent_index 1 first and misattributes every ROI
// result to the wrong parent box downstream.
void TestAsyncSortsOutOfOrderRoiCompletions() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry ordered = BuildCascadeRegistry();
    StageGraph ordered_graph;
    ordered_graph.Build(spec, ordered, "/models", false);
    AsyncExecutor forward(StageJobs(4));
    FrameReport expected = forward.RunFrame(ordered_graph, frame, 0);

    FakeModelRegistry reversed = BuildCascadeRegistry();
    reversed.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph reversed_graph;
    reversed_graph.Build(spec, reversed, "/models", false);
    AsyncExecutor backward(StageJobs(4));
    FrameReport actual = backward.RunFrame(reversed_graph, frame, 0);

    // The reversed stage really is the one the graph is holding, so the
    // fixture cannot silently degrade into a second forward-delivery run.
    GRAPH_CHECK(reversed.last_stage("casvit_t") != NULL);

    GRAPH_CHECK(expected == actual);
    GRAPH_CHECK(actual.roi_results["reid"].size() == 2);
    if (actual.roi_results["reid"].size() != 2) return;
    GRAPH_CHECK(actual.roi_results["reid"][0].origin.parent_index == 0);
    GRAPH_CHECK(actual.roi_results["reid"][1].origin.parent_index == 1);
}

// Review Focus #1 — a detector that returns nothing must still let the frame
// finalize. A counter that only decrements on a child completion hangs here.
void TestAsyncCompletesWhenProducerIsEmpty() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    ModelInfo detector = *registry.find("yolov8n");
    registry.AddModel(detector, StageDataPtr(new BoxesData(Shape::kBoxes)));

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(report.roi_results.count("reid") == 1);
    GRAPH_CHECK(report.roi_results["reid"].empty());
}

// Parity rule 2 — track ids must not depend on which frame finished first.
// RunFrame is frame-atomic (every stage is flushed before the frame's report
// is returned), so the tracker cannot be handed frame 1 while frame 0 is
// still in flight.
void TestAsyncKeepsTrackIdsFrameOrdered() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    AsyncExecutor executor(StageJobs(4));
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport first = executor.RunFrame(graph, frame, 0);
    FrameReport second = executor.RunFrame(graph, frame, 1);

    GRAPH_CHECK(first.roi_results["reid"].size() == 2);
    GRAPH_CHECK(second.roi_results["reid"].size() == 2);
    if (first.roi_results["reid"].size() != 2) return;
    if (second.roi_results["reid"].size() != 2) return;
    GRAPH_CHECK(first.roi_results["reid"][0].origin.track_id >= 0);
    GRAPH_CHECK(first.roi_results["reid"][0].origin.track_id ==
                second.roi_results["reid"][0].origin.track_id);
    GRAPH_CHECK(first.roi_results["reid"][1].origin.track_id ==
                second.roi_results["reid"][1].origin.track_id);
}

void TestAsyncIsolatesStageFailure() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFailure("casvit_t", "simulated inference failure");
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(!report.error.empty());
    GRAPH_CHECK(report.error.find("simulated") != std::string::npos);
    GRAPH_CHECK(report.node_results.count("od") == 1);
}

void TestAsyncFanOutRunsAllBranches() {
    FakeModelRegistry registry = BuildFanOutRegistry();

    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"dep\",\"model\":\"fastdepth_1\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"cam\",\"to\":\"dep\"}]}",
        "fanout.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(report.node_results.count("seg") == 1);
    GRAPH_CHECK(report.node_results.count("dep") == 1);
}

// Task 9's review round 1, fix 2, ported to the async side: a node with two
// ROI parents must see both parents' crops. The synchronous executor had this
// bug (roi_inbox[to] = crops, which overwrites); the async executor writes
// into the same kind of per-consumer inbox and can reintroduce it
// independently, so the parity suite has to pin it here too.
void TestAsyncMatchesSyncOnFanIn() {
    GraphSpec spec = ParseGraphText(FanInJson(), "fanin.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    // Reverse delivery on the shared consumer: its four crops arrive
    // interleaved across two parents and in reverse order within each batch.
    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(expected.roi_results["reid"].size() == 4);
    GRAPH_CHECK(expected == actual);
}

// Back-pressure must not change the answer. max_inflight 1 submits and
// flushes one crop at a time (so completions arrive strictly in submission
// order); max_inflight 4 submits all of them and lets the stage hand them
// back in reverse. Both must equal the synchronous report.
void TestAsyncMatchesSyncUnderBackPressure() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry one_registry = BuildCascadeRegistry();
    one_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph one_graph;
    one_graph.Build(spec, one_registry, "/models", false);
    AsyncExecutor one(StageJobs(1));
    GRAPH_CHECK(one.options().max_jobs_per_stage == 1);
    GRAPH_CHECK(expected == one.RunFrame(one_graph, frame, 0));

    FakeModelRegistry many_registry = BuildCascadeRegistry();
    many_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph many_graph;
    many_graph.Build(spec, many_registry, "/models", false);
    AsyncExecutor many(StageJobs(4));
    GRAPH_CHECK(expected == many.RunFrame(many_graph, frame, 0));

    // The reports being equal is the point, but it is also what makes the
    // limit invisible in them, so observe the limit where it acts: the
    // detector emits two crops, so an executor honouring max_inflight=1
    // never has both outstanding, while max_inflight=4 does.
    GRAPH_CHECK(one_registry.last_stage("casvit_t") != NULL);
    GRAPH_CHECK(many_registry.last_stage("casvit_t") != NULL);
    if (one_registry.last_stage("casvit_t") == NULL) return;
    if (many_registry.last_stage("casvit_t") == NULL) return;
    // Expected values, fixture-relatively: a ROI stage's peak own-queue
    // depth is min(crop count, max_inflight). The detector emits 2 person
    // boxes, so min(2, 1) = 1 and min(2, 4) = 2. These are per-STAGE and so
    // are insensitive to edge-declaration order, unlike the cross-stage
    // peaks in TestAsyncOverlapsSiblingNodes / TestAsyncOverlapsAcrossDepth.
    GRAPH_CHECK(one_registry.last_stage("casvit_t")->max_pending() == 1);
    GRAPH_CHECK(many_registry.last_stage("casvit_t")->max_pending() == 2);
}

// A producer that fails must leave its consumer in exactly the state the
// synchronous executor leaves it in: present in roi_results, empty, not
// missing and not hung. The async scheduler has to release a failed node's
// outgoing edges even though it propagates no data along them, or the
// consumer never becomes ready.
void TestAsyncMatchesSyncWhenProducerFails() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    sync_registry.SetFailure("yolov8n", "simulated detector failure");
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.SetFailure("yolov8n", "simulated detector failure");
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(expected.error.find("simulated detector failure") !=
                std::string::npos);
    GRAPH_CHECK(expected.node_results.count("od") == 0);
    GRAPH_CHECK(expected.roi_results.count("reid") == 1);
    GRAPH_CHECK(expected.roi_results["reid"].empty());
    GRAPH_CHECK(expected == actual);
}

// The frame-edge half of "release an edge without sending data along it".
// TestAsyncMatchesSyncWhenProducerFails covers a failed producer feeding a
// ROI edge; this covers one feeding a PLAIN edge, where the hazard is the
// mirror image. The consumer must still be released (or it hangs), but it
// must NOT be told a full frame is waiting for it — SyncExecutor never
// populates a consumer's frame inbox from a node that produced nothing, so
// "seg" ends up in roi_results, empty, rather than running on the raw frame
// and landing in node_results. Setting the frame flag unconditionally is the
// version of this loop that hides behind every other fixture.
//
// FINAL REVIEW ROUND: this fixture deliberately does NOT call
// ValidateGraph. A plain edge out of a MODEL node is now rejected
// (kGraphEdge, "a plain edge carries only a source frame or an image"),
// because both executors discard the producer's payload on it — the
// silent data loss that rejection exists to stop. The executors'
// defensive behaviour on such an edge is still worth pinning, exactly as
// StageGraph::Build's kModelUnknown guard is pinned although ValidateGraph
// makes it unreachable: Build() accepts an unvalidated spec, so the path
// exists.
// Constructing the graph without validating is therefore the point of this
// fixture, not an oversight.
void TestAsyncMatchesSyncWhenChainedProducerFails() {
    const char* kJson =
        "{\"version\":1,\"name\":\"chain\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"seg\"}]}";
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    GraphSpec spec = ParseGraphText(kJson, "chain.json");

    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    sync_registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));
    sync_registry.SetFailure("yolov8n", "detector down");
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));
    async_registry.SetFailure("yolov8n", "detector down");
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(expected.node_results.count("seg") == 0);
    GRAPH_CHECK(expected.roi_results.count("seg") == 1);
    GRAPH_CHECK(expected.roi_results["seg"].empty());
    GRAPH_CHECK(expected == actual);
}

// report.error holds one message, so when several nodes fail in one frame the
// two executors must agree on WHICH one survives. The synchronous executor
// overwrites in topological order, so the topologically last failure wins;
// the async executor dispatches in ready order, which on this fixture is the
// exact reverse. Resolving async's errors in topological order rather than in
// completion order is what keeps the two reports identical.
void TestAsyncMatchesSyncOnMultipleFailures() {
    GraphSpec spec = ParseGraphText(ReversedEdgeFanOutJson(), "fanout.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildFanOutRegistry();
    sync_registry.SetFailure("yolov8n", "detector down");
    sync_registry.SetFailure("bisenetv2", "segmenter down");
    sync_registry.SetFailure("fastdepth_1", "depth down");
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildFanOutRegistry();
    async_registry.SetFailure("yolov8n", "detector down");
    async_registry.SetFailure("bisenetv2", "segmenter down");
    async_registry.SetFailure("fastdepth_1", "depth down");
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    // "dep" is node index 3, so it is last in topological order and its
    // message is the one both executors must end up reporting. If async
    // reported completion order instead, this would read "detector down".
    GRAPH_CHECK(expected.error.find("depth down") != std::string::npos);
    GRAPH_CHECK(expected == actual);
}

// A narrow check, and its comment used to claim more than it does: these
// stages were DISPATCHED, so the per-node flush in the harvest loop already
// drained them and this would pass even with the end-of-frame DrainPending
// deleted. What it actually pins is that the harvest loop leaves nothing
// behind on a node it handled. The end-of-frame drain - the other half of
// "no work outlives its frame", and the half that matters for parity rule 2
// - is covered by TestAsyncDrainsNeverDispatchedStage.
void TestAsyncLeavesNoPendingWorkAfterFrame() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    AsyncExecutor executor(StageJobs(4));
    executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(registry.last_stage("yolov8n") != NULL);
    GRAPH_CHECK(registry.last_stage("casvit_t") != NULL);
    if (registry.last_stage("casvit_t") == NULL) return;
    GRAPH_CHECK(registry.last_stage("yolov8n")->pending() == 0);
    GRAPH_CHECK(registry.last_stage("casvit_t")->pending() == 0);
}

// Hardware fact (i_registry.hpp): a blocking Run() also fires the engine's
// registered callback, so one logical job can deliver twice. The executor
// must count a job as delivered once; otherwise the second fire appends a
// duplicate ROI result and the report silently reports three crops where the
// detector produced two.
void TestAsyncIgnoresDoubleDelivery() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.SetDoubleDelivery("yolov8n", true);
    async_registry.SetDoubleDelivery("casvit_t", true);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(actual.roi_results["reid"].size() == 2);
    GRAPH_CHECK(expected == actual);
}

// The companion to the case above. There, topological order and node-index
// order coincide, so it only proves async does not resolve errors in DISPATCH
// order. Here they deliberately disagree: "reid" is node index 1 but runs last
// topologically (it is behind "od", index 3), while "seg" is index 2 and runs
// earlier. Both fail, so the surviving message distinguishes a topological
// walk (reid) from an ascending-index walk over the per-node error map (seg) —
// two spellings that are equally deterministic and only one of which is what
// SyncExecutor does.
void TestAsyncResolvesErrorsInTopologicalOrder() {
    const char* kJson =
        "{\"version\":1,\"name\":\"skew\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    GraphSpec spec = ParseGraphText(kJson, "skew.json");

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;
    sync_registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));
    sync_registry.SetFailure("bisenetv2", "segmenter down");
    sync_registry.SetFailure("casvit_t", "reid down");
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));
    async_registry.SetFailure("bisenetv2", "segmenter down");
    async_registry.SetFailure("casvit_t", "reid down");
    async_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    // Confirms the fixture really is skewed: the topologically last failure
    // is "reid", not the higher-indexed "seg".
    GRAPH_CHECK(expected.error.find("reid down") != std::string::npos);
    GRAPH_CHECK(expected == actual);
}

void TestAsyncRejectsEmptyFrame() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    AsyncExecutor executor(StageJobs(4));
    FrameReport report = executor.RunFrame(graph, cv::Mat(), 0);

    GRAPH_CHECK(!report.error.empty());
    GRAPH_CHECK(report.error.find("empty") != std::string::npos);
    GRAPH_CHECK(report.node_results.empty());
}

// =============================================================================
// Task 10, fix round 1.
// =============================================================================

namespace {

cv::Mat MakeLabelMat(int seed) {
    cv::Mat labels(4, 4, CV_8U);
    for (int r = 0; r < labels.rows; ++r) {
        for (int c = 0; c < labels.cols; ++c) {
            labels.at<uchar>(r, c) = static_cast<uchar>((r * 4 + c + seed) % 7);
        }
    }
    return labels;
}

cv::Mat MakeDepthMat(float seed) {
    cv::Mat values(4, 4, CV_32F);
    for (int r = 0; r < values.rows; ++r) {
        for (int c = 0; c < values.cols; ++c) {
            values.at<float>(r, c) =
                seed + static_cast<float>(r) * 0.25f + static_cast<float>(c);
        }
    }
    return values;
}

StageDataPtr MakeLabelMapPayload(int seed) {
    std::shared_ptr<LabelMapData> data(new LabelMapData());
    data->labels = MakeLabelMat(seed);
    return data;
}

StageDataPtr MakeDenseMapPayload(float seed) {
    std::shared_ptr<DenseMapData> data(new DenseMapData());
    data->values = MakeDepthMat(seed);
    return data;
}

StageDataPtr MakeKeypointsPayload(float seed) {
    std::shared_ptr<KeypointsData> data(new KeypointsData());
    PoseResult pose;
    pose.confidence = 0.87f;
    pose.box.push_back(10.f); pose.box.push_back(20.f);
    pose.box.push_back(60.f); pose.box.push_back(120.f);
    for (int i = 0; i < 3; ++i) {
        Keypoint point;
        point.x = seed + static_cast<float>(i);
        point.y = static_cast<float>(i) * 2.f;
        point.confidence = 0.5f;
        pose.keypoints.push_back(point);
    }
    data->items.push_back(pose);
    return data;
}

ModelInfo MakeFullFrameInfo(const std::string& name, const std::string& task,
                            Shape shape) {
    ModelInfo info;
    info.model_name = name;
    info.task = task;
    info.dxnn_file = name + ".dxnn";
    info.output_shape = shape;
    info.input_contract = InputContract::kFullFrame;
    info.ready = true;
    return info;
}

/// The project's headline graph, whole and SUCCEEDING: a tracked detector
/// cascading into a ROI re-id stage, plus segmentation, pose and depth
/// branches off the same source. Every payload shape SamePayload implements
/// is present and populated, which is the point - before fix round 1 the
/// parity suite could only compare box-shaped payloads, because a fixture
/// with a live kLabelMap or kDenseMap node could never satisfy the
/// fail-closed comparator.
FakeModelRegistry BuildHeadlineRegistry() {
    FakeModelRegistry registry;

    ModelInfo detector =
        MakeFullFrameInfo("yolov8n", "object_detection", Shape::kBoxes);
    detector.input_width = 640;
    detector.input_height = 640;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10, 10, 40, 60, 0.8f, "person"));
    boxes->items.push_back(MakeItem(200, 20, 40, 60, 0.9f, "person"));
    registry.AddModel(detector, boxes);

    ModelInfo reid =
        MakeFullFrameInfo("casvit_t", "reid", Shape::kVector);
    reid.input_contract = InputContract::kRoi;
    std::shared_ptr<VectorData> vector(new VectorData());
    vector->values.push_back(0.5f);
    vector->values.push_back(-0.25f);
    registry.AddModel(reid, vector);

    registry.AddModel(
        MakeFullFrameInfo("bisenetv2", "semantic_segmentation", Shape::kLabelMap),
        MakeLabelMapPayload(0));
    registry.AddModel(
        MakeFullFrameInfo("yolov8n_pose", "pose_estimation", Shape::kKeypoints),
        MakeKeypointsPayload(1.f));
    registry.AddModel(
        MakeFullFrameInfo("fastdepth_1", "depth_estimation", Shape::kDenseMap),
        MakeDenseMapPayload(0.5f));

    return registry;
}

std::string HeadlineJson() {
    return "{\"version\":1,\"name\":\"headline\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"track\":{\"algo\":\"iou\"}},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"},"
           "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
           "{\"id\":\"pose\",\"model\":\"yolov8n_pose\"},"
           "{\"id\":\"dep\",\"model\":\"fastdepth_1\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"cam\",\"to\":\"seg\"},"
           "{\"from\":\"cam\",\"to\":\"pose\"},"
           "{\"from\":\"cam\",\"to\":\"dep\"},"
           "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

}  // namespace

// Direct tests for the three comparators fix round 1 added to SamePayload.
// The parity fixture below cannot stand in for these: it feeds both sides
// identical content, so a comparator hard-wired to return true would sail
// through it. These are the negative cases that make the positive one mean
// something, in the shape of the round-2 operator== tests.
void TestOperatorEqualityComparesLabelMapDenseMapAndKeypoints() {
    // kLabelMap — equal content, distinct objects.
    GRAPH_CHECK(MakeNodeReport("seg", MakeLabelMapPayload(0)) ==
                MakeNodeReport("seg", MakeLabelMapPayload(0)));
    // kLabelMap — one label index differs.
    GRAPH_CHECK(!(MakeNodeReport("seg", MakeLabelMapPayload(0)) ==
                  MakeNodeReport("seg", MakeLabelMapPayload(1))));
    // kLabelMap — same content, different geometry.
    {
        std::shared_ptr<LabelMapData> tiny(new LabelMapData());
        tiny->labels = cv::Mat::zeros(2, 2, CV_8U);
        std::shared_ptr<LabelMapData> large(new LabelMapData());
        large->labels = cv::Mat::zeros(4, 4, CV_8U);
        GRAPH_CHECK(!(MakeNodeReport("seg", tiny) ==
                      MakeNodeReport("seg", large)));
    }
    // kLabelMap — same element COUNT, different shape. An element-count
    // check instead of a size check would let a transposed or reshaped
    // label map compare equal, and a transposition is exactly the
    // segmentation defect this project already guards against elsewhere
    // (TestSegmentationConversionCatchesTransposition).
    {
        std::shared_ptr<LabelMapData> wide(new LabelMapData());
        wide->labels = cv::Mat::zeros(2, 8, CV_8U);
        std::shared_ptr<LabelMapData> square(new LabelMapData());
        square->labels = cv::Mat::zeros(4, 4, CV_8U);
        GRAPH_CHECK(wide->labels.total() == square->labels.total());
        GRAPH_CHECK(!(MakeNodeReport("seg", wide) ==
                      MakeNodeReport("seg", square)));
    }
    // kLabelMap — same geometry and same values, different element type.
    {
        std::shared_ptr<LabelMapData> bytes(new LabelMapData());
        bytes->labels = cv::Mat::zeros(4, 4, CV_8U);
        std::shared_ptr<LabelMapData> ints(new LabelMapData());
        ints->labels = cv::Mat::zeros(4, 4, CV_32S);
        GRAPH_CHECK(!(MakeNodeReport("seg", bytes) ==
                      MakeNodeReport("seg", ints)));
    }

    // kDenseMap — equal content, distinct objects, then one float apart.
    GRAPH_CHECK(MakeNodeReport("dep", MakeDenseMapPayload(0.5f)) ==
                MakeNodeReport("dep", MakeDenseMapPayload(0.5f)));
    GRAPH_CHECK(!(MakeNodeReport("dep", MakeDenseMapPayload(0.5f)) ==
                  MakeNodeReport("dep", MakeDenseMapPayload(0.5000001f))));

    // kDenseMap — NaN never satisfies parity, not even against itself. The
    // file documents this for the other shapes; kDenseMap is the payload
    // most likely to carry one, so it is asserted rather than assumed.
    {
        std::shared_ptr<DenseMapData> a(new DenseMapData());
        a->values = MakeDepthMat(0.5f);
        a->values.at<float>(1, 1) = std::numeric_limits<float>::quiet_NaN();
        std::shared_ptr<DenseMapData> b(new DenseMapData());
        b->values = MakeDepthMat(0.5f);
        b->values.at<float>(1, 1) = std::numeric_limits<float>::quiet_NaN();
        GRAPH_CHECK(!(MakeNodeReport("dep", a) == MakeNodeReport("dep", b)));
    }

    // kKeypoints — equal, then a moved point, then a different count.
    GRAPH_CHECK(MakeNodeReport("pose", MakeKeypointsPayload(1.f)) ==
                MakeNodeReport("pose", MakeKeypointsPayload(1.f)));
    GRAPH_CHECK(!(MakeNodeReport("pose", MakeKeypointsPayload(1.f)) ==
                  MakeNodeReport("pose", MakeKeypointsPayload(2.f))));
    {
        std::shared_ptr<KeypointsData> fewer(new KeypointsData());
        GRAPH_CHECK(!(MakeNodeReport("pose", MakeKeypointsPayload(1.f)) ==
                      MakeNodeReport("pose", fewer)));
    }
}

// The spec's dispatch rule, observed rather than inferred: "a node is
// submitted the moment its counter reaches zero". When "cam" completes it
// drops all three siblings' counters to zero at the same instant, so all
// three must be outstanding together. Submitting one, waiting for it, and
// submitting the next produces an IDENTICAL FrameReport - ByOrigin makes the
// output order-independent - so no output assertion can tell the two apart.
// The registry's cross-stage in-flight peak can.
void TestAsyncOverlapsSiblingNodes() {
    FakeModelRegistry registry = BuildFanOutRegistry();
    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"dep\",\"model\":\"fastdepth_1\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"cam\",\"to\":\"dep\"}]}",
        "fanout.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());

    // Expected peak, fixture-relatively: this graph is pure fan-out with no
    // cascade, so the peak is just the number of branches off "cam" - 3.
    // (The general form, for a graph with a ROI child, is
    //  (branches off cam) - 1 + (the ROI child's crop count): one branch is
    //  harvested to release the child, the rest are still outstanding when
    //  the child's crops are submitted. With no ROI child the second term
    //  is zero and the -1 does not apply, because nothing is harvested
    //  before the peak is reached.)
    //
    // Edit this number only alongside the fixture. It also encodes harvest
    // order, which is submission order, which for the nodes released by one
    // parent is EDGE-DECLARATION order - see the caveat on
    // TestAsyncOverlapsAcrossDepth. This graph is insensitive to that,
    // because none of its three branches has a child.
    GRAPH_CHECK(registry.peak_inflight() == 3);

    // ...and the 3 is cross-node overlap, not one stage batching: each of
    // the three stages only ever held a single job. Expected value: each
    // branch here runs on the whole frame, so exactly 1 job per stage. Only
    // a ROI stage holds more than one, and then at most min(crop count,
    // max_inflight) - see TestAsyncMatchesSyncUnderBackPressure.
    GRAPH_CHECK(registry.last_stage("yolov8n") != NULL);
    GRAPH_CHECK(registry.last_stage("bisenetv2") != NULL);
    GRAPH_CHECK(registry.last_stage("fastdepth_1") != NULL);
    if (registry.last_stage("yolov8n") == NULL) return;
    if (registry.last_stage("bisenetv2") == NULL) return;
    if (registry.last_stage("fastdepth_1") == NULL) return;
    GRAPH_CHECK(registry.last_stage("yolov8n")->max_pending() == 1);
    GRAPH_CHECK(registry.last_stage("bisenetv2")->max_pending() == 1);
    GRAPH_CHECK(registry.last_stage("fastdepth_1")->max_pending() == 1);
}

// The flagship graph, every branch succeeding, compared end to end: tracked
// detector -> ROI re-id, plus segmentation, pose and depth. This is the case
// the parity guarantee is actually sold on, and until fix round 1 it could
// not be written, because a live kLabelMap/kDenseMap/kKeypoints payload met
// a fail-closed comparator. Reverse ROI delivery is on so the cascade's
// crops still arrive adversarially.
void TestAsyncMatchesSyncOnHeadlineGraph() {
    GraphSpec spec = ParseGraphText(HeadlineJson(), "headline.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildHeadlineRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildHeadlineRegistry();
    async_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    // Every branch really did run and produce the shape it claims, so the
    // equality below is comparing populated payloads rather than agreeing
    // about four absent entries.
    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected.node_results.count("seg") == 1);
    GRAPH_CHECK(expected.node_results.count("pose") == 1);
    GRAPH_CHECK(expected.node_results.count("dep") == 1);
    GRAPH_CHECK(expected.roi_results["reid"].size() == 2);
    if (expected.node_results.count("seg") != 1) return;
    if (expected.node_results.count("pose") != 1) return;
    if (expected.node_results.count("dep") != 1) return;
    GRAPH_CHECK(expected.node_results["seg"].data->shape() == Shape::kLabelMap);
    GRAPH_CHECK(expected.node_results["pose"].data->shape() == Shape::kKeypoints);
    GRAPH_CHECK(expected.node_results["dep"].data->shape() == Shape::kDenseMap);
    // The two registries built their payloads independently, so these are
    // distinct objects and operator== must do a real content comparison
    // rather than short-circuiting on pointer equality.
    GRAPH_CHECK(expected.node_results["seg"].data.get() !=
                actual.node_results["seg"].data.get());

    GRAPH_CHECK(expected == actual);

    // Expected peak, fixture-relatively:
    //   (branches off "cam") - 1 + (reid's crop count) = 4 - 1 + 2 = 5.
    // One branch ("od") is harvested to release reid; seg, pose and dep are
    // still outstanding when reid's two crops are submitted. A
    // wave-synchronous executor peaks at 4 (it waits for every sibling
    // before releasing reid); a fully serialized one peaks at 2 (reid's own
    // batch). This assertion used to read 4 and was itself holding the
    // barrier in place, which is why it is spelled out rather than left as
    // a bare number.
    //
    // Order caveat: the formula assumes "cam -> od" is declared BEFORE the
    // other branches off "cam". Harvest is head-of-FIFO in submission
    // order, so declaring od last would make reid wait for seg, pose and
    // dep and drop the peak to 2 + 2 = 4 with no code change at all. If
    // this fires after a fixture edit, check the edge order before the
    // executor.
    GRAPH_CHECK(async_registry.peak_inflight() == 5);
}

// Within one node, SyncExecutor keeps the LAST crop's error message. Every
// crop used to carry the same string, so "keep the last" and "keep the
// first" were indistinguishable and that line was unverified (fix round 1).
// FakeStage now derives the message from the crop's own RoiRef - from the
// input, never from a call counter, so run() and submit() agree - and the
// two ROI crops therefore report "[roi 0]" and "[roi 1]".
void TestAsyncMatchesSyncOnPerCropFailureMessage() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    sync_registry.SetFailure("casvit_t", "simulated inference failure");
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    async_registry.SetFailure("casvit_t", "simulated inference failure");
    async_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    // The surviving message is the LAST crop's, and the crops really do
    // carry distinct messages - otherwise this assertion is vacuous.
    GRAPH_CHECK(expected.error.find("[roi 1]") != std::string::npos);
    GRAPH_CHECK(expected.error.find("[roi 0]") == std::string::npos);
    // ...and reverse delivery does not change which one survives.
    GRAPH_CHECK(expected == actual);
}

// =============================================================================
// Task 10, fix round 2.
// =============================================================================

namespace {

/// The cascade registry with the detector re-scripted to emit one box inside
/// the frame and one entirely outside it, so RouteRois clips the second away
/// and the frame carries skipped_out_of_bounds == 1.
FakeModelRegistry BuildClippingCascadeRegistry() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    ModelInfo detector = *registry.find("yolov8n");
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10, 10, 40, 60, 0.8f, "person"));
    // 640x480 frame: this one starts past the right and bottom edges, so
    // nothing survives the clip.
    boxes->items.push_back(MakeItem(700, 500, 40, 60, 0.9f, "person"));
    registry.AddModel(detector, boxes);
    return registry;
}

}  // namespace

// skipped_out_of_bounds is the only AGGREGATED field in a FrameReport, and it
// was the only one operator== compares that no executor-level fixture ever
// drove off zero: every parity graph so far kept all its boxes inside the
// frame, so async could stop accumulating the counter entirely and parity
// would still hold (both sides reporting 0). This fixture makes the two
// executors agree on a NON-zero value.
void TestAsyncMatchesSyncOnClippedRois() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildClippingCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildClippingCascadeRegistry();
    async_registry.SetDeliveryOrder("casvit_t", FakeModelRegistry::kReverse);
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    AsyncExecutor async_executor(StageJobs(4));
    FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);

    // The fixture really does clip something - otherwise the equality below
    // would be comparing 0 against 0 and proving nothing.
    GRAPH_CHECK(expected.skipped_out_of_bounds == 1);
    GRAPH_CHECK(actual.skipped_out_of_bounds == 1);
    GRAPH_CHECK(expected.roi_results["reid"].size() == 1);
    GRAPH_CHECK(expected == actual);
}

// The cascade half of dispatch-on-ready, and the half a wave barrier breaks.
// "od" and "seg" leave together; the moment "od" alone comes back, its ROI
// child must go to the hardware - it must NOT wait for "seg". So the peak is
// seg (still running) + reid's two crops = 3. An executor that waits for the
// whole wave before harvesting any of it peaks at 2, and one that fully
// serializes also peaks at 2. The FrameReport is identical in all three.
//
// ORDER-SENSITIVE FIXTURE. The peak of 3 depends on "cam" -> "od" being
// declared BEFORE "cam" -> "seg" in the JSON below. Harvest is head-of-FIFO
// in submission order, so with the two edges swapped "reid" is not released
// until "seg" has been waited on and the peak drops to 2 - a failure
// produced by editing the fixture, with no change to the executor at all.
// If this assertion fires, check the edge order here before going looking
// in graph_runner_async.cpp. That dependence is real and is documented in
// that file's header; removing it needs IStage to expose wait-for-any.
void TestAsyncOverlapsAcrossDepth() {
    FakeModelRegistry registry = BuildHeadlineRegistry();
    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"name\":\"depth\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}",
        "depth.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.roi_results["reid"].size() == 2);
    // Expected peak, fixture-relatively:
    //   (branches off "cam") - 1 + (reid's crop count) = 2 - 1 + 2 = 3,
    // given the edge order documented above.
    GRAPH_CHECK(registry.peak_inflight() == 3);
}

// DrainPending's own reason to exist. Every node RunFrame dispatches is
// already flushed as it is harvested, so the end-of-frame drain only matters
// for a stage that was NEVER dispatched and is still holding a completion -
// work abandoned by an earlier frame, which must not be delivered during the
// next one. StageGraph::Build accepts an unvalidated spec (ValidateGraph is
// what rejects an orphan), so an unreachable node is how that state is
// reachable in a test.
void TestAsyncDrainsNeverDispatchedStage() {
    GraphSpec spec;
    spec.version = 1;
    spec.name = "orphan";

    NodeSpec source;
    source.id = "cam";
    source.is_source = true;
    source.uri = "a.jpg";
    spec.nodes.push_back(source);

    NodeSpec detector;
    detector.id = "od";
    detector.model = "yolov8n";
    spec.nodes.push_back(detector);

    NodeSpec stranded;       // no edge reaches it: never dispatched
    stranded.id = "reid";
    stranded.model = "casvit_t";
    spec.nodes.push_back(stranded);

    EdgeSpec edge;
    edge.from = "cam";
    edge.to = "od";
    spec.edges.push_back(edge);

    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);  // deliberately unvalidated

    FakeStage* stranded_stage = registry.last_stage("casvit_t");
    GRAPH_CHECK(stranded_stage != NULL);
    if (stranded_stage == NULL) return;

    // Leave one completion outstanding on the stage nothing will dispatch.
    std::shared_ptr<int> delivered(new int(0));
    StageInput leftover;
    leftover.image = cv::Mat::zeros(8, 8, CV_8UC3);
    stranded_stage->submit(leftover, [delivered](const StageResult&,
                                                 const std::string&) {
        *delivered += 1;
    });
    GRAPH_CHECK(stranded_stage->pending() == 1);

    AsyncExecutor executor(StageJobs(4));
    FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);

    // Nothing from this frame may still be in flight when the next one
    // reaches the tracker, including work no node of this frame submitted.
    GRAPH_CHECK(stranded_stage->pending() == 0);
    GRAPH_CHECK(*delivered == 1);
    GRAPH_CHECK(report.error.find("never completed") != std::string::npos);
}

// Fix round 2, item 4 - a node fed BOTH a whole frame and crops of that frame.
// Every edge is individually legal (a kEither consumer accepts either kind),
// and both executors resolve it the same way: the frame wins and the crops
// are silently dropped. Parity therefore HOLDS while the user's data
// disappears, which is why a parity suite cannot be the thing that catches
// this. There is no defined answer to what such a node should run on, so
// ValidateGraph refuses it rather than inventing one.
void TestValidateRejectsMixedFrameAndRoiInput() {
    FakeModelRegistry registry = BuildCascadeRegistry();

    ModelInfo either;
    either.model_name = "either_model";
    either.task = "embedding";
    either.dxnn_file = "either_model.dxnn";
    either.output_shape = Shape::kVector;
    either.input_contract = InputContract::kEither;
    either.ready = true;
    registry.AddModel(either, StageDataPtr(new VectorData()));

    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"name\":\"mixed\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"mix\",\"model\":\"either_model\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam\",\"to\":\"mix\"},"
        "{\"from\":\"od\",\"to\":\"mix\",\"roi\":{\"classes\":[\"person\"]}}]}",
        "mixed.json");

    bool threw = false;
    try {
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        threw = true;
        GRAPH_CHECK(error.code() == GraphErrorCode::kGraphEdge);
        const std::string text = error.what();
        GRAPH_CHECK(text.find("mix") != std::string::npos);
        GRAPH_CHECK(text.find("full frame and ROI crops") != std::string::npos);
        // The remedy has to tell the user what to do instead.
        GRAPH_CHECK(text.find("split it into two nodes") != std::string::npos);
    }
    GRAPH_CHECK(threw);
}

// The same node fed only ROI crops from two parents stays legal - the
// rejection above must not over-reach into the fan-in graph Task 9 pinned.
void TestValidateAcceptsTwoRoiParents() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(FanInJson(), "fanin.json");
    bool threw = false;
    try {
        ValidateGraph(spec, registry);
    } catch (const GraphError&) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
}


// =====================================================================
// Task 11: the generated registry, the factory bridge, and the first
// tests in this file that touch real hardware.
// =====================================================================
namespace {

int g_skipped = 0;

/// Loud, counted, and never silent: a skipped case prints why and is
/// reported in the summary line, so "it passed" can never mean "it never
/// ran" without saying so.
void Skip(const char* test, const std::string& reason) {
    ++g_skipped;
    std::printf("SKIP %s: %s\n", test, reason.c_str());
}

std::string ProjectRoot() { return std::string(PROJECT_ROOT_DIR); }
std::string ModelDir() { return ProjectRoot() + "/assets/models"; }

bool FileIsReadable(const std::string& path) {
    std::FILE* file = std::fopen(path.c_str(), "rb");
    if (file == NULL) return false;
    std::fclose(file);
    return true;
}

/// "" when every named model's artifact is present, otherwise the reason
/// the hardware-dependent case cannot run.
///
/// This is the ONLY condition that legitimately skips a real-hardware case:
/// a dev checkout without models downloaded. Anything that fails AFTER this
/// returns "" - a device that will not open, a model the runtime rejects -
/// is a regression and is reported as a failure, not a skip.
std::string MissingArtifacts(const IModelRegistry& registry,
                             const std::vector<std::string>& models) {
    for (std::size_t i = 0; i < models.size(); ++i) {
        const ModelInfo* info = registry.find(models[i]);
        if (info == NULL) return "model \"" + models[i] + "\" is not registered";
        if (!info->ready) return "model \"" + models[i] + "\" is not graph-ready";
        const std::string path = ModelDir() + "/" + info->dxnn_file;
        if (!FileIsReadable(path)) {
            return path + " not present (run python3 scripts/download_models.py --models " +
                   models[i] + ")";
        }
    }
    return std::string();
}

/// Wrap a bare StageResult in a FrameReport so the ONE shared structural
/// comparator (stage_graph.hpp's operator==) decides whether two results
/// agree. A second comparator written here would be free to drift from the
/// one the parity tests use, which is exactly the defect this branch has
/// already been bitten by once.
FrameReport AsReport(const StageResult& result) {
    FrameReport report;
    report.node_results["x"] = result;
    return report;
}

/// A stage that breaks IStage's throw contract, to prove the asynchronous
/// executor survives it the way the synchronous one survives a throwing
/// run(). Nothing in the shipped tree does this; that is the point.
///
/// kThrowOnFlush and kThrowOnFirstPoll still honour clause (6) PROGRESS:
/// every accepted job is delivered from poll(), never only from flush(), so
/// the executor has something to harvest. What they break is the throw
/// contract, in flush() and in the first poll() respectively. Both deliver
/// an empty kBoxes payload, and run() returns the same payload, so a sync
/// run over the same stage is an exact oracle.
class ContractBreakingStage : public IStage {
 public:
    enum Where { kThrowOnSubmit, kThrowOnFlush, kThrowOnFirstPoll };

    explicit ContractBreakingStage(Where where)
        : where_(where), polls_(0) {}

    StageResult run(const StageInput& input) {
        if (where_ == kThrowOnSubmit) {
            throw std::runtime_error("contract-breaking stage");
        }
        return Payload(input);
    }

    void submit(const StageInput& input, StageCallback callback) {
        if (where_ == kThrowOnSubmit) {
            throw std::runtime_error("submit exploded");
        }
        pending_.push_back(std::make_pair(input, callback));
    }

    void flush() {
        if (where_ == kThrowOnFlush) {
            throw std::runtime_error("flush exploded");
        }
        DeliverAll();
    }

    void poll() {
        if (where_ == kThrowOnFirstPoll && polls_++ == 0) {
            throw std::runtime_error("poll exploded");
        }
        DeliverAll();
    }

    Shape outputShape() const { return Shape::kBoxes; }
    InputContract inputContract() const { return InputContract::kFullFrame; }

 private:
    static StageResult Payload(const StageInput& input) {
        StageResult result;
        result.data = std::make_shared<BoxesData>(Shape::kBoxes);
        result.origin = input.origin;
        return result;
    }

    void DeliverAll() {
        std::vector<std::pair<StageInput, StageCallback> > due;
        due.swap(pending_);
        for (std::size_t i = 0; i < due.size(); ++i) {
            due[i].second(Payload(due[i].first), std::string());
        }
    }

    Where where_;
    int polls_;
    std::vector<std::pair<StageInput, StageCallback> > pending_;
};

std::size_t NodeIndexById(const StageGraph& graph, const std::string& id) {
    const std::vector<NodeRuntime>& nodes = graph.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].id == id) return i;
    }
    return nodes.size();
}

std::string RealCascadeJson() {
    return "{\"version\":1,\"name\":\"real\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"sample.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov5n\"},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"},"
           "{\"id\":\"emb\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"max\":3}},"
           "{\"from\":\"od\",\"to\":\"emb\",\"roi\":{\"max\":3}}]}";
}

}  // namespace

// ---------------------------------------------------------------------
// The generated registry
// ---------------------------------------------------------------------

void TestStaticRegistryResolvesKnownModel() {
    StaticModelRegistry registry;
    const ModelInfo* info = registry.find("yolov8n");
    GRAPH_CHECK(info != NULL);
    if (info == NULL) return;
    GRAPH_CHECK(info->output_shape == Shape::kBoxes);
    GRAPH_CHECK(info->input_contract == InputContract::kFullFrame);
    GRAPH_CHECK(info->ready);
    GRAPH_CHECK(info->task == "object_detection");
    GRAPH_CHECK(!info->dxnn_file.empty());
    GRAPH_CHECK(registry.list().size() > 300);

    const ModelInfo* face = registry.find("scrfd10g");
    GRAPH_CHECK(face != NULL);
    if (face != NULL) GRAPH_CHECK(face->produces_landmarks);

    const ModelInfo* reid = registry.find("casvit_t");
    GRAPH_CHECK(reid != NULL);
    if (reid != NULL) {
        GRAPH_CHECK(reid->output_shape == Shape::kVector);
        // The task says classification; kVector comes from
        // IEmbeddingFactory, not from it.
        GRAPH_CHECK(reid->task == "image_classification");
        GRAPH_CHECK(AcceptsRoi(reid->input_contract));
    }

    GRAPH_CHECK(registry.find("no_such_model_anywhere") == NULL);
}

// One task with more than one interface behind it: 8d0b748 files
// casvit-t_224x224 (IEmbeddingFactory) under image_classification, beside
// alexnet_224x224 (IClassificationFactory). If the generator ever went back
// to deriving shape from the task string, these two models would be given
// the same shape. A face model and a PPU detector still differ in the
// landmark flag, by interface.
void TestStaticRegistryShapeComesFromFactoryNotTask() {
    StaticModelRegistry registry;
    const ModelInfo* scores = registry.find("alexnet_224x224");
    const ModelInfo* vector = registry.find("casvit-t_224x224");
    GRAPH_CHECK(scores != NULL && vector != NULL);
    if (scores != NULL && vector != NULL) {
        GRAPH_CHECK(scores->task == vector->task);
        GRAPH_CHECK(scores->output_shape == Shape::kScores);
        GRAPH_CHECK(vector->output_shape == Shape::kVector);
    }

    // SCRFD500M_PPU is example source only (not registered). scrfd500m is the
    // face model that still carries landmarks beside a PPU detector.
    const ModelInfo* ppu_face = registry.find("scrfd500m");
    const ModelInfo* ppu_det = registry.find("yolov5s_ppu");
    GRAPH_CHECK(ppu_face != NULL);
    GRAPH_CHECK(ppu_det != NULL);
    if (ppu_face == NULL || ppu_det == NULL) return;

    GRAPH_CHECK(ppu_face->produces_landmarks);
    GRAPH_CHECK(!ppu_det->produces_landmarks);
}

// Every row the generator emitted has to be usable by the engine: a ready
// model with no artifact name, no task, or a zero input size would fail
// only when someone tried to run it.
void TestStaticRegistryEveryEntryIsWellFormed() {
    StaticModelRegistry registry;
    const std::vector<ModelInfo> all = registry.list();
    GRAPH_CHECK(!all.empty());

    int malformed = 0;
    int ready = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        const ModelInfo& info = all[i];
        // A not-ready entry is legitimate (a model registered before its
        // factory lands), but it must say WHY, or --list-models has nothing
        // to tell the user.
        if (!info.ready) {
            if (info.not_ready_reason.empty()) ++malformed;
            continue;
        }
        ++ready;
        if (info.model_name.empty() || info.task.empty() ||
            info.dxnn_file.empty() || info.input_width <= 0 ||
            info.input_height <= 0) {
            ++malformed;
        }
        // list() and find() must agree: an entry returned by one and not
        // resolvable through the other would pass validation and then fail
        // at createStage.
        if (registry.find(info.model_name) == NULL) ++malformed;
    }
    GRAPH_CHECK(malformed == 0);
    GRAPH_CHECK(ready > 300);
}

void TestStaticRegistryCreateStageRejectsUnknownModel() {
    StaticModelRegistry registry;
    StageParams params;
    bool threw = false;
    std::string text;
    try {
        registry.createStage("no_such_model_anywhere", "/tmp/x.dxnn", params);
    } catch (const std::runtime_error& error) {
        threw = true;
        text = error.what();
    }
    GRAPH_CHECK(threw);
    GRAPH_CHECK(text.find("no_such_model_anywhere") != std::string::npos);
}

// A model whose artifact is absent must fail as std::runtime_error, not as
// a raw runtime exception - with the runtime's own text, unprefixed (final
// review M6). StageGraph::Build's MODEL_LOAD attributes it: it names the
// node, the model and its .dxnn, so a "model \"X\" failed to open: " prefix
// here printed the model name twice.
void TestStaticRegistryCreateStageWrapsRuntimeFailure() {
    StaticModelRegistry registry;
    StageParams params;
    bool threw = false;
    std::string text;
    try {
        registry.createStage("yolov8n", "/nonexistent/definitely-not-here.dxnn",
                             params);
    } catch (const std::runtime_error& error) {
        threw = true;
        text = error.what();
    }
    GRAPH_CHECK(threw);
    const bool unprefixed = !text.empty() &&
                            text.find("\"yolov8n\" failed to open") == std::string::npos &&
                            text.compare(0, 7, "model \"") != 0;
    GRAPH_CHECK(unprefixed);
    if (!unprefixed) std::printf("      %s\n", text.c_str());
}

// ---------------------------------------------------------------------
// The factory bridge, hardware-free parts
// ---------------------------------------------------------------------

// result_to_shape.hpp takes a vector for eight result types and a single
// value for the rest; TypedStage always holds a vector. Both arms of that
// C++14 overload-ranking dispatch are exercised here, including the empty
// vector, which is what a model that found nothing produces.
void TestStageDataFromResultsDispatchesBothOverloads() {
    std::vector<DetectionResult> detections;
    std::vector<float> box;
    box.push_back(1.f); box.push_back(2.f); box.push_back(11.f); box.push_back(22.f);
    detections.push_back(DetectionResult(box, 0.5f, 1, "cat"));
    StageDataPtr boxes = detail::StageDataFromResults(detections, 0);
    GRAPH_CHECK(boxes.get() != NULL);
    if (boxes) GRAPH_CHECK(boxes->shape() == Shape::kBoxes);

    std::vector<EmbeddingResult> embeddings;
    EmbeddingResult embedding;
    embedding.embedding.push_back(0.25f);
    embeddings.push_back(embedding);
    StageDataPtr vector_data = detail::StageDataFromResults(embeddings, 0);
    GRAPH_CHECK(vector_data.get() != NULL);
    if (vector_data) {
        GRAPH_CHECK(vector_data->shape() == Shape::kVector);
        const VectorData* values =
            dynamic_cast<const VectorData*>(vector_data.get());
        GRAPH_CHECK(values != NULL);
        if (values != NULL) GRAPH_CHECK(values->values.size() == 1);
    }

    // Empty is "the model produced nothing", not an error: the single-value
    // arm must still hand back a well-formed payload of the right shape.
    const std::vector<DepthResult> no_depth;
    StageDataPtr empty_depth = detail::StageDataFromResults(no_depth, 0);
    GRAPH_CHECK(empty_depth.get() != NULL);
    if (empty_depth) GRAPH_CHECK(empty_depth->shape() == Shape::kDenseMap);

    const std::vector<DetectionResult> no_boxes;
    StageDataPtr empty_boxes = detail::StageDataFromResults(no_boxes, 0);
    GRAPH_CHECK(empty_boxes.get() != NULL);
    if (empty_boxes) GRAPH_CHECK(empty_boxes->shape() == Shape::kBoxes);
}

// factory defaults < config.json < node params. The overlay half of that
// precedence is a graph node's numeric params reaching a factory through
// the only door a factory has, loadConfig(const ModelConfig&).
void TestNodeParamsOverlayReachesModelConfig() {
    StageParams params;
    params.numeric["score_threshold"] = 0.7;
    params.numeric["num_classes"] = 3;

    const std::string json = detail::ParamsToJson(params.numeric);
    GRAPH_CHECK(!json.empty());
    GRAPH_CHECK(json.find("score_threshold") != std::string::npos);

    ModelConfig config(json, ConfigSource::kText);
    GRAPH_CHECK(config.isLoaded());
    GRAPH_CHECK(std::fabs(config.get<float>("score_threshold", 0.f) - 0.7f) < 1e-6f);
    GRAPH_CHECK(config.get<int>("num_classes", 0) == 3);
    // A key the node did not override keeps the caller's current value -
    // that is what makes the second loadConfig an overlay and not a reload.
    GRAPH_CHECK(std::fabs(config.get<float>("nms_threshold", 0.45f) - 0.45f) < 1e-6f);

    // No params at all must not produce "{}" for ModelConfig to chew on.
    StageParams none;
    GRAPH_CHECK(detail::ParamsToJson(none.numeric).empty());
}

// ---------------------------------------------------------------------
// The asynchronous executor's error path
// ---------------------------------------------------------------------

// IStage says submit() does not throw. graph_runner_async.cpp had no
// try/catch at all while graph_runner_sync.cpp had two, so a stage that
// broke that rule terminated the process on the asynchronous path and
// became a reported error on the synchronous one.
void TestAsyncSurvivesStageThatThrowsFromSubmit() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    const std::size_t od = NodeIndexById(graph, "od");
    GRAPH_CHECK(od < graph.nodes().size());
    if (od >= graph.nodes().size()) return;
    graph.mutable_nodes()[od].stage.reset(
        new ContractBreakingStage(ContractBreakingStage::kThrowOnSubmit));

    AsyncExecutor executor(StageJobs(4));
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    const FrameReport report = executor.RunFrame(graph, frame, 0);

    GRAPH_CHECK(report.error.find("od") != std::string::npos);
    GRAPH_CHECK(report.error.find("submit exploded") != std::string::npos);
    // The consumer still ran and still landed in the report, empty, exactly
    // as SyncExecutor leaves it when its producer throws.
    GRAPH_CHECK(report.roi_results.count("reid") == 1);
}

// A stage whose flush() throws must not take the executor down. flush() is
// no longer on the data path: completions are harvested as they arrive, and
// clause (6) PROGRESS requires the stage to deliver without flush(), so this
// stage delivers from poll(). The throw only reaches FlushAll - after the
// frame, and in DrainPending - which swallows it by design, because there is
// no frame left to report it against. The frame's report is therefore the
// clean one a sync run over the same stage produces.
void TestAsyncSurvivesStageThatThrowsFromFlush() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    const std::size_t sync_od = NodeIndexById(sync_graph, "od");
    GRAPH_CHECK(sync_od < sync_graph.nodes().size());
    if (sync_od >= sync_graph.nodes().size()) return;
    sync_graph.mutable_nodes()[sync_od].stage.reset(
        new ContractBreakingStage(ContractBreakingStage::kThrowOnFlush));
    SyncExecutor sync_executor;
    const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const std::size_t od = NodeIndexById(graph, "od");
    GRAPH_CHECK(od < graph.nodes().size());
    if (od >= graph.nodes().size()) return;
    graph.mutable_nodes()[od].stage.reset(
        new ContractBreakingStage(ContractBreakingStage::kThrowOnFlush));

    AsyncOptions options = StageJobs(4);
    options.stall_timeout_ms = 2000;  // a regression fails instead of hanging
    AsyncExecutor executor(options);
    FrameReport report;
    bool run_threw = false;
    try {
        report = executor.RunFrame(graph, frame, 0);
    } catch (const std::exception& error) {
        run_threw = true;
        std::printf("      unexpected: %s\n", error.what());
    }
    GRAPH_CHECK(!run_threw);
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == report);

    bool drain_threw = false;
    try {
        executor.DrainPending(graph);
    } catch (...) {
        drain_threw = true;
    }
    GRAPH_CHECK(!drain_threw);
}

// ---------------------------------------------------------------------
// Real hardware. Every case below skips LOUDLY when the artifacts or the
// device are not here; none of them can pass vacuously.
// ---------------------------------------------------------------------

/**
 * @brief run() and submit()+flush() against ONE real stage must agree, and
 *        flush() must not return before the callback has.
 *
 * This is the case the fakes structurally cannot reach. On this hardware
 * run() and submit() share one registered engine callback (i_registry.hpp),
 * Wait() returns nothing once that callback exists, and a blocking Run()
 * fires it too - so a stage that mixed the two paths could deliver a result
 * twice or lose one. The fake has no such entanglement: it delivers inline
 * from flush(), so a flush() that returned early would still read green.
 */
void TestRealStageRunAndSubmitAgree() {
    const char* kTest = "TestRealStageRunAndSubmitAgree";
    StaticModelRegistry registry;
    std::vector<std::string> models;
    models.push_back("yolov5n");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) {
        Skip(kTest, missing);
        return;
    }
    const std::string image_path = ProjectRoot() + "/sample/img/sample_dog.jpg";
    const cv::Mat frame = cv::imread(image_path);
    // NOT a skip. sample/img/sample_dog.jpg is tracked in this repository,
    // so a failed read here means a broken checkout (missing/corrupt file,
    // wrong working directory), not an absent optional download - the same
    // reasoning MissingArtifacts()'s own comment gives for the .dxnn check
    // above, and an unattended run should say so rather than staying quiet.
    GRAPH_CHECK(!frame.empty());
    if (frame.empty()) {
        std::printf("      %s: could not read %s\n", kTest, image_path.c_str());
        return;
    }

    const ModelInfo* info = registry.find("yolov5n");
    if (info == NULL) return;
    std::unique_ptr<IStage> stage;
    StageParams params;
    std::string open_error;
    try {
        stage = registry.createStage("yolov5n", ModelDir() + "/" + info->dxnn_file,
                                     params);
    } catch (const std::exception& error) {
        open_error = error.what();
    }
    // NOT a skip. MissingArtifacts() already established that the .dxnn is
    // on disk, so failing to open it is a device or runtime REGRESSION, and
    // a regression must not share a signal with "this dev checkout has no
    // models downloaded". An unattended run that printed SKIP here would be
    // telling the truth about neither.
    GRAPH_CHECK(open_error.empty());
    if (!open_error.empty()) {
        std::printf("      %s: the .dxnn is present but the stage would not "
                    "open: %s\n", kTest, open_error.c_str());
        return;
    }

    StageInput input;
    input.image = frame;

    StageResult blocking = stage->run(input);
    GRAPH_CHECK(blocking.data.get() != NULL);
    if (blocking.data) GRAPH_CHECK(blocking.data->shape() == Shape::kBoxes);

    // A blocking run() also fires the engine's registered callback on this
    // hardware; nothing it delivered may be left outstanding.
    stage->flush();

    int fired = 0;
    StageResult submitted;
    std::string reported_error;
    stage->submit(input, [&fired, &submitted, &reported_error](
                             const StageResult& result,
                             const std::string& error) {
        ++fired;
        submitted = result;
        reported_error = error;
    });
    stage->flush();

    // The teeth: right after flush() returns, the callback must ALREADY
    // have run. A flush() that returned while the runtime's callback thread
    // was still working would leave this at 0, and AsyncExecutor would
    // harvest a node whose results had not arrived.
    GRAPH_CHECK(fired == 1);
    GRAPH_CHECK(reported_error.empty());
    GRAPH_CHECK(submitted.data.get() != NULL);

    // Same frame, same model, two paths: the shared structural comparator
    // must not be able to tell them apart.
    GRAPH_CHECK(AsReport(blocking) == AsReport(submitted));

    // Idempotent: a second flush with nothing outstanding returns at once
    // and delivers nothing new.
    stage->flush();
    GRAPH_CHECK(fired == 1);
}

/**
 * @brief The real-stage parity run: the SAME graph, built from the
 *        generated registry with real TypedStages, through both executors.
 *
 * Every other parity test in this file drives fakes. This one drives the
 * NPU, so it is the only one that can catch a divergence that exists only
 * because a real stage is asynchronous, shares one callback between run()
 * and submit(), and hands back data that dangles.
 *
 * One StageGraph, not two, deliberately: both executors then drive the same
 * TypedStage objects, which is exactly the run()-then-submit() mixture on
 * one engine that i_registry.hpp warns about.
 */
void TestRealStageSyncAsyncParity() {
    const char* kTest = "TestRealStageSyncAsyncParity";
    StaticModelRegistry registry;
    std::vector<std::string> models;
    models.push_back("yolov5n");
    models.push_back("resnet50");
    models.push_back("casvit_t");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) {
        Skip(kTest, missing);
        return;
    }
    const std::string image_path = ProjectRoot() + "/sample/img/sample_crowd.jpg";
    const cv::Mat frame = cv::imread(image_path);
    // NOT a skip - same reasoning as TestRealStageRunAndSubmitAgree above:
    // sample/img/sample_crowd.jpg is tracked in this repository, so a
    // failed read means a broken checkout, not an absent optional download.
    GRAPH_CHECK(!frame.empty());
    if (frame.empty()) {
        std::printf("      %s: could not read %s\n", kTest, image_path.c_str());
        return;
    }

    GraphSpec spec = ParseGraphText(RealCascadeJson(), "real.json");
    StageGraph graph;
    std::string build_error;
    try {
        ValidateGraph(spec, registry);
        graph.Build(spec, registry, ModelDir(), true);
    } catch (const std::exception& error) {
        build_error = error.what();
    }
    // NOT a skip, for the same reason as above: every artifact this graph
    // names was checked present, so a failure here is a device, runtime or
    // graph-validation regression, not an absent model.
    GRAPH_CHECK(build_error.empty());
    if (!build_error.empty()) {
        std::printf("      %s: every .dxnn is present but the graph would not "
                    "build: %s\n", kTest, build_error.c_str());
        return;
    }

    SyncExecutor sync_executor;
    const FrameReport first = sync_executor.RunFrame(graph, frame, 7);
    GRAPH_CHECK(first.error.empty());
    if (!first.error.empty()) {
        std::printf("      sync error: %s\n", first.error.c_str());
        return;
    }
    // The detector must actually have found something, or the comparison
    // below would be a comparison of two empty reports.
    GRAPH_CHECK(first.node_results.count("od") == 1);
    GRAPH_CHECK(first.roi_results.count("cls") == 1);
    GRAPH_CHECK(first.roi_results.count("emb") == 1);
    // GRAPH_CHECK records and CONTINUES, so nothing below may dereference
    // an iterator whose existence a check above only asserted.
    if (first.roi_results.count("cls") != 1 ||
        first.roi_results.count("emb") != 1) {
        return;
    }
    // More than one crop, so ByOrigin actually has something to order and
    // the asynchronous run below can get them back out of order.
    GRAPH_CHECK(first.roi_results.find("cls")->second.size() >= 2);
    GRAPH_CHECK(first.roi_results.find("emb")->second.size() >= 2);

    // Self-determinism first, so a failure below is attributable. If the
    // NPU itself is not reproducible frame to frame, THIS check fails and
    // the sync/async comparison is not the thing that is broken.
    const FrameReport second = sync_executor.RunFrame(graph, frame, 7);
    GRAPH_CHECK(first == second);
    if (!(first == second)) {
        std::printf("      two identical synchronous runs disagreed - the "
                    "stage is not reproducible, so the parity comparison "
                    "below cannot mean anything\n");
        return;
    }

    AsyncExecutor async_executor(StageJobs(4));
    const FrameReport actual = async_executor.RunFrame(graph, frame, 7);
    GRAPH_CHECK(actual.error.empty());
    GRAPH_CHECK(first == actual);

    // Back-pressure changes only the arrival order, never the report.
    AsyncExecutor throttled(StageJobs(1));
    const FrameReport throttled_report = throttled.RunFrame(graph, frame, 7);
    GRAPH_CHECK(first == throttled_report);
}

// Several frames in flight on real stages must reproduce, frame by frame,
// what the synchronous oracle produces. Two graphs, so each executor owns
// its own trackers and engines.
void TestRealStagePipelinedMatchesSyncAcrossFrames() {
    const char* kTest = "TestRealStagePipelinedMatchesSyncAcrossFrames";
    StaticModelRegistry registry;
    std::vector<std::string> models;
    models.push_back("yolov5n");
    models.push_back("resnet50");
    models.push_back("casvit_t");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) {
        Skip(kTest, missing);
        return;
    }
    const cv::Mat image = cv::imread(ProjectRoot() + "/sample/img/sample_crowd.jpg");
    GRAPH_CHECK(!image.empty());
    if (image.empty()) return;
    std::vector<cv::Mat> frames;
    for (int k = 0; k < 6; ++k) {
        const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 8 * k, 0, 1, 0);
        cv::Mat moved;
        cv::warpAffine(image, moved, shift, image.size());
        frames.push_back(moved);
    }

    GraphSpec spec = ParseGraphText(RealCascadeJson(), "real.json");
    StageGraph sync_graph;
    StageGraph async_graph;
    std::string build_error;
    try {
        ValidateGraph(spec, registry);
        sync_graph.Build(spec, registry, ModelDir(), true);
        async_graph.Build(spec, registry, ModelDir(), true);
    } catch (const std::exception& error) {
        build_error = error.what();
    }
    GRAPH_CHECK(build_error.empty());
    if (!build_error.empty()) return;

    SyncExecutor sync_executor;
    std::vector<FrameReport> expected;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        expected.push_back(sync_executor.RunFrame(sync_graph, frames[i], i));
    }

    AsyncOptions options;
    options.max_frames_in_flight = 4;
    AsyncExecutor executor(options);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        executor.Submit(async_graph, frames[i], i);
    }
    executor.Finish(async_graph);
    GRAPH_CHECK(executor.peak_frames_in_flight() > 1);
    for (std::size_t i = 0; i < frames.size(); ++i) {
        FrameReport actual;
        GRAPH_CHECK(executor.TryNext(&actual));
        GRAPH_CHECK(actual.error.empty());
        GRAPH_CHECK(expected[i] == actual);
    }
}


// stage_graph.cpp's SamePayload is fail-closed: a shape it does not
// implement is never equal. Task 11's real parity fixture is the first to
// produce kScores, so kScores moved out of that group - and a comparator
// that always returned true would now make the parity test vacuous for
// classification. These two cases pin both directions.
void TestOperatorEqualityComparesScores() {
    std::shared_ptr<ScoresData> left(new ScoresData());
    ClassificationResult top;
    top.class_id = 3;
    top.class_name = "tabby";
    top.confidence = 0.75f;
    top.top_k.push_back(std::make_pair(3, 0.75f));
    top.top_k.push_back(std::make_pair(4, 0.20f));
    left->items.push_back(top);

    std::shared_ptr<ScoresData> same(new ScoresData());
    same->items = left->items;

    FrameReport a;
    a.node_results["cls"].data = left;
    FrameReport b;
    b.node_results["cls"].data = same;
    GRAPH_CHECK(a == b);

    // Confidence divergence.
    std::shared_ptr<ScoresData> shifted(new ScoresData());
    shifted->items = left->items;
    shifted->items[0].confidence = 0.7500001f;
    FrameReport c;
    c.node_results["cls"].data = shifted;
    GRAPH_CHECK(a != c);

    // top_k divergence with an identical winning class: a comparator that
    // only looked at class_id/confidence would call these equal.
    std::shared_ptr<ScoresData> reordered(new ScoresData());
    reordered->items = left->items;
    reordered->items[0].top_k[1].first = 9;
    FrameReport d;
    d.node_results["cls"].data = reordered;
    GRAPH_CHECK(a != d);

    // class_name divergence.
    std::shared_ptr<ScoresData> renamed(new ScoresData());
    renamed->items = left->items;
    renamed->items[0].class_name = "siamese";
    FrameReport e;
    e.node_results["cls"].data = renamed;
    GRAPH_CHECK(a != e);
}

// =============================================================================
// Task 12 — composite visualization.
//
// RestoreBox/RestorePoint (roi_router.hpp) are translation-only; they never
// applied ref.inv_align. A crop that went through face5 alignment or OBB
// un-rotation is not a translation of the source, so a translation-only
// restore of a point inside that crop lands in the wrong place — silently,
// since a plain axis-aligned crop (every other fixture in this file) has
// inv_align == identity and cannot tell the difference. The fixture below
// is deliberately NOT a plain crop: a 2x scale plus a translation, so a
// translation-only restore is provably wrong for it (see the comment
// inline) and only a restore that actually composes inv_align passes.
// =============================================================================

void TestRestorePointWarpedUndoesAlignment() {
    // A crop produced by a 2x scale + (10, 20) translation.
    RoiRef ref;
    ref.from_roi = true;
    ref.src_box = cv::Rect2f(0.f, 0.f, 224.f, 224.f);
    const cv::Matx23f forward(2.f, 0.f, 10.f, 0.f, 2.f, 20.f);
    cv::Mat as_mat(2, 3, CV_32F);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) as_mat.at<float>(r, c) = forward(r, c);
    }
    cv::Mat inverted;
    cv::invertAffineTransform(as_mat, inverted);
    cv::Mat as_float;
    inverted.convertTo(as_float, CV_32F);
    ref.inv_align = cv::Matx23f(
        as_float.at<float>(0, 0), as_float.at<float>(0, 1), as_float.at<float>(0, 2),
        as_float.at<float>(1, 0), as_float.at<float>(1, 1), as_float.at<float>(1, 2));

    // Source point (30, 40) maps forward to (70, 100); restoring must
    // return it. A translation-only restore would instead return
    // (70 + 0, 100 + 0) = (70, 100) unchanged (src_box is (0,0)-origin
    // here), which is NOT (30, 40) — proof this fixture actually
    // exercises inv_align rather than degenerating to a translation.
    const cv::Point2f restored = RestorePointWarped(cv::Point2f(70.f, 100.f), ref);
    GRAPH_CHECK(std::abs(restored.x - 30.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.y - 40.f) < 0.01f);
}

// A plain (non-warped) ROI must still reduce to translation: inv_align
// defaults to identity, so RestorePointWarped must agree exactly with
// RestorePoint/RestoreBox's own translation-only contract.
// Fix round 3: inv_align alone maps crop-local to source now (RouteRois
// folds the crop's own translation into it at construction), so a
// hand-built plain-crop RoiRef must set inv_align to that translation
// itself, exactly like TestRestoreBoxMapsRoiLocalToSource.
void TestRestorePointWarpedIsTranslationForPlainCrop() {
    RoiRef ref;
    ref.from_roi = true;
    ref.src_box = cv::Rect2f(15.f, 25.f, 100.f, 80.f);
    ref.inv_align = cv::Matx23f(1.f, 0.f, 15.f, 0.f, 1.f, 25.f);

    const cv::Point2f restored = RestorePointWarped(cv::Point2f(5.f, 7.f), ref);
    GRAPH_CHECK(std::abs(restored.x - 20.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.y - 32.f) < 0.01f);
}

void TestRestorePointWarpedIsIdentityForFullFrame() {
    RoiRef ref;  // from_roi defaults to false.
    const cv::Point2f restored = RestorePointWarped(cv::Point2f(12.f, 34.f), ref);
    GRAPH_CHECK(restored.x == 12.f);
    GRAPH_CHECK(restored.y == 34.f);
}

void TestRenderReportIsDeterministic() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    const cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport report = executor.RunFrame(graph, source, 0);

    const cv::Mat first = RenderReport(source, report);
    const cv::Mat second = RenderReport(source, report);
    GRAPH_CHECK(!first.empty());
    GRAPH_CHECK(first.size() == source.size());
    GRAPH_CHECK(cv::countNonZero(first.reshape(1) != second.reshape(1)) == 0);
    // Two person boxes were drawn, so the canvas is no longer black.
    GRAPH_CHECK(cv::countNonZero(first.reshape(1)) > 0);
}

void TestRenderReportHandlesEmptyReport() {
    FrameReport empty;
    const cv::Mat source = cv::Mat::zeros(120, 160, CV_8UC3);
    const cv::Mat rendered = RenderReport(source, empty);
    GRAPH_CHECK(!rendered.empty());
    GRAPH_CHECK(rendered.size() == source.size());
}

// The empty-canvas edge case RenderReport's own doc comment promises:
// nothing to restore, nothing to draw, nothing to crash on.
void TestRenderReportHandlesEmptySource() {
    FrameReport empty;
    const cv::Mat rendered = RenderReport(cv::Mat(), empty);
    GRAPH_CHECK(rendered.empty());
}

// =============================================================================
// Task 12, fix round 1 — RestoreBox composes inv_align; DrawInstances is
// unit-tested directly (it needs nothing from the executor pipeline, same
// as Task 9's SamePayload pattern).
// =============================================================================

// RestoreBox must compose ref.inv_align, not just translate by src_box.
// Same 2x-scale fixture as TestRestorePointWarpedUndoesAlignment: source
// rect [30,40]-[50,60] (20x20) maps forward to [70,100]-[110,140]
// (40x40) under a 2x scale + (10,20) translation, so restoring the
// forward-mapped rect must recover the ORIGINAL rect, not merely shift it
// by src_box (which is (0,0) here, so a translation-only restore would
// wrongly return the input unchanged).
void TestRestoreBoxComposesInvAlign() {
    RoiRef ref;
    ref.from_roi = true;
    ref.src_box = cv::Rect2f(0.f, 0.f, 224.f, 224.f);
    const cv::Matx23f forward(2.f, 0.f, 10.f, 0.f, 2.f, 20.f);
    cv::Mat as_mat(2, 3, CV_32F);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) as_mat.at<float>(r, c) = forward(r, c);
    }
    cv::Mat inverted;
    cv::invertAffineTransform(as_mat, inverted);
    cv::Mat as_float;
    inverted.convertTo(as_float, CV_32F);
    ref.inv_align = cv::Matx23f(
        as_float.at<float>(0, 0), as_float.at<float>(0, 1), as_float.at<float>(0, 2),
        as_float.at<float>(1, 0), as_float.at<float>(1, 1), as_float.at<float>(1, 2));

    const cv::Rect2f restored = RestoreBox(cv::Rect2f(70.f, 100.f, 40.f, 40.f), ref);
    GRAPH_CHECK(std::abs(restored.x - 30.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.y - 40.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.width - 20.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.height - 20.f) < 0.01f);
}

// A rotated crop's box restores to the axis-aligned bounding box of its
// four transformed corners, not a (wrong) same-size translated rect: a
// 90-degree rotation swaps width/height, which only the corners-based
// restore captures.
void TestRestoreBoxUnderRotationIsAxisAlignedBoundingBox() {
    RoiRef ref;
    ref.from_roi = true;
    ref.src_box = cv::Rect2f(0.f, 0.f, 0.f, 0.f);
    // 90-degree rotation about the origin: (x, y) -> (-y, x).
    ref.inv_align = cv::Matx23f(0.f, -1.f, 0.f,
                                1.f, 0.f, 0.f);

    // A 10-wide, 4-tall box at the origin: corners (0,0),(10,0),(0,4),(10,4)
    // rotate to (0,0),(0,10),(-4,0),(-4,10) - AABB is x in [-4,0], y in
    // [0,10], i.e. a 4-wide, 10-tall box: width/height swapped versus the
    // input, which only a corners-based restore reproduces.
    const cv::Rect2f restored = RestoreBox(cv::Rect2f(0.f, 0.f, 10.f, 4.f), ref);
    GRAPH_CHECK(std::abs(restored.x - (-4.f)) < 0.01f);
    GRAPH_CHECK(std::abs(restored.y - 0.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.width - 4.f) < 0.01f);
    GRAPH_CHECK(std::abs(restored.height - 10.f) < 0.01f);
}

// DrawInstances (kInstances, Pass 2) blends a per-instance mask into the
// canvas only where the mask is non-zero, confined to its own target
// region - this fixture's mask already matches that region's pixel size,
// so no resize runs; the sibling test below covers the resize branch.
// Built directly, like Task 9's SamePayload tests: nothing here needs the
// executor, a graph, or a registry.
void TestRenderReportBlendsInstanceMaskAtItsOwnRegion() {
    std::shared_ptr<BoxesData> instances(new BoxesData(Shape::kInstances));
    BoxItem item;
    item.box = cv::Rect2f(5.f, 5.f, 10.f, 10.f);
    item.mask = cv::Mat::zeros(30, 40, CV_8UC1);  // rows=30, cols=40: matches source
    item.mask(cv::Rect(5, 5, 10, 10)).setTo(cv::Scalar(255));
    instances->items.push_back(item);

    FrameReport report;
    report.node_results["inst"].data = instances;  // from_roi defaults to false

    const cv::Mat source = cv::Mat::zeros(30, 40, CV_8UC3);
    const cv::Mat rendered = RenderReport(source, report);

    GRAPH_CHECK(rendered.size() == source.size());
    if (rendered.size() != source.size()) return;

    // Inside the mask: blended against the class color, so no longer black.
    // Task 5 (C2): the instance's box (5,5,10,10) and its label are drawn
    // now too, and the label's filled background (drawn below the box's
    // top edge, since the box is near the canvas top) covers the old (10,10)
    // probe. The probe moved to (8,8) - inside the mask, off the outline
    // and the label - and checks the mask's exact blend.
    const cv::Vec3b c = SEGMENTATION_COLORS[0];
    cv::Mat blend;
    cv::addWeighted(cv::Mat::zeros(1, 1, CV_8UC3), 0.55,
                    cv::Mat(1, 1, CV_8UC3, cv::Scalar(c[0], c[1], c[2])), 0.45, 0, blend);
    const cv::Vec3b inside = rendered.at<cv::Vec3b>(8, 8);
    GRAPH_CHECK(inside == blend.at<cv::Vec3b>(0, 0));
    // Outside the mask (but inside the full-canvas target region): the
    // mask is 0 there, so copyTo must leave the source pixel untouched.
    const cv::Vec3b outside = rendered.at<cv::Vec3b>(1, 1);
    GRAPH_CHECK(outside[0] == 0 && outside[1] == 0 && outside[2] == 0);
}

// The resize branch: the mask's pixel size does not match its stage's
// target region (here, an ROI stage's own src_box), so DrawInstances must
// resize the mask before blending - and confine the blend to src_box, not
// leak across the whole canvas.
//
// Task 5 (C2) fixed this fixture's RoiRef: it used to leave inv_align at
// the identity beside src_box (10,10) - a ref RouteRois never produces -
// which put the instance's (now drawn) outline at the canvas origin, on
// the "outside" probe. It is now what RouteRois builds for a plain crop
// (inv_align = translate(src_box), crop_size = src_box's size), moved
// down so the label is drawn above the crop, and the item's box is small,
// so neither the outline nor the label covers the far-corner probe.
void TestRenderReportResizesInstanceMaskToRoiRegion() {
    std::shared_ptr<BoxesData> instances(new BoxesData(Shape::kInstances));
    BoxItem item;
    item.box = cv::Rect2f(0.f, 0.f, 4.f, 4.f);
    item.mask = cv::Mat(5, 5, CV_8UC1, cv::Scalar(255));  // fully "on", undersized

    instances->items.push_back(item);

    FrameReport report;
    StageResult result;
    result.data = instances;
    result.origin.from_roi = true;
    result.origin.src_box = cv::Rect2f(10.f, 30.f, 20.f, 20.f);
    result.origin.inv_align = cv::Matx23f(1.f, 0.f, 10.f, 0.f, 1.f, 30.f);
    result.origin.crop_size = cv::Size(20, 20);
    report.roi_results["crop_seg"].push_back(result);

    const cv::Mat source = cv::Mat::zeros(60, 60, CV_8UC3);
    const cv::Mat rendered = RenderReport(source, report);

    GRAPH_CHECK(rendered.size() == source.size());
    if (rendered.size() != source.size()) return;

    // The resized (fully "on") mask must cover the WHOLE 20x20 src_box
    // region, including its far corner - proof the resize ran, not just
    // that copyTo didn't crash on a mismatched size. Exactly the mask's
    // blend (instance 0's colour at 0.45 over black), so an outline or a
    // label there could not pass for it.
    const cv::Vec3b c = SEGMENTATION_COLORS[0];
    cv::Mat blend;
    cv::addWeighted(cv::Mat::zeros(1, 1, CV_8UC3), 0.55,
                    cv::Mat(1, 1, CV_8UC3, cv::Scalar(c[0], c[1], c[2])), 0.45, 0, blend);
    GRAPH_CHECK(rendered.at<cv::Vec3b>(49, 29) == blend.at<cv::Vec3b>(0, 0));
    // Outside src_box entirely: must stay untouched.
    const cv::Vec3b outside = rendered.at<cv::Vec3b>(0, 0);
    GRAPH_CHECK(outside[0] == 0 && outside[1] == 0 && outside[2] == 0);
}

// Task 12, fix round 2's RestoreAngleWarped (and its three tests that were
// here) was removed in fix round 4: it had zero non-test callers once
// round 3 switched DrawObBoxes to drawing real corners instead of
// reconstructing a cx/cy/w/h/angle tuple, and no other consumer in this
// codebase - including Task 13's --report JSON serializer, checked
// directly - restores an OBB's angle to a scalar. See the Fix round 4
// report section.

// =============================================================================
// Task 12, fix round 3 — the composition-order defect (Spec review): the
// translation folded into ref.inv_align has to go BEFORE the rotation for
// an OBB crop, and not at all for a face5 crop, but every hand-written-
// matrix fixture in this file (including the ones above) put ref.src_box
// at the origin, where every composition order agrees by coincidence. The
// two tests below are round-trips built from RouteRois's own real output —
// a marker painted into the source frame, located in the actual warped
// crop image RouteRois produces, then restored via the real ref RouteRois
// attached to that crop — so there is no hand-derived matrix anywhere for
// the composition order to silently disagree with.
// =============================================================================

namespace {

// Locate a painted marker's centroid in a crop, in crop-local pixel
// coordinates. Centroid (not brightest-pixel) is robust to the blur a
// rotation's linear interpolation puts on a filled circle's edge.
cv::Point2f FindMarkerCentroid(const cv::Mat& crop) {
    cv::Mat gray;
    cv::cvtColor(crop, gray, cv::COLOR_BGR2GRAY);
    cv::Mat mask;
    cv::threshold(gray, mask, 100, 255, cv::THRESH_BINARY);
    const cv::Moments mo = cv::moments(mask, true);
    if (mo.m00 <= 0.0) return cv::Point2f(-1.f, -1.f);
    return cv::Point2f(static_cast<float>(mo.m10 / mo.m00),
                       static_cast<float>(mo.m01 / mo.m00));
}

}  // namespace

// The reviewer's own numeric example: src_box=(300,200,80,60), theta=0.5
// rad. A translation-only restore (round 1/2's bug) returns the input
// point essentially unchanged; the correct order composes the un-rotation
// AFTER translating into the rotated frame, per RouteRois's own comment.
void TestRestoreBoxCornerRoundTripsThroughRealObbCrop() {
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    BoxItem obb = MakeItem(300.f, 200.f, 80.f, 60.f, 0.9f, "obj");
    obb.angle = 0.5f;  // radians
    boxes.items.push_back(obb);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    // A point clearly off the box's own centre (340,230), so a rotation
    // that merely fixed the centre couldn't spuriously pass this test.
    const cv::Point2f p_src(335.f, 210.f);
    cv::circle(source, cv::Point(static_cast<int>(p_src.x), static_cast<int>(p_src.y)),
              4, cv::Scalar(255, 255, 255), cv::FILLED);

    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;

    RouteStats stats;
    std::vector<RoiCrop> crops = RouteRois(boxes, source, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;

    const cv::Point2f found_local = FindMarkerCentroid(crops[0].image);
    GRAPH_CHECK(found_local.x >= 0.f);
    if (found_local.x < 0.f) return;

    const cv::Rect2f restored =
        RestoreBox(cv::Rect2f(found_local.x, found_local.y, 0.f, 0.f), crops[0].ref);
    GRAPH_CHECK(std::abs(restored.x - p_src.x) < 2.f);
    GRAPH_CHECK(std::abs(restored.y - p_src.y) < 2.f);
}

// Same round-trip for face5: BuildFace5Transform's forward warp has no
// src_box relationship at all (round 1/2's bug added ref.src_box.x/y to
// every restore regardless), so a wrong-order restore here is off by
// roughly (src_box.x, src_box.y) - hundreds of pixels for this fixture.
void TestRestorePointWarpedRoundTripsThroughRealFace5Crop() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    BoxItem face = MakeItem(300.f, 200.f, 80.f, 100.f, 0.9f, "face");
    // A plausible 5-point face layout (both eyes, nose, both mouth
    // corners), scaled/placed within the detection box - not required to
    // match roi_router.cpp's internal reference layout exactly, only to be
    // non-degenerate enough for cv::estimateAffinePartial2D to fit well.
    face.landmarks.push_back(Keypoint(320.f, 225.f));  // left eye
    face.landmarks.push_back(Keypoint(360.f, 225.f));  // right eye
    face.landmarks.push_back(Keypoint(340.f, 250.f));  // nose
    face.landmarks.push_back(Keypoint(322.f, 275.f));  // left mouth
    face.landmarks.push_back(Keypoint(358.f, 275.f));  // right mouth
    boxes.items.push_back(face);

    cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    // The nose landmark itself: guaranteed to be well inside the fitted
    // crop's own canvas, since it is one of the 5 fit correspondences.
    const cv::Point2f p_src(340.f, 250.f);
    cv::circle(source, cv::Point(static_cast<int>(p_src.x), static_cast<int>(p_src.y)),
              4, cv::Scalar(255, 255, 255), cv::FILLED);

    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;
    spec.align = "face5";

    RouteStats stats;
    std::vector<RoiCrop> crops = RouteRois(boxes, source, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;

    const cv::Point2f found_local = FindMarkerCentroid(crops[0].image);
    GRAPH_CHECK(found_local.x >= 0.f);
    if (found_local.x < 0.f) return;

    const cv::Point2f restored = RestorePointWarped(found_local, crops[0].ref);
    // A generous tolerance for estimateAffinePartial2D's own least-squares
    // fit residual (this face layout is plausible, not an exact scaled
    // copy of the internal reference) - still two orders of magnitude
    // tighter than the ~hundreds-of-pixels error the src_box-added bug
    // produces for this fixture.
    GRAPH_CHECK(std::abs(restored.x - p_src.x) < 10.f);
    GRAPH_CHECK(std::abs(restored.y - p_src.y) < 10.f);
}

// Brief defect coverage gap (fix round 3, item 4): TestRenderReportIs-
// Deterministic never actually distinguished "roi_results was drawn" from
// "roi_results was silently skipped", since BuildCascadeRegistry's
// node_results["od"] kBoxes alone already makes the canvas non-black.
// Clearing roi_results from an otherwise-identical, executor-produced
// report must strictly reduce the drawn pixel count, proving RenderReport
// draws content roi_results uniquely carries (here: "reid"'s kVector
// text) - the same property ForEachResult walking both containers exists
// to guarantee.
void TestRenderReportDrawsRoiResultsNotJustNodeResults() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    const cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport with_roi = executor.RunFrame(graph, source, 0);
    GRAPH_CHECK(!with_roi.roi_results.empty());
    if (with_roi.roi_results.empty()) return;

    FrameReport without_roi = with_roi;
    without_roi.roi_results.clear();

    const cv::Mat rendered_with = RenderReport(source, with_roi);
    const cv::Mat rendered_without = RenderReport(source, without_roi);

    // NOT a pixel-count comparison: DrawVector's own text paints a black
    // background rectangle before its white glyphs, and that rectangle can
    // overwrite pixels an earlier pass (DetectionVisualizer's own box
    // label, drawn near the same anchor) already made non-zero - so
    // "roi_results was drawn" can net *fewer* non-zero pixels than
    // "roi_results was skipped", not more. The property that actually
    // distinguishes "processed" from "skipped" is that the two renders are
    // not byte-identical: if ForEachResult stopped walking roi_results,
    // both calls would ignore it equally and render identically.
    const int changed_pixels =
        cv::countNonZero(rendered_with.reshape(1) != rendered_without.reshape(1));
    GRAPH_CHECK(changed_pixels > 0);
}

// --- Task 13 round 2: unknown keys are an error, not a shrug ----------
//
// A key the parser did not read used to be dropped in silence. In a
// product configured by editing JSON that is worse than an error: the
// graph runs and quietly does something other than what was asked, and
// nothing prompts the user to look. --check could not catch it either -
// its table prints nodes and models, not "roi" options.
//
// Each of these was confirmed to discriminate by removing the
// corresponding RejectUnknownKeys call and watching it fail; see
// task-13-report.md, "Fix round 2 of 5".

void TestParseRejectsUnknownRoiKey() {
    std::string message;
    // The reviewer's own probe.
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"o\",\"model\":\"m\"},{\"id\":\"r\",\"model\":\"n\"}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"o\"},"
        "{\"from\":\"o\",\"to\":\"r\",\"roi\":{\"paddng\":0.5}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown key") != std::string::npos);
    GRAPH_CHECK(message.find("paddng") != std::string::npos);
    // Names WHERE, not just what: the edge and its two endpoints.
    GRAPH_CHECK(message.find("\"o\"->\"r\"") != std::string::npos);
    GRAPH_CHECK(message.find("roi") != std::string::npos);
    // "paddng" is three plain edits from "pad" - close enough only
    // because Closeness() also treats a shared prefix as a near match.
    GRAPH_CHECK(message.find("did you mean \"pad\"") != std::string::npos);
    // And the accepted set is enumerated whether or not a guess was made.
    GRAPH_CHECK(message.find("min_score") != std::string::npos);
    GRAPH_CHECK(message.find("align") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"o\",\"model\":\"m\"},{\"id\":\"r\",\"model\":\"n\"}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"o\"},"
        "{\"from\":\"o\",\"to\":\"r\",\"roi\":{\"classez\":[\"person\"]}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"classes\"") != std::string::npos);
}

void TestParseRejectsUnknownNodeKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"modle\":\"m\"}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown key") != std::string::npos);
    GRAPH_CHECK(message.find("modle") != std::string::npos);
    // The node's own id, so the reader does not count array entries.
    GRAPH_CHECK(message.find("\"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("did you mean \"model\"") != std::string::npos);
}

void TestParseRejectsUnknownTrackKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\",\"track\":{\"maxage\":30}}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("maxage") != std::string::npos);
    GRAPH_CHECK(message.find("track") != std::string::npos);
    GRAPH_CHECK(message.find("did you mean \"max_age\"") != std::string::npos);
}

// Pins the Damerau half of EditDistance. Under plain Levenshtein "rio"
// is two edits from BOTH "roi" and "to", and declaration order then
// advises "to" - a suggestion that would send the reader the wrong way.
void TestParseSuggestsTheTransposedKeyNotTheShorterOne() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"od\",\"rio\":{}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"roi\"") != std::string::npos);
    GRAPH_CHECK(message.find("did you mean \"to\"") == std::string::npos);
}

void TestParseRejectsUnknownTopLevelKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodez\":[],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("nodez") != std::string::npos);
    GRAPH_CHECK(message.find("did you mean \"nodes\"") != std::string::npos);
}

// No guess is better than a wrong guess: a key nothing resembles gets the
// accepted list and no "did you mean".
void TestParseOffersNoSuggestionForAnUnrelatedKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\",\"zzzzzzzz\":1}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean") == std::string::npos);
    GRAPH_CHECK(message.find("accepted keys:") != std::string::npos);
}

// The version gate must win. A graph written for a later release carries
// keys this build has never heard of BY DEFINITION, and answering it with
// "unknown key" instead of "version 2 is not supported" would send the
// reader to fix the wrong thing.
void TestUnsupportedVersionOutranksUnknownKeys() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":2,\"nodez\":[],\"whatever\":1}",
        GraphErrorCode::kGraphVersion, &message));
    GRAPH_CHECK(message.find("unknown key") == std::string::npos);
}

// Likewise a reserved key keeps its own, more informative code.
void TestReservedKeyOutranksUnknownKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"c\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\",\"prompt\":\"a person\"}],"
        "\"edges\":[{\"from\":\"c\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphReserved, &message));
    GRAPH_CHECK(message.find("reserved") != std::string::npos);
}

// The other half of the rule, and the more dangerous one to get wrong: a
// rejection that also rejected valid graphs would be a worse bug than the
// silence it replaced. This fixture uses EVERY key the schema defines, at
// every level, plus an arbitrary "params" entry - params keys are model
// parameter names and are open by design.
void TestParseAcceptsEverySchemaKeyAndLeavesParamsOpen() {
    const std::string json =
        "{\"$schema\":\"https://example/graph.schema.json\","
        "\"version\":1,\"name\":\"all-keys\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\","
        "\"params\":{\"score_threshold\":0.35,\"anything_at_all\":2},"
        "\"track\":{\"algo\":\"iou\",\"iou\":0.3,\"max_age\":30}},"
        "{\"id\":\"cls\",\"model\":\"n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"],"
        "\"min_score\":0.4,\"min_area\":100,\"pad\":0.05,\"max\":16,"
        "\"align\":\"face5\"}}]}";
    GraphSpec spec;
    std::string thrown;
    try {
        spec = ParseGraphText(json, "all_keys.json");
    } catch (const std::exception& error) {
        thrown = error.what();
    }
    GRAPH_CHECK(thrown.empty());
    if (!thrown.empty()) {
        std::printf("      TestParseAcceptsEverySchemaKey...: %s\n",
                    thrown.c_str());
        return;
    }
    GRAPH_CHECK(spec.nodes.size() == 3);
    GRAPH_CHECK(spec.edges.size() == 2);
    GRAPH_CHECK(spec.nodes[1].params.numeric["anything_at_all"] == 2.0);
    GRAPH_CHECK(spec.nodes[1].track.max_age == 30);
    GRAPH_CHECK(spec.edges[1].roi.max == 16);
    GRAPH_CHECK(spec.edges[1].roi.align == "face5");
}

// The five graphs this example ships must parse unchanged. A sample the
// new rule rejects would break the product's headline promise, not just a
// test.
void TestEveryShippedSampleGraphStillParses() {
    const char* kSamples[] = {"fanout_od_seg_pose.json",
                              "fanout_od_seg_depth.json",
                              "cascade_od_reid_track.json",
                              "cascade_od_attr.json",
                              "cascade_obb_cls.json"};
    const std::size_t count = sizeof(kSamples) / sizeof(kSamples[0]);
    for (std::size_t i = 0; i < count; ++i) {
        const std::string path = ProjectRoot() +
            "/src/cpp_example/multi_model_graph/" + kSamples[i];
        std::string thrown;
        try {
            const GraphSpec spec = ParseGraphFile(path);
            GRAPH_CHECK(!spec.nodes.empty());
        } catch (const std::exception& error) {
            thrown = error.what();
        }
        GRAPH_CHECK(thrown.empty());
        if (!thrown.empty()) {
            std::printf("      TestEveryShippedSampleGraphStillParses: %s: "
                        "%s\n", kSamples[i], thrown.c_str());
        }
    }
}

namespace {

// True if any pixel within a small window around (cx, cy) is non-black.
bool HasNonZeroNear(const cv::Mat& canvas, int cx, int cy, int radius) {
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            const int x = cx + dx;
            const int y = cy + dy;
            if (x < 0 || y < 0 || x >= canvas.cols || y >= canvas.rows) continue;
            const cv::Vec3b& px = canvas.at<cv::Vec3b>(y, x);
            if (px[0] != 0 || px[1] != 0 || px[2] != 0) return true;
        }
    }
    return false;
}

}  // namespace

// Fix round 4 — the regression fix round 3 itself introduced: DrawObBoxes
// called RestoreBoxCorners(item.box, ...) directly, but item.box
// (shape.hpp) is the OBB's UN-rotated axis-aligned extent in crop-local
// space, and item.angle never appeared anywhere in that function - the
// MOST COMMON OBB case (a detector on a plain crop, inv_align a pure
// translation) drew an axis-aligned rectangle instead of the real rotated
// one. This is the fixture that would have caught it: a kObBoxes result
// with a non-zero item.angle, driven through the real RenderReport, with
// no fixture anywhere else in this suite doing that at all.
void TestRenderReportRotatesObBoxByItemAngle() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kObBoxes));
    BoxItem item;
    // A thin, long, un-rotated HORIZONTAL extent - centre (200,200),
    // spanning x:[125,275], y:[190,210].
    item.box = cv::Rect2f(125.f, 190.f, 150.f, 20.f);
    item.angle = static_cast<float>(CV_PI) / 2.f;  // exactly 90 degrees.
    item.score = 0.9f;
    item.class_id = 0;
    item.class_name = "obj";
    boxes->items.push_back(item);

    FrameReport report;
    report.node_results["obb"].data = boxes;  // from_roi defaults to false.

    const cv::Mat source = cv::Mat::zeros(400, 400, CV_8UC3);
    const cv::Mat rendered = RenderReport(source, report);

    GRAPH_CHECK(rendered.size() == source.size());
    if (rendered.size() != source.size()) return;

    // A 90-degree rotation about the centre turns the horizontal strip
    // VERTICAL: same convention as OBBResult::getCorners() (i_processor.
    // hpp), R(angle) = [[cos,-sin],[sin,cos]] applied to the offset from
    // centre - at exactly 90 degrees, (dx,dy) -> (-dy,dx), so the
    // un-rotated top-left corner (offset (-75,-10)) rotates to offset
    // (10,-75), landing at (210,125). The rotated box now spans
    // x:[190,210], y:[125,275] - a vertical strip nowhere near the
    // original horizontal one except at the shared centre.
    GRAPH_CHECK(HasNonZeroNear(rendered, 210, 125, 2));

    // Fix round 5 — a corner check alone cannot pin the winding order: a
    // bowtie (round 3's own unrelated bug, fixed alongside this one) has
    // the SAME four vertices as the correctly-wound quad, differing only
    // in which edges connect them, and every corner this test samples
    // sits inside x in [190,210], nowhere near where the two windings'
    // edges diverge - the identical blind spot that let round 3 ship the
    // bowtie in the first place. Pin it with an EDGE midpoint instead: the
    // rotated quad's true left edge runs from (190,125) to (190,275)
    // (BL'-BR' in DrawObBoxes's own naming), midpoint (190,200). Under the
    // correct winding (TL,BL,BR,TR) that edge is drawn; under the bowtie
    // (TL,BL,TR,BR - the swap removed) the edge at x=190 spanning
    // y:125-275 does not exist at all - both diagonals that replace it
    // cross near y=200 at x~=200, 10px away, outside this window.
    GRAPH_CHECK(HasNonZeroNear(rendered, 190, 200, 2));

    // The un-rotated box's own TRUE left edge, x=125 (the previous
    // (130,200) probe missed this by ~5px, outside its own radius-2
    // window, and so never actually fired under any mutation - fixed
    // here to the real edge position). Deep inside the ORIGINAL
    // horizontal strip's boundary, but the rotated (vertical) strip's
    // x-range is [190,210], 65px away in x. If item.angle were ignored
    // (the round-3 regression), the axis-aligned horizontal rectangle
    // would still be drawn and its left edge would pass directly
    // through here.
    GRAPH_CHECK(!HasNonZeroNear(rendered, 125, 200, 2));
}

// --- Task 13 round 3: the holes round 2 left -------------------------

// A MISTYPED required key used to be answered "missing required key",
// with no suggestion and with the location degraded to a bare array
// index - leaving the two most important keys in the schema as the two
// without a "did you mean". RejectUnknownKeys now runs BEFORE the
// required keys are read, so a typo is answered as a typo.
void TestParseSuggestsForAMistypedRequiredKey() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\"}],"
        "\"edges\":[{\"form\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    // The suggestion, not merely the fact of an error: the OLD behaviour
    // already errored here, so a test that only checked for a throw would
    // have passed before this change.
    GRAPH_CHECK(message.find("did you mean \"from\"") != std::string::npos);
    GRAPH_CHECK(message.find("unknown key \"form\"") != std::string::npos);
    // And the label: the endpoint that IS readable, with "?" for the one
    // the typo removed.
    GRAPH_CHECK(message.find("?->\"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("missing required key") == std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"di\":\"od\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"id\"") != std::string::npos);
    // The id is the very thing missing, so the label falls back to the
    // model name rather than to the index alone.
    GRAPH_CHECK(message.find("(model \"m\")") != std::string::npos);
}

// A required key that is simply absent has nothing to suggest, but it
// still gets a label and - new here - a recovery line, so every error
// this parser raises now has all three parts.
void TestMissingRequiredKeyStillCarriesLabelAndRemedy() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"model\":\"m\"}],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("missing required key \"id\"") != std::string::npos);
    GRAPH_CHECK(message.find("(model \"m\")") != std::string::npos);
    GRAPH_CHECK(message.find("-> add \"id\"") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\"}],\"edges\":[{\"from\":\"cam\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("missing required key \"to\"") != std::string::npos);
    GRAPH_CHECK(message.find("\"cam\"->?") != std::string::npos);
    GRAPH_CHECK(message.find("-> add \"to\"") != std::string::npos);
}

// Every unknown key in one object, in one message. nlohmann iterates in
// sorted key order, so reporting only the first hid "paddng" behind
// "classez" and cost a second round trip to find it.
void TestParseReportsEveryUnknownKeyInOneMessage() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\"},{\"id\":\"r\",\"model\":\"n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"r\","
        "\"roi\":{\"paddng\":0.5,\"classez\":[\"person\"]}}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("2 unknown keys") != std::string::npos);
    GRAPH_CHECK(message.find("\"classez\"") != std::string::npos);
    GRAPH_CHECK(message.find("\"paddng\"") != std::string::npos);
    // Each suggestion stays attached to the key it is for.
    GRAPH_CHECK(message.find("\"classes\" for \"classez\"") != std::string::npos);
    GRAPH_CHECK(message.find("\"pad\" for \"paddng\"") != std::string::npos);
}

// A node "type" the schema does not define. The dangerous case is a node
// that ALSO carries "model": it used to be silently accepted as a model
// node, and {"type":"model", ...} - the most natural thing a hand-editor
// writes - worked only by that accident.
void TestParseRejectsUnknownNodeType() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"type\":\"model\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown node type \"model\"") != std::string::npos);
    GRAPH_CHECK(message.find("\"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("no \"type\"") != std::string::npos);
    // Nothing close to "source", so no guess is offered.
    GRAPH_CHECK(message.find("did you mean") == std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"type\":\"sink\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown node type \"sink\"") != std::string::npos);

    // A near miss does get one.
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"sourse\",\"uri\":\"a.jpg\"}],"
        "\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"source\"") != std::string::npos);

    // "fuse" keeps its own, more informative code.
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"f\",\"type\":\"fuse\"}],"
        "\"edges\":[]}",
        GraphErrorCode::kGraphReserved, &message));
    GRAPH_CHECK(message.find("reserved") != std::string::npos);
}

// "$schema" is accepted and ignored: editor tooling adds it, and no
// schema key is within edit distance 2 of it, so tolerating it cannot
// mask a typo for a real key. The tolerance stops there - a "_comment"
// key is still rejected, because every extra tolerated key is somewhere
// a typo can hide.
void TestTopLevelSchemaKeyIsAcceptedAndInert() {
    const std::string json =
        "{\"$schema\":\"https://example/graph.schema.json\",\"version\":1,"
        "\"name\":\"s\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"m\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}";
    std::string thrown;
    GraphSpec spec;
    try {
        spec = ParseGraphText(json, "schema.json");
    } catch (const std::exception& error) {
        thrown = error.what();
    }
    GRAPH_CHECK(thrown.empty());
    if (!thrown.empty()) {
        std::printf("      TestTopLevelSchemaKeyIsAcceptedAndInert: %s\n",
                    thrown.c_str());
        return;
    }
    GRAPH_CHECK(spec.name == "s");
    GRAPH_CHECK(spec.nodes.size() == 2);

    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"_comment\":\"hi\",\"nodes\":[],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("_comment") != std::string::npos);
}

// --- Final review round: composition holes ---------------------------

/// Validate `json` against a registry, returning the GraphError message
/// and whether the code matched. Mirrors ThrowsWithCode, which only runs
/// the parser.
bool ValidateThrowsWithCode(const std::string& json,
                            const IModelRegistry& registry,
                            GraphErrorCode expected, std::string* message) {
    try {
        GraphSpec spec = ParseGraphText(json, "test.json");
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        *message = error.what();
        return error.code() == expected;
    } catch (...) {
        *message = "non-GraphError exception";
        return false;
    }
    *message = "no exception";
    return false;
}

// A plain edge carries the SOURCE FRAME and nothing else: SyncExecutor
// sets frame_inbox[edge.to] and drops the producer's payload, and
// AsyncExecutor assigns input.image = frame. So an edge out of a MODEL
// node conveyed ordering and threw that model's output away, while
// validating clean and exiting 0. On hardware, "cam->od, od->seg" and
// "cam->od, cam->seg" gave "seg" a byte-identical payload digest.
//
// Both executors lost the same data, so the parity suite agreed -
// agreement is not correctness, and this is the case that shows why.
void TestValidateRejectsAPlainEdgeThatWouldDiscardItsProducer() {
    FakeModelRegistry registry = BuildCascadeRegistry();

    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;
    registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));

    std::string message;
    GRAPH_CHECK(ValidateThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"seg\"}]}",
        registry, GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("would be discarded") != std::string::npos);
    GRAPH_CHECK(message.find("plain edge carries only a source frame or an image") !=
                std::string::npos);
    // A box producer can be cropped, so that is the remedy offered.
    GRAPH_CHECK(message.find("add \"roi\": {}") != std::string::npos);

    // Task 2 (image hand-off on a plain edge): an image producer's output IS
    // now handed off across a plain edge, so this graph - once rejected as
    // "does not pass one model's output to another" - is accepted.
    ModelInfo denoiser;
    denoiser.model_name = "dncnn_15";
    denoiser.task = "image_denoising";
    denoiser.output_shape = Shape::kImage;
    denoiser.input_contract = InputContract::kEither;
    denoiser.ready = true;
    registry.AddModel(denoiser, StageDataPtr(new ImageData()));

    GraphSpec handoff_spec = ParseGraphText(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"dn\",\"model\":\"dncnn_15\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"dn\"},"
        "{\"from\":\"dn\",\"to\":\"od\"}]}",
        "test.json");
    bool threw = false;
    try {
        ValidateGraph(handoff_spec, registry);
    } catch (const std::exception&) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
}

// The other half, and the more dangerous one to get wrong: a rule that
// rejects a legitimate composition is worse than the silence it replaces.
// Every plain edge out of a SOURCE must still validate, including a
// three-way fan-out and a source feeding a node that also has ROI
// children.
void TestValidateStillAcceptsEveryPlainEdgeFromASource() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    ModelInfo segmenter;
    segmenter.model_name = "bisenetv2";
    segmenter.task = "semantic_segmentation";
    segmenter.output_shape = Shape::kLabelMap;
    segmenter.input_contract = InputContract::kFullFrame;
    segmenter.ready = true;
    registry.AddModel(segmenter, StageDataPtr(new LabelMapData()));

    const std::string json =
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{}}]}";
    std::string thrown;
    try {
        GraphSpec spec = ParseGraphText(json, "valid.json");
        ValidateGraph(spec, registry);
    } catch (const std::exception& error) {
        thrown = error.what();
    }
    GRAPH_CHECK(thrown.empty());
    if (!thrown.empty()) {
        std::printf("      TestValidateStillAcceptsEveryPlainEdge...: %s\n",
                    thrown.c_str());
    }
}

// The model name is the one string the README tells a user to edit, and
// it used to be the only typo answered with no suggestion at all - just
// "run --list-models" against a 348-row table. Registry spelling differs
// from the model zoo's for 119 of those 348, so arriving with the zoo's
// spelling is the normal mistake, not an unusual one.
void TestValidateSuggestsANearbyModelName() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    std::string message;
    GRAPH_CHECK(ValidateThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"YoloV8N\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        registry, GraphErrorCode::kModelUnknown, &message));
    GRAPH_CHECK(message.find("did you mean \"yolov8n\"") != std::string::npos);
    GRAPH_CHECK(message.find("--list-models") != std::string::npos);

    // Nothing close: no guess, but still the pointer at the full list.
    GRAPH_CHECK(ValidateThrowsWithCode(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"zzzzzzzzzzzz\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        registry, GraphErrorCode::kModelUnknown, &message));
    GRAPH_CHECK(message.find("did you mean") == std::string::npos);
    GRAPH_CHECK(message.find("docs/graph_models.md") != std::string::npos);
}

namespace {

// Implements exactly IStage's pure members. If poll() were pure, this type
// would be abstract - and so would TypedStage, which does not override poll().
// The non-pure default is what lets clause (6) land without touching the
// stage that runs on real hardware.
class OnlyPureStage : public IStage {
 public:
    StageResult run(const StageInput&) { return StageResult(); }
    void submit(const StageInput&, StageCallback) {}
    void flush() {}
    Shape outputShape() const { return Shape::kBoxes; }
    InputContract inputContract() const { return InputContract::kFullFrame; }
};
static_assert(!std::is_abstract<OnlyPureStage>::value,
              "IStage::poll() must keep a default so existing stages need no change");

std::vector<int> ShuffledDeliveryOrder(unsigned seed) {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetShuffleSeed("casvit_t", seed);
    std::unique_ptr<IStage> owned =
        registry.createStage("casvit_t", "", StageParams());
    FakeStage* stage = registry.last_stage("casvit_t");
    std::shared_ptr<std::vector<int> > order(new std::vector<int>());
    if (stage == NULL) return *order;
    for (int i = 0; i < 6; ++i) {
        StageInput input;
        input.origin.from_roi = true;
        input.origin.roi_index = i;
        stage->submit(input, [order](const StageResult& result,
                                     const std::string&) {
            order->push_back(result.origin.roi_index);
        });
    }
    stage->poll();
    return *order;
}

}  // namespace

void TestFakeStagePollDeliversPending() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    std::unique_ptr<IStage> owned =
        registry.createStage("casvit_t", "", StageParams());
    FakeStage* stage = registry.last_stage("casvit_t");
    GRAPH_CHECK(stage != NULL);
    if (stage == NULL) return;
    std::shared_ptr<int> fired(new int(0));
    for (int i = 0; i < 2; ++i) {
        stage->submit(StageInput(), [fired](const StageResult&,
                                            const std::string&) { *fired += 1; });
    }
    GRAPH_CHECK(stage->pending() == 2);
    stage->poll();
    GRAPH_CHECK(*fired == 2);
    GRAPH_CHECK(stage->pending() == 0);
}

// The progress clause exists because a stage may legally hold completions
// until somebody asks. This fake mode is that stage: flush() delivers
// nothing, poll() delivers everything.
void TestFakeStageCanWithholdCompletionsFromFlush() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFlushDelivers("casvit_t", false);
    std::unique_ptr<IStage> owned =
        registry.createStage("casvit_t", "", StageParams());
    FakeStage* stage = registry.last_stage("casvit_t");
    GRAPH_CHECK(stage != NULL);
    if (stage == NULL) return;
    std::shared_ptr<int> fired(new int(0));
    for (int i = 0; i < 2; ++i) {
        stage->submit(StageInput(), [fired](const StageResult&,
                                            const std::string&) { *fired += 1; });
    }
    stage->flush();
    GRAPH_CHECK(*fired == 0);
    GRAPH_CHECK(stage->pending() == 2);
    stage->poll();
    GRAPH_CHECK(*fired == 2);
}

void TestFakeStageShuffleIsDeterministic() {
    const std::vector<int> a = ShuffledDeliveryOrder(7);
    const std::vector<int> b = ShuffledDeliveryOrder(7);
    std::vector<int> forward;
    for (int i = 0; i < 6; ++i) forward.push_back(i);
    GRAPH_CHECK(a.size() == 6);
    GRAPH_CHECK(a == b);        // same seed, same order, every run
    GRAPH_CHECK(a != forward);  // and it really permutes
}

void TestAsyncOptionsDefaults() {
    AsyncOptions options;
    GRAPH_CHECK(options.max_frames_in_flight == 16);
    GRAPH_CHECK(options.max_jobs_per_stage == 0);
}

void TestAsyncRejectsZeroFramesInFlight() {
    AsyncOptions options;
    options.max_frames_in_flight = 0;
    bool threw = false;
    try {
        AsyncExecutor executor(options);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    GRAPH_CHECK(threw);
}

// Clause (6) in action: a stage that delivers only when poll()ed. The old
// flush-driven harvest reads an empty collector from such a stage and loses
// the detector's result; a completion-driven harvest must not.
void TestAsyncCompletesWhenStageDeliversOnlyOnPoll() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFlushDelivers("yolov8n", false);
    registry.SetFlushDelivers("casvit_t", false);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncOptions options;
    options.stall_timeout_ms = 2000;  // a regression fails instead of hanging
    AsyncExecutor executor(options);
    FrameReport actual;
    try {
        actual = executor.RunFrame(graph, frame, 0);
    } catch (const std::exception& error) {
        std::printf("      unexpected: %s\n", error.what());
    }
    GRAPH_CHECK(expected == actual);
}

namespace {
struct StuckRecorder {
    int calls;
    std::vector<std::string> ids;
    StuckRecorder() : calls(0) {}
};
}  // namespace

void TestAsyncStallTimeoutNamesTheStuckStage() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFlushDelivers("yolov8n", false);
    registry.SetPollDelivers("yolov8n", false);  // never delivers at all
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncOptions options;
    options.stall_timeout_ms = 100;
    AsyncExecutor executor(options);
    std::string message;
    try {
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    } catch (const std::runtime_error& error) {
        message = error.what();
    }
    GRAPH_CHECK(message.find("stalled") != std::string::npos);
    GRAPH_CHECK(message.find("\"od\"") != std::string::npos);
}

// Review Focus #5: Ctrl-C while a stage never delivers must say which stage,
// once - not print on every wait slice, not stay silent.
void TestAsyncReportsStuckStagesOnceWhenInterrupted() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetFlushDelivers("yolov8n", false);
    registry.SetPollDelivers("yolov8n", false);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    std::shared_ptr<StuckRecorder> seen(new StuckRecorder());
    // Final review F2: the report waits for stuck_report_ms of silence, so
    // it must come after that much time and still before the stall.
    typedef std::chrono::steady_clock Clock;
    const Clock::time_point start = Clock::now();
    std::shared_ptr<long> reported_after_ms(new long(-1));
    AsyncOptions options;
    options.stall_timeout_ms = 200;
    options.stuck_report_ms = 50;
    options.interrupted = [] { return true; };
    options.on_stuck = [seen, start, reported_after_ms](
                           const std::vector<std::string>& ids) {
        seen->calls += 1;
        seen->ids = ids;
        *reported_after_ms = static_cast<long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                Clock::now() - start).count());
    };
    AsyncExecutor executor(options);
    bool threw = false;
    try {
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    GRAPH_CHECK(threw);
    GRAPH_CHECK(seen->calls == 1);
    GRAPH_CHECK(*reported_after_ms >= 50);
    GRAPH_CHECK(seen->ids.size() == 1);
    if (seen->ids.size() != 1) return;
    GRAPH_CHECK(seen->ids[0] == "od");
}

void TestAsyncOptionsWaitDefaults() {
    AsyncOptions options;
    GRAPH_CHECK(options.wait_slice_ms == 5);
    GRAPH_CHECK(options.stall_timeout_ms == 0);
    GRAPH_CHECK(!options.interrupted);
    GRAPH_CHECK(!options.on_stuck);
}

// The poll()-throw path. An exception escaping poll() is not dropped: it is
// kept against the node and reported when that node is next harvested, the
// way SyncExecutor reports a throwing run(). This stage throws on its first
// poll() and delivers normally on the next, so the frame completes and the
// message must be in its report.
void TestAsyncReportsExceptionEscapingPoll() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const std::size_t od = NodeIndexById(graph, "od");
    GRAPH_CHECK(od < graph.nodes().size());
    if (od >= graph.nodes().size()) return;
    graph.mutable_nodes()[od].stage.reset(
        new ContractBreakingStage(ContractBreakingStage::kThrowOnFirstPoll));

    AsyncOptions options;
    options.stall_timeout_ms = 2000;  // a regression fails instead of hanging
    AsyncExecutor executor(options);
    FrameReport report;
    bool threw = false;
    try {
        report = executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    } catch (const std::exception& error) {
        threw = true;
        std::printf("      unexpected: %s\n", error.what());
    }
    GRAPH_CHECK(!threw);
    GRAPH_CHECK(report.error == "node \"od\": poll exploded");
    GRAPH_CHECK(report.node_results.count("od") == 1);
    GRAPH_CHECK(report.roi_results.count("reid") == 1);
}

// One executor, one StageGraph object, rebuilt in place with a different
// node count. The executor must not trust the graph's address as identity:
// per-node state sized for the first build would be indexed out of bounds
// by the second.
void TestAsyncSurvivesGraphRebuiltInPlace() {
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FakeModelRegistry cascade_registry = BuildCascadeRegistry();
    FakeModelRegistry headline_registry = BuildHeadlineRegistry();
    const GraphSpec cascade = ParseGraphText(CascadeJson(), "t.json");
    const GraphSpec headline = ParseGraphText(HeadlineJson(), "headline.json");

    FakeModelRegistry sync_registry = BuildHeadlineRegistry();
    StageGraph sync_graph;
    sync_graph.Build(headline, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);

    AsyncOptions options;
    options.stall_timeout_ms = 2000;  // a regression fails instead of hanging
    AsyncExecutor executor(options);
    StageGraph graph;
    graph.Build(cascade, cascade_registry, "/models", false);
    GRAPH_CHECK(graph.nodes().size() == 3);
    FrameReport actual;
    try {
        executor.RunFrame(graph, frame, 0);
        graph.Build(headline, headline_registry, "/models", false);
        GRAPH_CHECK(graph.nodes().size() == 6);
        actual = executor.RunFrame(graph, frame, 0);
    } catch (const std::exception& error) {
        std::printf("      unexpected: %s\n", error.what());
    }
    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
}

namespace {

cv::Mat FrameWithKey(int key) {
    cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    frame.at<cv::Vec3b>(0, 0)[0] = static_cast<unsigned char>(key);
    return frame;
}

std::shared_ptr<BoxesData> PeopleAt(bool left, bool right) {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    if (left) boxes->items.push_back(MakeItem(10, 10, 40, 60, 0.9f, "person"));
    if (right) boxes->items.push_back(MakeItem(300, 200, 40, 60, 0.9f, "person"));
    return boxes;
}

/// The tracked cascade, but the detector sees a different scene per frame:
/// frame 0 left person, frame 1 both, frame 2 right person, frame 3 nobody.
/// Tracked in order: left=0, right=1. Tracked in reverse (2,1,0): right=0,
/// left=1 - so a gate that lets frames reach the tracker out of order
/// produces different track ids, which the report comparison catches.
FakeModelRegistry BuildMovingCascadeRegistry() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetScriptByFirstPixel("yolov8n", 0, PeopleAt(true, false));
    registry.SetScriptByFirstPixel("yolov8n", 1, PeopleAt(true, true));
    registry.SetScriptByFirstPixel("yolov8n", 2, PeopleAt(false, true));
    registry.SetScriptByFirstPixel("yolov8n", 3, PeopleAt(false, false));
    return registry;
}

std::vector<FrameReport> SyncReports(const std::vector<int>& keys) {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    std::vector<FrameReport> out;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out.push_back(executor.RunFrame(graph, FrameWithKey(keys[i]), i));
    }
    return out;
}

AsyncOptions Frames(std::size_t frames) {
    AsyncOptions options;
    options.max_frames_in_flight = frames;
    options.stall_timeout_ms = 2000;  // a regression fails instead of hanging
    return options;
}

}  // namespace

// Spec 8.1: output equality cannot show a pipeline - frame-atomic and
// pipelined execution produce byte-identical reports - so measure the
// frames in flight. Reintroducing frame-atomicity makes the peak 1.
void TestAsyncKeepsSeveralFramesInFlight() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(3));
    std::string failure;
    int emitted = 0;
    try {
        for (int i = 0; i < 6; ++i) executor.Submit(graph, FrameWithKey(i % 3), i);
        GRAPH_CHECK(executor.peak_frames_in_flight() == 3);
        executor.Finish(graph);
        FrameReport report;
        while (executor.TryNext(&report)) ++emitted;
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(emitted == 6);
}

// The tracker gate: detector completions arrive in REVERSE frame order.
void TestAsyncTrackerGateOrdersReversedCompletions() {
    std::vector<int> keys;
    keys.push_back(0); keys.push_back(1); keys.push_back(2);
    const std::vector<FrameReport> expected = SyncReports(keys);

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(3));
    std::string failure;
    try {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            executor.Submit(graph, FrameWithKey(keys[i]), i);
        }
        executor.Finish(graph);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            FrameReport actual;
            GRAPH_CHECK(executor.TryNext(&actual));
            GRAPH_CHECK(actual.frame_index == i);
            GRAPH_CHECK(expected[i] == actual);
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
}

// The detector seed is chosen so the test discriminates the tracker gate.
// Seed 11 (the original) happened to deliver the detector's frames in an
// order the tracker tolerates, so it passed with the gate disabled too. A
// scratch search over detector seeds 0..999 (casvit_t kept at 23), with
// CompleteRun's gate lookup forced to "no gate", found 325 seeds whose
// reports differ from sync; 3 is the smallest (6 of 8 frames differ).
void TestAsyncMatchesSyncAcrossFramesUnderShuffledDelivery() {
    std::vector<int> keys;
    for (int i = 0; i < 8; ++i) keys.push_back(i % 4);
    const std::vector<FrameReport> expected = SyncReports(keys);

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetShuffleSeed("yolov8n", 3);
    registry.SetShuffleSeed("casvit_t", 23);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(4));
    std::string failure;
    try {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            executor.Submit(graph, FrameWithKey(keys[i]), i);
        }
        executor.Finish(graph);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            FrameReport actual;
            GRAPH_CHECK(executor.TryNext(&actual));
            GRAPH_CHECK(expected[i] == actual);
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
}

// Frame 1 (nobody in it) finishes after the detector; frame 0 still waits
// for its reid crop. TryNext must not hand out frame 1 first.
void TestAsyncEmitsReportsInFrameOrder() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    FakeStage* reid = registry.last_stage("casvit_t");
    GRAPH_CHECK(reid != NULL);
    if (reid == NULL) return;
    reid->SetPollDelivers(false);
    reid->SetFlushDelivers(false);

    AsyncExecutor executor(Frames(2));
    std::string failure;
    try {
        executor.Submit(graph, FrameWithKey(0), 0);  // one person -> one crop
        executor.Submit(graph, FrameWithKey(3), 1);  // nobody -> zero crops
        FrameReport report;
        GRAPH_CHECK(!executor.TryNext(&report));      // frame 1 is done, frame 0 is not
        reid->SetPollDelivers(true);
        reid->SetFlushDelivers(true);
        GRAPH_CHECK(executor.TryNext(&report));
        GRAPH_CHECK(report.frame_index == 0);
        GRAPH_CHECK(executor.TryNext(&report));
        GRAPH_CHECK(report.frame_index == 1);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
}

// Review Focus #4.
void TestAsyncEmptyFrameDoesNotBlockTheTrackerGate() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(3));
    std::string failure;
    try {
        executor.Submit(graph, FrameWithKey(0), 0);
        executor.Submit(graph, cv::Mat(), 1);
        executor.Submit(graph, FrameWithKey(2), 2);
        executor.Finish(graph);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    FrameReport report;
    GRAPH_CHECK(executor.TryNext(&report) && report.error.empty());
    GRAPH_CHECK(executor.TryNext(&report) &&
                report.error == "source produced an empty frame");
    GRAPH_CHECK(executor.TryNext(&report) && report.frame_index == 2 &&
                report.error.empty());
}

// Review Focus #3.
void TestAsyncGateAdvancesPastFailedTrackedNode() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetFailure("yolov8n", "device busy");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(3));
    std::string failure;
    try {
        for (int i = 0; i < 4; ++i) executor.Submit(graph, FrameWithKey(i), i);
        executor.Finish(graph);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    int failed = 0;
    FrameReport report;
    while (executor.TryNext(&report)) {
        if (report.error.find("device busy") != std::string::npos) ++failed;
    }
    GRAPH_CHECK(failed == 4);
}

// Review Focus #1.
void TestAsyncNeverAdmitsPastTheFrameCap() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    bool within = true;
    std::string failure;
    try {
        FrameReport report;
        for (int i = 0; i < 40; ++i) {
            executor.Submit(graph, FrameWithKey(i % 4), i);
            if (executor.frames_in_flight() > 2) within = false;
            while (executor.TryNext(&report)) {}
        }
        executor.Finish(graph);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(within);
    GRAPH_CHECK(executor.frames_in_flight() == 0);
}

// Review Focus #2.
void TestAsyncSubmitsEverythingThenDrainsInOrder() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    std::size_t next = 0;
    bool ordered = true;
    std::string failure;
    try {
        for (int i = 0; i < 10; ++i) executor.Submit(graph, FrameWithKey(i % 4), i);
        executor.Finish(graph);
        FrameReport report;
        while (executor.TryNext(&report)) {
            if (report.frame_index != next) ordered = false;
            ++next;
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(ordered);
    GRAPH_CHECK(next == 10);
}

void TestAsyncRunFrameRefusesWhileFramesAreInFlight() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    std::string failure;
    bool threw = false;
    try {
        executor.Submit(graph, FrameWithKey(0), 0);
        try {
            executor.RunFrame(graph, FrameWithKey(1), 1);
        } catch (const std::logic_error&) {
            threw = true;
        }
        executor.Finish(graph);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(threw);
}

// The other half of RunFrame's precondition: nothing in flight, but a
// finished report nobody took. Running would hand back that stale report
// (TryNext emits in frame order) instead of this frame's, so it refuses.
void TestAsyncRunFrameRefusesWhileAFinishedReportIsUntaken() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    std::string failure;
    bool threw = false;
    std::string message;
    try {
        executor.Submit(graph, FrameWithKey(0), 0);
        executor.Finish(graph);
        GRAPH_CHECK(executor.frames_in_flight() == 0);
        try {
            executor.RunFrame(graph, FrameWithKey(1), 1);
        } catch (const std::logic_error& error) {
            threw = true;
            message = error.what();
        }
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(threw);
    GRAPH_CHECK(message ==
                "AsyncExecutor::RunFrame: frames from Submit() are still in "
                "flight or their reports are untaken; call Finish() and drain "
                "with TryNext() first");
}

void TestAsyncTryNextHandsBackTheReportsOwnFrame() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    std::string failure;
    try {
        executor.Submit(graph, FrameWithKey(2), 0);
        executor.Submit(graph, FrameWithKey(1), 1);
        executor.Finish(graph);
        FrameReport report;
        cv::Mat frame;
        GRAPH_CHECK(executor.TryNext(&report, &frame));
        GRAPH_CHECK(!frame.empty() && frame.at<cv::Vec3b>(0, 0)[0] == 2);
        GRAPH_CHECK(executor.TryNext(&report, &frame));
        GRAPH_CHECK(!frame.empty() && frame.at<cv::Vec3b>(0, 0)[0] == 1);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
}

// Bind re-derives the tracker gates whenever the executor is idle. A graph
// rebuilt in place keeps its address and here its node count, but gains a
// tracked node; keyed on address and size alone the gate set would stay
// empty, and reversed detector completions would reach the tracker out of
// order.
void TestAsyncRebindsTrackerGatesAfterInPlaceRebuild() {
    std::vector<int> keys;
    keys.push_back(0); keys.push_back(1); keys.push_back(2);
    const std::vector<FrameReport> expected = SyncReports(keys);

    const std::string untracked =
        "{\"version\":1,\"name\":\"t\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
    GraphSpec plain = ParseGraphText(untracked, "u.json");
    FakeModelRegistry plain_registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(plain, plain_registry, "/models", false);
    AsyncExecutor executor(Frames(3));
    std::string bind_failure;
    try {
        executor.RunFrame(graph, FrameWithKey(0), 0);  // binds with no tracked node
    } catch (const std::exception& error) {
        bind_failure = error.what();
    }
    GRAPH_CHECK(bind_failure.empty());

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
    graph.Build(spec, registry, "/models", false);
    GRAPH_CHECK(graph.nodes().size() == 3);
    std::string failure;
    try {
        for (std::size_t i = 0; i < keys.size(); ++i) {
            executor.Submit(graph, FrameWithKey(keys[i]), i);
        }
        executor.Finish(graph);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        FrameReport actual;
        GRAPH_CHECK(executor.TryNext(&actual));
        GRAPH_CHECK(expected[i] == actual);
    }
}

// =============================================================================
// Final whole-branch review fixes (F1-F4).
// =============================================================================

namespace {

/// A per-frame scene over keys 0..3, interleaved so consecutive frames
/// differ and every scene recurs.
std::vector<int> SceneKeys(std::size_t count) {
    std::vector<int> keys;
    for (std::size_t i = 0; i < count; ++i) {
        keys.push_back(static_cast<int>((i * 7 + i / 3) % 4));
    }
    return keys;
}

/// The headline graph with the moving scene on its tracked detector.
FakeModelRegistry BuildMovingHeadlineRegistry() {
    FakeModelRegistry registry = BuildHeadlineRegistry();
    registry.SetScriptByFirstPixel("yolov8n", 0, PeopleAt(true, false));
    registry.SetScriptByFirstPixel("yolov8n", 1, PeopleAt(true, true));
    registry.SetScriptByFirstPixel("yolov8n", 2, PeopleAt(false, true));
    registry.SetScriptByFirstPixel("yolov8n", 3, PeopleAt(false, false));
    return registry;
}

/// SyncExecutor over `keys` on a fresh graph built from `registry`.
std::vector<FrameReport> SyncReportsOn(const std::string& json,
                                       FakeModelRegistry registry,
                                       const std::vector<int>& keys) {
    GraphSpec spec = ParseGraphText(json, "x.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    std::vector<FrameReport> out;
    for (std::size_t i = 0; i < keys.size(); ++i) {
        out.push_back(executor.RunFrame(graph, FrameWithKey(keys[i]), i));
    }
    return out;
}

struct PipelinedRun {
    std::vector<FrameReport> reports;
    std::string failure;
    std::size_t peak;
    PipelinedRun() : peak(0) {}
};

/// Every frame through one AsyncExecutor. `stream` drains after every
/// Submit, the way the CLI does; otherwise everything is submitted first
/// and drained after Finish, which keeps max_frames_in_flight frames in
/// flight even with a fake that delivers the moment it is polled.
PipelinedRun RunPipelined(StageGraph& graph, const std::vector<int>& keys,
                          const AsyncOptions& options, bool stream) {
    PipelinedRun run;
    try {
        AsyncExecutor executor(options);
        FrameReport report;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            executor.Submit(graph, FrameWithKey(keys[i]), i);
            if (!stream) continue;
            while (executor.TryNext(&report)) run.reports.push_back(report);
        }
        executor.Finish(graph);
        while (executor.TryNext(&report)) run.reports.push_back(report);
        run.peak = executor.peak_frames_in_flight();
    } catch (const std::exception& error) {
        run.failure = error.what();
    }
    return run;
}

/// Frame-by-frame comparison: position, frame_index and content.
int CountMismatches(const std::vector<FrameReport>& expected,
                    const std::vector<FrameReport>& actual) {
    int bad = expected.size() == actual.size() ? 0 : 1;
    const std::size_t n =
        expected.size() < actual.size() ? expected.size() : actual.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (actual[i].frame_index != i || !(expected[i] == actual[i])) ++bad;
    }
    return bad;
}

std::size_t CropsIn(const std::vector<FrameReport>& reports,
                    const std::string& node) {
    std::size_t total = 0;
    for (std::size_t i = 0; i < reports.size(); ++i) {
        std::map<std::string, std::vector<StageResult> >::const_iterator it =
            reports[i].roi_results.find(node);
        if (it != reports[i].roi_results.end()) total += it->second.size();
    }
    return total;
}

/// Replace every stage of `graph` with a ThreadedStage around it.
void ThreadEveryStage(StageGraph& graph, unsigned seed, int min_delay_ms,
                      int max_delay_ms) {
    std::vector<NodeRuntime>& nodes = graph.mutable_nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].stage) continue;
        nodes[i].stage.reset(new ThreadedStage(
            std::move(nodes[i].stage), seed + static_cast<unsigned>(i),
            min_delay_ms, max_delay_ms));
    }
}

}  // namespace

// F1. The reorder buffer sits INSIDE the frame cap. Frame 0's reid crop is
// withheld; frames 1.. have nobody in them and finish at once. Counting
// only unfinished frames, Submit would admit every later frame and park its
// report (and frame copy) behind frame 0 without bound - the review's probe
// admitted 200 frames at cap 2 and emitted none. With the cap spanning from
// the oldest unfinished frame, frame 2 must not be admitted: Submit waits
// for frame 0 and the stall timeout ends the wait.
void TestAsyncHeldReportsCountAgainstTheFrameCap() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    FakeStage* reid = registry.last_stage("casvit_t");
    GRAPH_CHECK(reid != NULL);
    if (reid == NULL) return;
    reid->SetPollDelivers(false);
    reid->SetFlushDelivers(false);

    AsyncOptions options = Frames(2);
    options.stall_timeout_ms = 200;
    std::size_t admitted = 0;
    std::size_t emitted = 0;
    std::size_t in_flight_after = 99;
    bool within = true;
    std::string stall;
    {
        AsyncExecutor executor(options);
        FrameReport report;
        try {
            for (std::size_t i = 0; i < 12; ++i) {
                // frame 0: one person -> one withheld crop; later: nobody
                executor.Submit(graph, FrameWithKey(i == 0 ? 0 : 3), i);
                ++admitted;
                while (executor.TryNext(&report)) ++emitted;
                // Nothing was emitted past frame 0, so admitted - emitted is
                // next_seq minus the oldest unfinished frame's sequence.
                if (admitted - emitted > options.max_frames_in_flight) within = false;
            }
        } catch (const std::runtime_error& error) {
            stall = error.what();
        }
        in_flight_after = executor.frames_in_flight();
    }  // destroyed after the throw, as the header requires
    GRAPH_CHECK(stall.find("stalled") != std::string::npos);
    GRAPH_CHECK(stall.find("\"reid\"") != std::string::npos);
    GRAPH_CHECK(admitted == 2);         // frame 2 was refused, not admitted
    GRAPH_CHECK(emitted == 0);          // frame 1 is finished but held
    GRAPH_CHECK(in_flight_after == 1);  // only frame 0 is unfinished
    GRAPH_CHECK(within);
}

// Bind's idle test is frames.empty() alone: untaken finished reports hold
// no per-graph state, so a graph rebuilt in place (here with a different
// node count) must be re-bound rather than rejected as "a different
// StageGraph while frames are in flight".
void TestAsyncRebindsWhileReportsAreUntaken() {
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FakeModelRegistry cascade_registry = BuildCascadeRegistry();
    FakeModelRegistry headline_registry = BuildHeadlineRegistry();
    const GraphSpec cascade = ParseGraphText(CascadeJson(), "t.json");
    const GraphSpec headline = ParseGraphText(HeadlineJson(), "headline.json");

    FakeModelRegistry sync_registry = BuildHeadlineRegistry();
    StageGraph sync_graph;
    sync_graph.Build(headline, sync_registry, "/models", false);
    SyncExecutor sync_executor;
    const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 1);

    AsyncExecutor executor(Frames(2));
    StageGraph graph;
    graph.Build(cascade, cascade_registry, "/models", false);
    std::string failure;
    FrameReport first;
    FrameReport second;
    bool got_first = false;
    bool got_second = false;
    try {
        executor.Submit(graph, frame, 0);
        executor.Finish(graph);  // frame 0 finished, its report untaken
        graph.Build(headline, headline_registry, "/models", false);
        executor.Submit(graph, frame, 1);
        executor.Finish(graph);
        got_first = executor.TryNext(&first);
        got_second = executor.TryNext(&second);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(got_first && first.frame_index == 0 &&
                first.roi_results["reid"].size() == 2);
    GRAPH_CHECK(got_second && expected == second);
}

// F2. A healthy stage delivering within stuck_report_ms of the interrupt
// must not be named. Every job takes 20-30 ms on the stage's own thread -
// hardware-like turnaround - so the executor sees many quiet 5 ms slices
// with interrupted() true; reporting at the first of them names healthy
// stages, which is what the review saw on a healthy Ctrl-C.
void TestAsyncDoesNotNameHealthyStagesWhenInterrupted() {
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    ThreadEveryStage(graph, 7, 20, 30);

    std::shared_ptr<StuckRecorder> seen(new StuckRecorder());
    AsyncOptions options = Frames(2);
    options.stuck_report_ms = 1000;
    options.interrupted = [] { return true; };
    options.on_stuck = [seen](const std::vector<std::string>& ids) {
        seen->calls += 1;
        seen->ids = ids;
    };
    std::vector<int> keys;
    for (int i = 0; i < 4; ++i) keys.push_back(i);
    const PipelinedRun run = RunPipelined(graph, keys, options, true);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.reports.size() == 4);
    GRAPH_CHECK(seen->calls == 0);
}

void TestAsyncStuckReportDefault() {
    AsyncOptions options;
    GRAPH_CHECK(options.stuck_report_ms == 1000);
}

// F3(a), spec 8.2: the flagship od+seg+pose(+reid+depth) graph, several
// frames in flight, every stage delivering in a seeded shuffled order.
void TestAsyncMatchesSyncOnHeadlineAcrossFramesUnderShuffledDelivery() {
    const std::vector<int> keys = SceneKeys(12);
    const std::vector<FrameReport> expected =
        SyncReportsOn(HeadlineJson(), BuildMovingHeadlineRegistry(), keys);
    GRAPH_CHECK(CropsIn(expected, "reid") > 0);

    GraphSpec spec = ParseGraphText(HeadlineJson(), "headline.json");
    FakeModelRegistry registry = BuildMovingHeadlineRegistry();
    registry.SetShuffleSeed("yolov8n", 11);
    registry.SetShuffleSeed("casvit_t", 12);
    registry.SetShuffleSeed("bisenetv2", 13);
    registry.SetShuffleSeed("yolov8n_pose", 14);
    registry.SetShuffleSeed("fastdepth_1", 15);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const PipelinedRun run = RunPipelined(graph, keys, Frames(6), false);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak == 6);
    GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
}

// F3(b): fan-in - one node with two ROI parents - across frames.
void TestAsyncMatchesSyncOnFanInAcrossFrames() {
    const std::vector<int> keys = SceneKeys(12);
    const std::vector<FrameReport> expected =
        SyncReportsOn(FanInJson(), BuildMovingCascadeRegistry(), keys);
    GRAPH_CHECK(CropsIn(expected, "reid") > 0);

    GraphSpec spec = ParseGraphText(FanInJson(), "fanin.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetShuffleSeed("yolov8n", 31);
    registry.SetShuffleSeed("casvit_t", 32);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const PipelinedRun run = RunPipelined(graph, keys, Frames(5), false);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak == 5);
    GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
}

// F3(c): the tracked node fails on every frame; the reports - errors
// included - must equal sync's frame by frame, not merely count errors.
void TestAsyncMatchesSyncAcrossFramesWhenTrackedNodeFails() {
    const std::vector<int> keys = SceneKeys(6);
    FakeModelRegistry sync_registry = BuildMovingCascadeRegistry();
    sync_registry.SetFailure("yolov8n", "device busy");
    const std::vector<FrameReport> expected =
        SyncReportsOn(CascadeJson(), sync_registry, keys);
    bool all_failed = expected.size() == keys.size();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i].error.find("device busy") == std::string::npos) {
            all_failed = false;
        }
    }
    GRAPH_CHECK(all_failed);

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetFailure("yolov8n", "device busy");
    registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const PipelinedRun run = RunPipelined(graph, keys, Frames(3), false);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak == 3);
    GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
}

// F3(d): frames in flight with one job per stage and reversed detector
// delivery on the tracked cascade. With a per-stage cap of 1 no stage ever
// holds two jobs, so the frames overlap ACROSS stages and the backlog feeds
// each stage in order; the test pins that combination and the cap itself.
void TestAsyncMatchesSyncAcrossFramesWithOneJobPerStage() {
    const std::vector<int> keys = SceneKeys(12);
    const std::vector<FrameReport> expected =
        SyncReportsOn(CascadeJson(), BuildMovingCascadeRegistry(), keys);
    GRAPH_CHECK(CropsIn(expected, "reid") > 0);

    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncOptions options = Frames(4);
    options.max_jobs_per_stage = 1;
    const PipelinedRun run = RunPipelined(graph, keys, options, false);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak == 4);
    GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
    FakeStage* od = registry.last_stage("yolov8n");
    FakeStage* reid = registry.last_stage("casvit_t");
    GRAPH_CHECK(od != NULL && od->max_pending() == 1);
    GRAPH_CHECK(reid != NULL && reid->max_pending() == 1);
}

// F3(e), clause (5)/(6) for real: every stage delivers from its own worker
// thread, out of order, with small random delays, while the driver streams
// frames the way the CLI does. Nothing here is deterministic except the
// expected reports.
//
// The cascade runs with max_age 0. With the default (30) a track outlives
// every gap in this scene, so ids depend only on which person appears
// first, and frame 0 - submitted to an empty queue - is nearly always
// delivered first: a missing tracker gate went unnoticed in 10 of 10 runs.
// With max_age 0 every absence drops the track and every reappearance mints
// a new id, so the id sequence depends on the whole frame order.
void TestAsyncMatchesSyncWithCrossThreadDelivery() {
    const std::string json =
        "{\"version\":1,\"name\":\"t\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\","
        "\"track\":{\"algo\":\"iou\",\"max_age\":0}},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
    const std::vector<int> keys = SceneKeys(30);
    const std::vector<FrameReport> expected =
        SyncReportsOn(json, BuildMovingCascadeRegistry(), keys);
    GRAPH_CHECK(CropsIn(expected, "reid") > 0);

    GraphSpec spec = ParseGraphText(json, "t.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    ThreadEveryStage(graph, 1234, 0, 2);
    AsyncOptions options = Frames(8);
    options.stall_timeout_ms = 5000;
    const PipelinedRun run = RunPipelined(graph, keys, options, true);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak > 1);
    GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
}

// =============================================================================
// Payload hand-off, Task 1 - coordinates. A node's `image` output can become
// its consumer's full-frame input; FrameView carries that image together with
// the map from its pixels back to the source frame, and every restore helper
// applies RoiRef::inv_align instead of short-circuiting on from_roi == false.
// A graph without hand-off only ever sees identity views, so everything below
// that touches the old paths pins them bit-exact.
// =============================================================================

namespace {
cv::Matx23f Affine(float a, float b, float c, float d, float e, float f) {
    return cv::Matx23f(a, b, c, d, e, f);
}
bool NearAffine(const cv::Matx23f& x, const cv::Matx23f& y) {
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 3; ++c)
            if (std::fabs(x(r, c) - y(r, c)) > 1e-5f) return false;
    return true;
}
bool NearRect(const cv::Rect2f& a, const cv::Rect2f& b) {
    return std::fabs(a.x - b.x) < 1e-3f && std::fabs(a.y - b.y) < 1e-3f &&
           std::fabs(a.width - b.width) < 1e-3f && std::fabs(a.height - b.height) < 1e-3f;
}
std::shared_ptr<ImageData> SolidImage(int w, int h, int type) {
    std::shared_ptr<ImageData> data(new ImageData());
    data->image = cv::Mat(h, w, type, cv::Scalar::all(7));
    return data;
}

/// Exact, field-by-field: the identity view must reproduce the frame
/// overload bit for bit, so no tolerance anywhere.
void CheckSameCrops(const std::vector<RoiCrop>& a, const std::vector<RoiCrop>& b) {
    GRAPH_CHECK(a.size() == b.size());
    if (a.size() != b.size()) return;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const RoiRef& x = a[i].ref;
        const RoiRef& y = b[i].ref;
        GRAPH_CHECK(x.from_roi == y.from_roi);
        GRAPH_CHECK(x.src_box.x == y.src_box.x && x.src_box.y == y.src_box.y &&
                    x.src_box.width == y.src_box.width &&
                    x.src_box.height == y.src_box.height);
        bool same_align = true;
        for (int r = 0; r < 2; ++r)
            for (int c = 0; c < 3; ++c)
                if (x.inv_align(r, c) != y.inv_align(r, c)) same_align = false;
        GRAPH_CHECK(same_align);
        GRAPH_CHECK(x.crop_size == y.crop_size);
        GRAPH_CHECK(x.track_id == y.track_id);
        GRAPH_CHECK(x.parent_node == y.parent_node);
        GRAPH_CHECK(x.parent_index == y.parent_index);
        GRAPH_CHECK(x.roi_index == y.roi_index);
        GRAPH_CHECK(a[i].image.size() == b[i].image.size());
        GRAPH_CHECK(a[i].image.type() == b[i].image.type());
        if (a[i].image.size() != b[i].image.size() ||
            a[i].image.type() != b[i].image.type()) continue;
        GRAPH_CHECK(cv::norm(a[i].image, b[i].image, cv::NORM_INF) == 0);
    }
}

/// 1280x960 BGR whose channel 0 at (x, y) is (x / 10) % 256, so a crop's
/// own pixels say which image (and which column) it was cut from.
FrameView ScaledColumnView() {
    FrameView view;
    view.image = cv::Mat(960, 1280, CV_8UC3, cv::Scalar::all(0));
    for (int y = 0; y < view.image.rows; ++y) {
        for (int x = 0; x < view.image.cols; ++x) {
            view.image.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>((x / 10) % 256);
        }
    }
    view.to_source = Affine(0.5f, 0, 0, 0, 0.5f, 0);
    return view;
}

/// The bounding box of every non-black pixel, or an empty rect.
cv::Rect NonBlackBounds(const cv::Mat& canvas) {
    cv::Mat gray;
    cv::cvtColor(canvas, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point> points;
    cv::findNonZero(gray, points);
    if (points.empty()) return cv::Rect();
    return cv::boundingRect(points);
}
}  // namespace

void TestHandOffProblemNamesEachUnusableImage() {
    GRAPH_CHECK(HandOffProblem(StageDataPtr()) == "handed off no image");
    GRAPH_CHECK(HandOffProblem(StageDataPtr(new BoxesData(Shape::kBoxes))) ==
                "handed off no image");
    std::shared_ptr<ImageData> empty(new ImageData());
    GRAPH_CHECK(HandOffProblem(empty) == "handed off an empty image");
    GRAPH_CHECK(HandOffProblem(SolidImage(8, 8, CV_8UC1)) == "handed off a non-BGR image");
    GRAPH_CHECK(HandOffProblem(SolidImage(8, 8, CV_8UC3)).empty());
}

void TestHandOffViewScalesPerAxis() {
    FrameView source;
    source.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0));
    const FrameView sr = HandOffView(*SolidImage(1280, 960, CV_8UC3), source);   // x2
    GRAPH_CHECK(sr.image.cols == 1280 && sr.image.rows == 960);
    GRAPH_CHECK(NearAffine(sr.to_source, Affine(0.5f, 0, 0, 0, 0.5f, 0)));
    const FrameView dn = HandOffView(*SolidImage(512, 512, CV_8UC3), source);    // squashed
    GRAPH_CHECK(NearAffine(dn.to_source, Affine(1.25f, 0, 0, 0, 0.9375f, 0)));
}

void TestHandOffViewComposesAChain() {
    FrameView source;
    source.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0));
    const FrameView dn = HandOffView(*SolidImage(512, 512, CV_8UC3), source);
    const FrameView sr = HandOffView(*SolidImage(1024, 1024, CV_8UC3), dn);      // x2 of 512
    // 1024 px -> 512 px -> 640 x 480 source
    GRAPH_CHECK(NearAffine(sr.to_source, Affine(0.625f, 0, 0, 0, 0.46875f, 0)));
}

void TestRestoreHelpersApplyInvAlignForFullFrameResults() {
    RoiRef scaled;                                   // from_roi == false
    scaled.inv_align = Affine(0.5f, 0, 0, 0, 0.5f, 0);
    GRAPH_CHECK(NearRect(RestoreBox(cv::Rect2f(100, 60, 40, 20), scaled),
                         cv::Rect2f(50, 30, 20, 10)));
    Keypoint k; k.x = 10; k.y = 30; k.confidence = 0.9f;
    const Keypoint rk = RestorePoint(k, scaled);
    GRAPH_CHECK(std::fabs(rk.x - 5.f) < 1e-5f && std::fabs(rk.y - 15.f) < 1e-5f);
    const cv::Point2f rp = RestorePointWarped(cv::Point2f(10, 30), scaled);
    GRAPH_CHECK(std::fabs(rp.x - 5.f) < 1e-5f && std::fabs(rp.y - 15.f) < 1e-5f);
    const RoiRef identity;                           // unchanged, bit-exact
    const cv::Rect2f odd(0.1f, 0.2f, 0.3f, 0.7f);
    const cv::Rect2f same = RestoreBox(odd, identity);
    GRAPH_CHECK(same.x == odd.x && same.y == odd.y && same.width == odd.width &&
                same.height == odd.height);
}

void TestRestorePointOnPlainCropIsUnchanged() {
    RoiRef crop;
    crop.from_roi = true;
    crop.src_box = cv::Rect2f(12.5f, 7.25f, 30, 40);
    crop.inv_align = Affine(1, 0, 12.5f, 0, 1, 7.25f);   // what RouteRois builds for a plain crop
    Keypoint k; k.x = 3.75f; k.y = 9.5f; k.confidence = 1.f;
    const Keypoint r = RestorePoint(k, crop);
    GRAPH_CHECK(r.x == 3.75f + 12.5f && r.y == 9.5f + 7.25f);
}

// Review Focus 1: a graph with no hand-off routes over identity views only,
// and those must be byte-identical to the frame overload for all three crop
// kinds (plain, OBB un-rotate, face5 align). Fixtures are the existing
// RouteRois tests' own: TestRouteFiltersByClassAndScore,
// TestRestoreBoxCornerRoundTripsThroughRealObbCrop and
// TestRestorePointWarpedRoundTripsThroughRealFace5Crop.
void TestRouteRoisIdentityViewEqualsFrameOverload() {
    // Plain.
    {
        BoxesData boxes = MakeBoxes(Shape::kBoxes);
        boxes.items.push_back(MakeItem(10, 10, 40, 60, 0.90f, "person"));
        boxes.items.push_back(MakeItem(80, 10, 40, 60, 0.85f, "car"));
        boxes.items.push_back(MakeItem(10, 90, 40, 60, 0.20f, "person"));
        cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
        RoiSpec spec;
        spec.present = true;
        spec.classes.push_back("person");
        spec.min_score = 0.5f;

        FrameView view;
        view.image = source;
        RouteStats frame_stats;
        RouteStats view_stats;
        const std::vector<RoiCrop> by_frame =
            RouteRois(boxes, source, spec, "od", NULL, &frame_stats);
        const std::vector<RoiCrop> by_view =
            RouteRois(boxes, view, spec, "od", NULL, &view_stats);
        GRAPH_CHECK(by_frame.size() == 1);
        CheckSameCrops(by_frame, by_view);
        GRAPH_CHECK(frame_stats.filtered == view_stats.filtered);
        GRAPH_CHECK(frame_stats.clipped_away == view_stats.clipped_away);
    }
    // OBB with a non-zero angle.
    {
        BoxesData boxes = MakeBoxes(Shape::kObBoxes);
        BoxItem obb = MakeItem(300.f, 200.f, 80.f, 60.f, 0.9f, "obj");
        obb.angle = 0.5f;
        boxes.items.push_back(obb);
        cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::circle(source, cv::Point(335, 210), 4, cv::Scalar(255, 255, 255), cv::FILLED);
        RoiSpec spec;
        spec.present = true;
        spec.pad = 0.f;

        FrameView view;
        view.image = source;
        const std::vector<RoiCrop> by_frame = RouteRois(boxes, source, spec, "od", NULL, NULL);
        const std::vector<RoiCrop> by_view = RouteRois(boxes, view, spec, "od", NULL, NULL);
        GRAPH_CHECK(by_frame.size() == 1);
        if (by_frame.size() == 1) {
            GRAPH_CHECK(std::fabs(by_frame[0].ref.inv_align(0, 1)) > 0.01f);  // really rotated
        }
        CheckSameCrops(by_frame, by_view);
    }
    // face5-aligned, with landmarks.
    {
        BoxesData boxes = MakeBoxes(Shape::kBoxes);
        BoxItem face = MakeItem(300.f, 200.f, 80.f, 100.f, 0.9f, "face");
        face.landmarks.push_back(Keypoint(320.f, 225.f));
        face.landmarks.push_back(Keypoint(360.f, 225.f));
        face.landmarks.push_back(Keypoint(340.f, 250.f));
        face.landmarks.push_back(Keypoint(322.f, 275.f));
        face.landmarks.push_back(Keypoint(358.f, 275.f));
        boxes.items.push_back(face);
        cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::circle(source, cv::Point(340, 250), 4, cv::Scalar(255, 255, 255), cv::FILLED);
        RoiSpec spec;
        spec.present = true;
        spec.pad = 0.f;
        spec.align = "face5";

        FrameView view;
        view.image = source;
        const std::vector<RoiCrop> by_frame = RouteRois(boxes, source, spec, "od", NULL, NULL);
        const std::vector<RoiCrop> by_view = RouteRois(boxes, view, spec, "od", NULL, NULL);
        GRAPH_CHECK(by_frame.size() == 1);
        if (by_frame.size() == 1) {
            GRAPH_CHECK(by_frame[0].ref.crop_size.width == by_frame[0].ref.crop_size.height);
            GRAPH_CHECK(by_frame[0].ref.inv_align(0, 0) != 1.f);  // really warped
        }
        CheckSameCrops(by_frame, by_view);
    }
}

void TestRouteRoisOverScaledViewCutsFromTheViewImage() {
    const FrameView view = ScaledColumnView();
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(100, 100, 80, 120, 0.9f, "person"));
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    const std::vector<RoiCrop> crops = RouteRois(boxes, view, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    const RoiCrop& crop = crops[0];
    GRAPH_CHECK(crop.image.size() == cv::Size(80, 120));
    if (crop.image.size() != cv::Size(80, 120)) return;
    GRAPH_CHECK(crop.image.at<cv::Vec3b>(0, 0)[0] == 10);   // view image, x = 100
    GRAPH_CHECK(crop.image.at<cv::Vec3b>(0, 79)[0] == 17);  // view image, x = 179
    GRAPH_CHECK(crop.ref.from_roi);
    GRAPH_CHECK(crop.ref.crop_size == cv::Size(80, 120));
    GRAPH_CHECK(NearRect(crop.ref.src_box, cv::Rect2f(50, 50, 40, 60)));
    GRAPH_CHECK(NearAffine(crop.ref.inv_align, Affine(0.5f, 0, 50, 0, 0.5f, 50)));
    GRAPH_CHECK(NearRect(RestoreBox(cv::Rect2f(0, 0, 80, 120), crop.ref), crop.ref.src_box));
}

// Review Focus 5: a box near the border of a super-resolved image is clipped
// against the SR image, not the 640x480 source, and its src_box lands
// inside the source frame.
void TestRouteRoisOverScaledViewClipsToTheViewImage() {
    const FrameView view = ScaledColumnView();
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(1240, 900, 100, 100, 0.9f, "person"));
    RoiSpec spec;
    spec.present = true;

    RouteStats stats;
    const std::vector<RoiCrop> crops = RouteRois(boxes, view, spec, "od", NULL, &stats);
    GRAPH_CHECK(crops.size() == 1);
    GRAPH_CHECK(stats.clipped_away == 0);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].image.size() == cv::Size(40, 60));
    GRAPH_CHECK(crops[0].ref.crop_size == cv::Size(40, 60));
    GRAPH_CHECK(NearRect(crops[0].ref.src_box, cv::Rect2f(620, 450, 20, 30)));
    GRAPH_CHECK(crops[0].ref.src_box.x + crops[0].ref.src_box.width <= 640.f);
    GRAPH_CHECK(crops[0].ref.src_box.y + crops[0].ref.src_box.height <= 480.f);
}

// SamePayload compares kImage exactly now (spec H9): a hand-off makes image
// payloads load-bearing, so parity on them can no longer fail closed.
void TestSamePayloadComparesImages() {
    std::shared_ptr<ImageData> left(new ImageData());
    left->image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    std::shared_ptr<ImageData> same(new ImageData());
    same->image = cv::Mat(4, 4, CV_8UC3, cv::Scalar(10, 20, 30));
    GRAPH_CHECK(MakeNodeReport("sr", left) == MakeNodeReport("sr", same));

    std::shared_ptr<ImageData> one_byte(new ImageData());
    one_byte->image = left->image.clone();
    one_byte->image.at<cv::Vec3b>(3, 2)[1] = 21;
    GRAPH_CHECK(!(MakeNodeReport("sr", left) == MakeNodeReport("sr", one_byte)));

    std::shared_ptr<ImageData> bigger(new ImageData());
    bigger->image = cv::Mat(4, 5, CV_8UC3, cv::Scalar(10, 20, 30));
    GRAPH_CHECK(!(MakeNodeReport("sr", left) == MakeNodeReport("sr", bigger)));

    // The roi_results path goes through the same comparator.
    RoiRef origin;
    origin.from_roi = true;
    origin.parent_node = "od";
    origin.parent_index = 0;
    origin.roi_index = 0;
    GRAPH_CHECK(MakeRoiReport("sr", origin, left) == MakeRoiReport("sr", origin, same));
    GRAPH_CHECK(!(MakeRoiReport("sr", origin, left) == MakeRoiReport("sr", origin, one_byte)));
}

// TargetRegion is file-local to graph_visualizer.cpp, so this goes through
// the public renderer: an ROI image result lands on the canvas exactly where
// TargetRegion says.
void TestTargetRegionPlacesScaledCropOnItsSourceBox() {
    const cv::Mat canvas = cv::Mat::zeros(480, 640, CV_8UC3);

    // A crop cut from a 2x view: 80x120 pixels covering source (50,50,40,60).
    {
        RoiRef origin;
        origin.from_roi = true;
        origin.src_box = cv::Rect2f(50, 50, 40, 60);
        origin.inv_align = Affine(0.5f, 0, 50, 0, 0.5f, 50);
        origin.crop_size = cv::Size(80, 120);
        origin.parent_node = "od";
        origin.parent_index = 0;
        origin.roi_index = 0;
        std::shared_ptr<ImageData> white(new ImageData());
        white->image = cv::Mat(120, 80, CV_8UC3, cv::Scalar::all(255));
        const cv::Mat rendered =
            RenderReport(canvas, MakeRoiReport("restore", origin, white));
        const cv::Rect bounds = NonBlackBounds(rendered);
        GRAPH_CHECK(std::abs(bounds.x - 50) <= 1 && std::abs(bounds.y - 50) <= 1);
        GRAPH_CHECK(std::abs(bounds.width - 40) <= 1 && std::abs(bounds.height - 60) <= 1);
    }

    // A plain crop (every crop today): the old formula, pixel for pixel -
    // the image copied at src_box's corner with crop_size, nothing else.
    {
        RoiRef origin;
        origin.from_roi = true;
        origin.src_box = cv::Rect2f(50, 50, 40, 60);
        origin.inv_align = Affine(1, 0, 50, 0, 1, 50);
        origin.crop_size = cv::Size(40, 60);
        origin.parent_node = "od";
        origin.parent_index = 0;
        origin.roi_index = 0;
        std::shared_ptr<ImageData> textured(new ImageData());
        textured->image = cv::Mat(60, 40, CV_8UC3);
        for (int y = 0; y < 60; ++y)
            for (int x = 0; x < 40; ++x)
                textured->image.at<cv::Vec3b>(y, x) =
                    cv::Vec3b(static_cast<uchar>(x * 6 + 1), static_cast<uchar>(y * 4 + 1), 200);
        const cv::Mat rendered =
            RenderReport(canvas, MakeRoiReport("restore", origin, textured));
        cv::Mat expected = canvas.clone();
        textured->image.copyTo(expected(cv::Rect(50, 50, 40, 60)));
        GRAPH_CHECK(rendered.size() == expected.size());
        if (rendered.size() != expected.size()) return;
        GRAPH_CHECK(cv::norm(rendered, expected, cv::NORM_INF) == 0);
    }
}

namespace {

ModelInfo ImageModel(const std::string& name) {
    ModelInfo info;
    info.model_name = name;
    info.task = "super_resolution";
    info.dxnn_file = name + ".dxnn";
    info.output_shape = Shape::kImage;
    info.input_contract = InputContract::kFullFrame;
    info.input_width = 192;
    info.input_height = 192;
    info.ready = true;
    return info;
}

/// cam 640x480 -> "sr" (x2 fake image) -> "od" (boxes in SR coordinates)
/// -> roi person -> "cls". Plus image producers with unusable output and a
/// classifier, for the error and validation cases.
///
/// DEVIATION (Task 2, ruling R1): every image model built by ImageModel()
/// above has input_contract kFullFrame, so a plain edge feeding a per-crop
/// node (od -> sr, roi) would already fail "requires a full frame" at the
/// AcceptsRoi(contract) check in step 4, before the per-crop-hand-off check
/// in the plain-edge branch is ever reached. "sr_either" is added so
/// TestValidateRejectsHandOffFromPerCropNode can feed a per-crop node from
/// an image producer whose contract accepts the ROI edge and still reach
/// the per-crop rejection this test is meant to pin.
FakeModelRegistry BuildHandOffRegistry() {
    FakeModelRegistry registry;

    std::shared_ptr<ImageData> x2(new ImageData());
    x2->image = cv::Mat(960, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
    for (int x = 0; x < 1280; ++x)
        for (int y = 0; y < 960; ++y) x2->image.at<cv::Vec3b>(y, x)[0] = (x / 10) % 256;
    registry.AddModel(ImageModel("sr_x2"), x2);

    // R1: same model as sr_x2 (and the same x2 payload), except its input
    // contract accepts an ROI edge, so it can stand in the per-crop-node
    // slot of TestValidateRejectsHandOffFromPerCropNode.
    ModelInfo sr_either = ImageModel("sr_either");
    sr_either.input_contract = InputContract::kEither;
    registry.AddModel(sr_either, x2);

    std::shared_ptr<ImageData> half(new ImageData());           // 640x480 -> 1280x960 -> 640x480
    half->image = cv::Mat(480, 640, CV_8UC3, cv::Scalar(1, 2, 3));
    registry.AddModel(ImageModel("sr_half"), half);

    registry.AddModel(ImageModel("sr_empty"), std::shared_ptr<ImageData>(new ImageData()));
    std::shared_ptr<ImageData> gray(new ImageData());
    gray->image = cv::Mat(960, 1280, CV_8UC1, cv::Scalar(5));
    registry.AddModel(ImageModel("sr_gray"), gray);

    ModelInfo detector;
    detector.model_name = "yolov8n";
    detector.task = "object_detection";
    detector.dxnn_file = "yolov8n.dxnn";
    detector.output_shape = Shape::kBoxes;
    detector.input_contract = InputContract::kFullFrame;
    detector.input_width = 640;
    detector.input_height = 640;
    detector.ready = true;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(100, 100, 80, 120, 0.9f, "person"));
    // Overhangs the SR image.
    boxes->items.push_back(MakeItem(1240, 900, 100, 100, 0.8f, "person"));
    registry.AddModel(detector, boxes);

    ModelInfo cls;
    cls.model_name = "resnet50";
    cls.task = "classification";
    cls.dxnn_file = "resnet50.dxnn";
    cls.output_shape = Shape::kScores;
    cls.input_contract = InputContract::kEither;
    cls.input_width = 224;
    cls.input_height = 224;
    cls.ready = true;
    std::shared_ptr<ScoresData> scores(new ScoresData());
    ClassificationResult top;
    top.class_id = 1;
    top.confidence = 0.7f;
    scores->items.push_back(top);
    registry.AddModel(cls, scores);
    return registry;
}

std::string HandOffJson(const std::string& sr_model) {
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"" + sr_model + "\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// ValidateGraph's error text, or "" if the graph is accepted.
std::string ValidationError(const std::string& json, const IModelRegistry& registry) {
    try {
        ValidateGraph(ParseGraphText(json, "h.json"), registry);
    } catch (const GraphError& error) {
        return error.what();
    }
    return std::string();
}

}  // namespace

void TestValidateAcceptsImageHandOff() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    GRAPH_CHECK(ValidationError(HandOffJson("sr_x2"), registry).empty());
}

// V3: an image producer that ALSO runs per-crop (it has an ROI in-edge)
// cannot hand off on a plain edge - there is no single full-frame image to
// hand off, only one crop-sized image per crop. Uses sr_either (ruling R1)
// so the ROI edge into "sr" itself validates (AcceptsRoi) and the per-crop
// check is what actually fires.
void TestValidateRejectsHandOffFromPerCropNode() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string json =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"sr\",\"model\":\"sr_either\"},"
        "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"sr\",\"roi\":{\"classes\":[\"person\"]}},"
        "{\"from\":\"sr\",\"to\":\"cls\"}]}";
    const std::string message = ValidationError(json, registry);
    GRAPH_CHECK(message.find("GRAPH_EDGE") != std::string::npos);
    GRAPH_CHECK(message.find("\"sr\" runs once per crop, so a plain edge "
                             "would hand off one image per crop") !=
                std::string::npos);
    GRAPH_CHECK(message.find("give \"cls\" an \"roi\" edge instead") !=
                std::string::npos);
}

// V4 (non-ROI-producer half): a node that produces neither an image nor
// boxes cannot be handed off on a plain edge at all.
void TestValidateRejectsHandOffOfNonImage() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string json =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cls\",\"model\":\"resnet50\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"cls\"},"
        "{\"from\":\"cls\",\"to\":\"od\"}]}";
    const std::string message = ValidationError(json, registry);
    GRAPH_CHECK(message.find("\"cls\" produces scores, and only an image "
                             "can be handed off on a plain edge") !=
                std::string::npos);
    GRAPH_CHECK(message.find("feed \"od\" from a source node or from a "
                             "node that produces an image") !=
                std::string::npos);
}

// V5: two DIFFERENT full-frame images into one node has no single meaning
// to run on, so it is refused - unlike two sources, which carry the same
// original frame no matter how many land on one node.
void TestValidateRejectsTwoFullFrameImages() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string two_images =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"sr2\",\"model\":\"sr_half\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
        "{\"from\":\"cam\",\"to\":\"sr2\"},"
        "{\"from\":\"sr\",\"to\":\"od\"},"
        "{\"from\":\"sr2\",\"to\":\"od\"}]}";
    std::string message = ValidationError(two_images, registry);
    GRAPH_CHECK(message.find("node \"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("receives full-frame input from more than one "
                             "image: \"sr\", \"sr2\"") != std::string::npos);
    GRAPH_CHECK(message.find("keep one plain edge into this node, or split "
                             "it into one node per input") !=
                std::string::npos);

    const std::string source_and_image =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
        "{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"sr\",\"to\":\"od\"}]}";
    message = ValidationError(source_and_image, registry);
    GRAPH_CHECK(message.find("node \"od\"") != std::string::npos);
    GRAPH_CHECK(message.find("receives full-frame input from more than one "
                             "image: \"cam\", \"sr\"") != std::string::npos);
}

// Two plain edges from two DIFFERENT sources are two streams (SP2): each
// frame of either stream brings "od" one full frame, so this is accepted.
void TestValidateStillAcceptsTwoSourceEdges() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string json =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"cam2\",\"to\":\"od\"}]}";
    GRAPH_CHECK(ValidationError(json, registry).empty());
}

// V6 is unchanged by hand-off: a node fed both a full frame (source or
// hand-off image) and ROI crops still has no defined answer to run on.
void TestValidateStillRejectsFrameAndRoiIntoOneNode() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string json =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
        "{\"from\":\"sr\",\"to\":\"od\"},"
        "{\"from\":\"sr\",\"to\":\"cls\"},"
        "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"]}}]}";
    const std::string message = ValidationError(json, registry);
    GRAPH_CHECK(message.find("receives both a full frame and ROI crops") !=
                std::string::npos);
}

// =============================================================================
// Payload hand-off, Task 3 - sync executor. The producer's image becomes its
// consumer's full-frame input on a plain edge, and ROI crops route from the
// view a node actually ran on, not always the source frame.
// =============================================================================

void TestSyncHandsOffTheProducersImage() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report;
    std::string failure;
    try {
        report = executor.RunFrame(
            graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results.count("sr") == 1);
    GRAPH_CHECK(report.node_results.count("od") == 1);

    const ImageData* sr_image =
        dynamic_cast<const ImageData*>(report.node_results["sr"].data.get());
    GRAPH_CHECK(sr_image != NULL);
    if (sr_image == NULL) return;

    FakeStage* yolo = registry.last_stage("yolov8n");
    GRAPH_CHECK(yolo != NULL);
    if (yolo == NULL) return;
    GRAPH_CHECK(yolo->seen().size() == 1);
    if (yolo->seen().size() != 1) return;

    const StageInput& input = yolo->seen()[0];
    GRAPH_CHECK(input.image.data == sr_image->image.data);
    GRAPH_CHECK(input.image.cols == 1280 && input.image.rows == 960);
    GRAPH_CHECK(!input.origin.from_roi);
    GRAPH_CHECK(NearAffine(input.origin.inv_align,
                          Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));

    // Propagated by the fake as TypedStage does: run() copies input.origin
    // into result.origin verbatim.
    GRAPH_CHECK(NearAffine(report.node_results["od"].origin.inv_align,
                          input.origin.inv_align));
}

void TestSyncRoutesCropsFromTheHandedOffImage() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report;
    std::string failure;
    try {
        report = executor.RunFrame(
            graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(report.error.empty());

    GRAPH_CHECK(report.roi_results.count("cls") == 1);
    const std::vector<StageResult>& cls_results = report.roi_results["cls"];
    GRAPH_CHECK(cls_results.size() == 2);
    if (cls_results.size() != 2) return;

    FakeStage* cls = registry.last_stage("resnet50");
    GRAPH_CHECK(cls != NULL);
    if (cls == NULL) return;
    GRAPH_CHECK(cls->seen().size() == 2);
    if (cls->seen().size() != 2) return;

    const cv::Mat& first_image = cls->seen()[0].image;
    const cv::Mat& second_image = cls->seen()[1].image;
    GRAPH_CHECK(first_image.cols == 80 && first_image.rows == 120);
    GRAPH_CHECK(second_image.cols == 40 && second_image.rows == 60);
    GRAPH_CHECK(first_image.at<cv::Vec3b>(0, 0)[0] == 10);

    GRAPH_CHECK(NearRect(cls_results[0].origin.src_box, cv::Rect2f(50, 50, 40, 60)));
    GRAPH_CHECK(NearAffine(cls_results[0].origin.inv_align,
                          Affine(0.5f, 0.f, 50.f, 0.f, 0.5f, 50.f)));
    GRAPH_CHECK(NearRect(cls_results[1].origin.src_box, cv::Rect2f(620, 450, 20, 30)));
}

// Review Focus #3: two hand-offs in a chain compose their scales, whichever
// order denoise and super-resolution run in.
void TestSyncHandOffChainComposesScales() {
    const std::string denoise_then_sr =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"dn\",\"model\":\"sr_half\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"dn\"},"
        "{\"from\":\"dn\",\"to\":\"sr\"},"
        "{\"from\":\"sr\",\"to\":\"od\"}]}";
    {
        FakeModelRegistry registry = BuildHandOffRegistry();
        const std::string validation_error = ValidationError(denoise_then_sr, registry);
        GRAPH_CHECK(validation_error.empty());
        if (!validation_error.empty()) return;
        GraphSpec spec = ParseGraphText(denoise_then_sr, "h.json");
        ValidateGraph(spec, registry);

        StageGraph graph;
        graph.Build(spec, registry, "/models", false);

        SyncExecutor executor;
        FrameReport report;
        std::string failure;
        try {
            report = executor.RunFrame(
                graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        GRAPH_CHECK(failure.empty());
        GRAPH_CHECK(report.error.empty());

        FakeStage* yolo = registry.last_stage("yolov8n");
        GRAPH_CHECK(yolo != NULL);
        if (yolo == NULL) return;
        GRAPH_CHECK(yolo->seen().size() == 1);
        if (yolo->seen().size() != 1) return;
        GRAPH_CHECK(NearAffine(yolo->seen()[0].origin.inv_align,
                              Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));
    }

    const std::string sr_then_denoise =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"dn\",\"model\":\"sr_half\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
        "{\"from\":\"sr\",\"to\":\"dn\"},"
        "{\"from\":\"dn\",\"to\":\"od\"}]}";
    {
        FakeModelRegistry registry = BuildHandOffRegistry();
        const std::string validation_error = ValidationError(sr_then_denoise, registry);
        GRAPH_CHECK(validation_error.empty());
        if (!validation_error.empty()) return;
        GraphSpec spec = ParseGraphText(sr_then_denoise, "h.json");
        ValidateGraph(spec, registry);

        StageGraph graph;
        graph.Build(spec, registry, "/models", false);

        SyncExecutor executor;
        FrameReport report;
        std::string failure;
        try {
            report = executor.RunFrame(
                graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        GRAPH_CHECK(failure.empty());
        GRAPH_CHECK(report.error.empty());

        FakeStage* yolo = registry.last_stage("yolov8n");
        GRAPH_CHECK(yolo != NULL);
        if (yolo == NULL) return;
        GRAPH_CHECK(yolo->seen().size() == 1);
        if (yolo->seen().size() != 1) return;
        GRAPH_CHECK(NearAffine(yolo->seen()[0].origin.inv_align,
                              Affine(1.f, 0.f, 0.f, 0.f, 1.f, 0.f)));

        const ImageData* dn_image = dynamic_cast<const ImageData*>(
            report.node_results["dn"].data.get());
        GRAPH_CHECK(dn_image != NULL);
        if (dn_image == NULL) return;
        GRAPH_CHECK(yolo->seen()[0].image.data == dn_image->image.data);
        GRAPH_CHECK(yolo->seen()[0].image.cols == 640 &&
                    yolo->seen()[0].image.rows == 480);
    }
}

// Review Focus #2: an unusable hand-off image is reported on the CONSUMER,
// exactly like today's "producer produced nothing" case leaves the consumer
// in roi_results with an empty vector rather than dropping it outright.
void TestSyncHandOffErrorIsReportedOnTheConsumer() {
    {
        FakeModelRegistry registry = BuildHandOffRegistry();
        GraphSpec spec = ParseGraphText(HandOffJson("sr_empty"), "h.json");
        ValidateGraph(spec, registry);

        StageGraph graph;
        graph.Build(spec, registry, "/models", false);

        SyncExecutor executor;
        FrameReport report;
        std::string failure;
        try {
            report = executor.RunFrame(
                graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        GRAPH_CHECK(failure.empty());
        GRAPH_CHECK(report.error ==
                    "node \"od\": \"sr\" handed off an empty image");

        FakeStage* yolo = registry.last_stage("yolov8n");
        GRAPH_CHECK(yolo != NULL);
        if (yolo != NULL) GRAPH_CHECK(yolo->seen().empty());

        GRAPH_CHECK(report.roi_results.count("od") == 1);
        GRAPH_CHECK(report.roi_results.count("cls") == 1);
        GRAPH_CHECK(report.roi_results["od"].empty());
        GRAPH_CHECK(report.roi_results["cls"].empty());

        // Compare against the existing "producer produced nothing" shape:
        // a producer that throws outright also leaves "od" and "cls" behind
        // in roi_results, both empty - hand-off failure and outright
        // producer failure agree on what the consumer's report looks like.
        FakeModelRegistry failed_registry = BuildHandOffRegistry();
        failed_registry.SetFailure("sr_x2", "sr exploded");
        GraphSpec failed_spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
        ValidateGraph(failed_spec, failed_registry);
        StageGraph failed_graph;
        failed_graph.Build(failed_spec, failed_registry, "/models", false);
        SyncExecutor failed_executor;
        FrameReport failed_report;
        std::string failed_failure;
        try {
            failed_report = failed_executor.RunFrame(
                failed_graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
        } catch (const std::exception& error) {
            failed_failure = error.what();
        }
        GRAPH_CHECK(failed_failure.empty());
        GRAPH_CHECK(report.roi_results.size() == failed_report.roi_results.size());
        GRAPH_CHECK(report.roi_results["od"].size() ==
                    failed_report.roi_results["od"].size());
        GRAPH_CHECK(report.roi_results["cls"].size() ==
                    failed_report.roi_results["cls"].size());
    }

    {
        FakeModelRegistry registry = BuildHandOffRegistry();
        GraphSpec spec = ParseGraphText(HandOffJson("sr_gray"), "h.json");
        ValidateGraph(spec, registry);

        StageGraph graph;
        graph.Build(spec, registry, "/models", false);

        SyncExecutor executor;
        FrameReport report;
        std::string failure;
        try {
            report = executor.RunFrame(
                graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
        } catch (const std::exception& error) {
            failure = error.what();
        }
        GRAPH_CHECK(failure.empty());
        GRAPH_CHECK(report.error ==
                    "node \"od\": \"sr\" handed off a non-BGR image");

        // The same consumer-side shape as the sr_empty case above.
        FakeStage* yolo = registry.last_stage("yolov8n");
        GRAPH_CHECK(yolo != NULL);
        if (yolo != NULL) GRAPH_CHECK(yolo->seen().empty());

        GRAPH_CHECK(report.roi_results.count("od") == 1);
        GRAPH_CHECK(report.roi_results.count("cls") == 1);
        GRAPH_CHECK(report.roi_results["od"].empty());
        GRAPH_CHECK(report.roi_results["cls"].empty());
    }
}

void TestSyncHandOffProducerFailureSkipsTheConsumer() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    registry.SetFailure("sr_x2", "sr exploded");

    GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    FrameReport report;
    std::string failure;
    try {
        report = executor.RunFrame(
            graph, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(0)), 0);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(report.error == "node \"sr\": sr exploded");

    FakeStage* yolo = registry.last_stage("yolov8n");
    GRAPH_CHECK(yolo != NULL);
    if (yolo != NULL) GRAPH_CHECK(yolo->seen().empty());
}

// Review Focus #1: a graph with no hand-off must stay byte-identical - the
// od stage still receives the identity view over the source frame itself.
void TestSyncWithoutHandOffIsUnchanged() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);

    StageGraph graph;
    graph.Build(spec, registry, "/models", false);

    SyncExecutor executor;
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport report;
    std::string failure;
    try {
        report = executor.RunFrame(graph, frame, 0);
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(report.error.empty());

    FakeStage* detector = registry.last_stage("yolov8n");
    GRAPH_CHECK(detector != NULL);
    if (detector == NULL) return;
    GRAPH_CHECK(detector->seen().size() == 1);
    if (detector->seen().size() != 1) return;

    const StageInput& input = detector->seen()[0];
    GRAPH_CHECK(IsIdentityAffine(input.origin.inv_align));
    GRAPH_CHECK(input.image.data == frame.data);
}

// =============================================================================
// Payload hand-off, Task 4 - async executor. The same hand-off as sync, frame
// by frame, with frames in flight and completions arriving out of order.
// =============================================================================

namespace {

/// One frame of HandOffJson(sr_model), once through SyncExecutor and once
/// through AsyncExecutor(Frames(2)), each on a fresh graph and registry.
/// A non-empty `sr_failure` makes the SR model fail on both.
struct HandOffPair {
    FrameReport sync;
    FrameReport async;
    std::vector<StageInput> sync_seen;   ///< yolov8n's inputs under sync
    std::vector<StageInput> async_seen;  ///< yolov8n's inputs under async
    std::string failure;
};

HandOffPair RunHandOffBothWays(const std::string& sr_model,
                               const std::string& sr_failure) {
    HandOffPair out;
    try {
        GraphSpec spec = ParseGraphText(HandOffJson(sr_model), "h.json");
        const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));

        FakeModelRegistry sync_registry = BuildHandOffRegistry();
        if (!sr_failure.empty()) sync_registry.SetFailure(sr_model, sr_failure);
        ValidateGraph(spec, sync_registry);
        StageGraph sync_graph;
        sync_graph.Build(spec, sync_registry, "/models", false);
        SyncExecutor sync_executor;
        out.sync = sync_executor.RunFrame(sync_graph, frame, 0);
        FakeStage* sync_yolo = sync_registry.last_stage("yolov8n");
        if (sync_yolo != NULL) out.sync_seen = sync_yolo->seen();

        FakeModelRegistry async_registry = BuildHandOffRegistry();
        if (!sr_failure.empty()) async_registry.SetFailure(sr_model, sr_failure);
        StageGraph async_graph;
        async_graph.Build(spec, async_registry, "/models", false);
        AsyncExecutor async_executor(Frames(2));
        out.async = async_executor.RunFrame(async_graph, frame, 0);
        FakeStage* async_yolo = async_registry.last_stage("yolov8n");
        if (async_yolo != NULL) out.async_seen = async_yolo->seen();
    } catch (const std::exception& error) {
        out.failure = error.what();
    }
    return out;
}

/// PeopleAt's moving scene in the coordinates of a x2 SR image.
std::shared_ptr<BoxesData> SrPeopleAt(bool left, bool right) {
    std::shared_ptr<BoxesData> boxes = PeopleAt(left, right);
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        cv::Rect2f& box = boxes->items[i].box;
        box = cv::Rect2f(box.x * 2.f, box.y * 2.f, box.width * 2.f, box.height * 2.f);
    }
    return boxes;
}

/// A 1280x960 SR image whose pixel (0,0) channel 0 carries `key`, so the
/// detector's per-frame script (keyed on its own input) can tell frames apart.
std::shared_ptr<ImageData> SrImageWithKey(int key) {
    std::shared_ptr<ImageData> data(new ImageData());
    data->image = cv::Mat(960, 1280, CV_8UC3, cv::Scalar(10, 20, 30));
    data->image.at<cv::Vec3b>(0, 0)[0] = static_cast<unsigned char>(key);
    return data;
}

/// HandOffJson("sr_x2") with od tracked.
std::string TrackedHandOffJson() {
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\","
           "\"track\":{\"algo\":\"iou\",\"max_age\":0}},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// The hand-off registry with a moving scene behind the SR stage: source
/// key k -> SR image carrying k -> the detector's scene k, in SR space.
FakeModelRegistry BuildMovingHandOffRegistry() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    for (int k = 0; k < 4; ++k) registry.SetScriptByFirstPixel("sr_x2", k, SrImageWithKey(k));
    registry.SetScriptByFirstPixel("yolov8n", 0, SrPeopleAt(true, false));
    registry.SetScriptByFirstPixel("yolov8n", 1, SrPeopleAt(true, true));
    registry.SetScriptByFirstPixel("yolov8n", 2, SrPeopleAt(false, true));
    registry.SetScriptByFirstPixel("yolov8n", 3, SrPeopleAt(false, false));
    return registry;
}

/// Frame 2's SR output is an empty image; every other frame gets the x2 image.
FakeModelRegistry BuildMidStreamHandOffErrorRegistry() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    registry.SetScriptByFirstPixel("sr_x2", 2, std::shared_ptr<ImageData>(new ImageData()));
    return registry;
}

/// od's boxes in `report`, or none.
std::vector<cv::Rect2f> OdBoxes(const FrameReport& report) {
    std::vector<cv::Rect2f> out;
    std::map<std::string, StageResult>::const_iterator it = report.node_results.find("od");
    if (it == report.node_results.end()) return out;
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(it->second.data.get());
    if (boxes == NULL) return out;
    for (std::size_t i = 0; i < boxes->items.size(); ++i) out.push_back(boxes->items[i].box);
    return out;
}

}  // namespace

void TestAsyncHandOffMatchesSync() {
    struct Case {
        const char* model;
        const char* failure;
    };
    const Case cases[] = {{"sr_x2", ""}, {"sr_empty", ""}, {"sr_gray", ""},
                          {"sr_x2", "sr exploded"}};
    for (std::size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        const HandOffPair run = RunHandOffBothWays(cases[c].model, cases[c].failure);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(run.sync == run.async);
        GRAPH_CHECK(run.sync_seen.size() == run.async_seen.size());
        if (run.sync_seen.size() != run.async_seen.size()) continue;
        for (std::size_t i = 0; i < run.sync_seen.size(); ++i) {
            GRAPH_CHECK(run.sync_seen[i].image.size() == run.async_seen[i].image.size());
            GRAPH_CHECK(run.sync_seen[i].origin.from_roi == run.async_seen[i].origin.from_roi);
            GRAPH_CHECK(NearAffine(run.sync_seen[i].origin.inv_align,
                                  run.async_seen[i].origin.inv_align));
        }
    }

    // The comparison is not vacuous: sync really hands od the x2 image, and
    // really reports each unusable hand-off on the consumer.
    const HandOffPair x2 = RunHandOffBothWays("sr_x2", "");
    GRAPH_CHECK(x2.sync.error.empty());
    GRAPH_CHECK(x2.async_seen.size() == 1);
    if (x2.async_seen.size() == 1) {
        GRAPH_CHECK(x2.async_seen[0].image.cols == 1280 && x2.async_seen[0].image.rows == 960);
        GRAPH_CHECK(NearAffine(x2.async_seen[0].origin.inv_align,
                              Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));
    }
    const HandOffPair empty = RunHandOffBothWays("sr_empty", "");
    GRAPH_CHECK(empty.sync.error == "node \"od\": \"sr\" handed off an empty image");
    GRAPH_CHECK(empty.async.error == "node \"od\": \"sr\" handed off an empty image");
    const HandOffPair gray = RunHandOffBothWays("sr_gray", "");
    GRAPH_CHECK(gray.sync.error == "node \"od\": \"sr\" handed off a non-BGR image");
    GRAPH_CHECK(gray.async.error == "node \"od\": \"sr\" handed off a non-BGR image");
    const HandOffPair failed = RunHandOffBothWays("sr_x2", "sr exploded");
    GRAPH_CHECK(failed.sync.error == "node \"sr\": sr exploded");
    GRAPH_CHECK(failed.async.error == "node \"sr\": sr exploded");
}

void TestAsyncHandOffMatchesSyncAcrossFramesUnderShuffledDelivery() {
    try {
        const std::vector<int> keys = SceneKeys(8);
        // The moving scene (as in the tracked test below), so frames can be
        // told apart and a report delivered to the wrong frame is caught.
        const std::vector<FrameReport> expected =
            SyncReportsOn(HandOffJson("sr_x2"), BuildMovingHandOffRegistry(), keys);
        GRAPH_CHECK(expected.size() == 8);
        if (expected.size() != 8) return;
        GRAPH_CHECK(expected[0].error.empty());
        GRAPH_CHECK(expected[0].roi_results.count("cls") == 1);
        GRAPH_CHECK(!(expected[0] == expected[1]));

        GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
        FakeModelRegistry registry = BuildMovingHandOffRegistry();
        registry.SetShuffleSeed("sr_x2", 41);
        registry.SetShuffleSeed("yolov8n", 42);
        registry.SetShuffleSeed("resnet50", 43);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunPipelined(graph, keys, Frames(4), false);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(run.peak == 4);
        GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// Review Focus #2: only the frame whose hand-off is unusable reports it.
void TestAsyncHandOffErrorFrameMidStreamMatchesSync() {
    try {
        std::vector<int> keys;
        for (int i = 0; i < 5; ++i) keys.push_back(i);
        const std::vector<FrameReport> expected =
            SyncReportsOn(HandOffJson("sr_x2"), BuildMidStreamHandOffErrorRegistry(), keys);
        GRAPH_CHECK(expected.size() == 5);
        if (expected.size() != 5) return;
        GRAPH_CHECK(expected[2].error == "node \"od\": \"sr\" handed off an empty image");

        GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
        FakeModelRegistry registry = BuildMidStreamHandOffErrorRegistry();
        registry.SetDeliveryOrder("sr_x2", FakeModelRegistry::kReverse);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunPipelined(graph, keys, Frames(3), false);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
        GRAPH_CHECK(run.reports.size() == 5);
        if (run.reports.size() != 5) return;
        for (std::size_t i = 0; i < run.reports.size(); ++i) {
            if (i == 2) {
                GRAPH_CHECK(run.reports[i].error ==
                            "node \"od\": \"sr\" handed off an empty image");
            } else {
                GRAPH_CHECK(run.reports[i].error.empty());
            }
        }
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// Review Focus #4: a tracked detector fed by super-resolution, its
// completions reversed, assigns the same track ids as sync.
void TestAsyncTrackedDetectorAfterHandOffMatchesSync() {
    try {
        std::vector<int> keys;
        for (int i = 0; i < 4; ++i) keys.push_back(i);
        const std::vector<FrameReport> expected =
            SyncReportsOn(TrackedHandOffJson(), BuildMovingHandOffRegistry(), keys);
        GRAPH_CHECK(expected.size() == 4);
        if (expected.size() != 4) return;
        for (std::size_t i = 0; i < expected.size(); ++i) GRAPH_CHECK(expected[i].error.empty());
        // The scene really moves: frame 0 has the left person, frame 2 the
        // right one, both in SR coordinates.
        const std::vector<cv::Rect2f> first = OdBoxes(expected[0]);
        const std::vector<cv::Rect2f> third = OdBoxes(expected[2]);
        GRAPH_CHECK(first.size() == 1 && third.size() == 1);
        if (first.size() != 1 || third.size() != 1) return;
        GRAPH_CHECK(first[0] != third[0]);
        GRAPH_CHECK(first[0] == cv::Rect2f(20, 20, 80, 120));
        GRAPH_CHECK(third[0] == cv::Rect2f(600, 400, 80, 120));

        GraphSpec spec = ParseGraphText(TrackedHandOffJson(), "h.json");
        FakeModelRegistry registry = BuildMovingHandOffRegistry();
        registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunPipelined(graph, keys, Frames(4), false);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(run.peak == 4);
        GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// Fix round 1, F1: tracking replaces a detector's result with the tracked
// copy of its boxes. That copy must keep the origin the stage reported, or
// a tracked detector after super-resolution is drawn (and serialized) in
// source coordinates at twice its real position. Both executors.
void TestTrackedDetectorAfterHandOffKeepsItsOrigin() {
    try {
        GraphSpec spec = ParseGraphText(TrackedHandOffJson(), "h.json");
        const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));
        const cv::Matx23f half = Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f);

        FakeModelRegistry sync_registry = BuildHandOffRegistry();
        ValidateGraph(spec, sync_registry);
        StageGraph sync_graph;
        sync_graph.Build(spec, sync_registry, "/models", false);
        SyncExecutor sync_executor;
        FrameReport sync_report = sync_executor.RunFrame(sync_graph, frame, 0);

        FakeModelRegistry async_registry = BuildHandOffRegistry();
        StageGraph async_graph;
        async_graph.Build(spec, async_registry, "/models", false);
        AsyncExecutor async_executor(Frames(2));
        FrameReport async_report = async_executor.RunFrame(async_graph, frame, 0);

        const FrameReport* reports[] = {&sync_report, &async_report};
        for (std::size_t r = 0; r < 2; ++r) {
            const FrameReport& report = *reports[r];
            GRAPH_CHECK(report.error.empty());
            std::map<std::string, StageResult>::const_iterator od =
                report.node_results.find("od");
            GRAPH_CHECK(od != report.node_results.end());
            if (od == report.node_results.end()) continue;
            // The tracked branch really ran: the stored boxes carry track ids.
            const BoxesData* boxes = dynamic_cast<const BoxesData*>(od->second.data.get());
            GRAPH_CHECK(boxes != NULL && !boxes->items.empty());
            if (boxes == NULL || boxes->items.empty()) continue;
            GRAPH_CHECK(boxes->items[0].track_id >= 0);
            GRAPH_CHECK(!od->second.origin.from_roi);
            GRAPH_CHECK(NearAffine(od->second.origin.inv_align, half));
        }
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

namespace {

std::string RealDenoiseHandOffJson() {
    return "{\"version\":1,\"name\":\"real-denoise\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"sample.jpg\"},"
           "{\"id\":\"denoise\",\"model\":\"dncnn_color_blind\"},"
           "{\"id\":\"od\",\"model\":\"yolov5n\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"denoise\"},"
           "{\"from\":\"denoise\",\"to\":\"od\"}]}";
}

std::string RealSuperResolutionHandOffJson() {
    return "{\"version\":1,\"name\":\"real-sr\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"sample.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"realesrgan_x2\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"params\":{\"score_threshold\":0.35}},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"],\"pad\":0.05}}]}";
}

/// One real hand-off graph through both executors on the NPU. `to_source`
/// is what od's origin.inv_align must be: only a real TypedStage can show
/// that the origin the executor hands in comes back out on the result.
///
/// One StageGraph drives both executors (as in TestRealStageSyncAsyncParity):
/// these graphs have no tracker, and the M1 cannot register a second
/// realesrgan_x2 engine next to yolov8n and resnet50 ("Failed to register
/// memory cache"), so a second graph would not build.
void CheckRealHandOff(const char* test, const std::string& json,
                      const std::vector<std::string>& models,
                      const cv::Matx23f& to_source, bool expect_crops) {
    StaticModelRegistry registry;
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) {
        Skip(test, missing);
        return;
    }
    const cv::Mat image = cv::imread(ProjectRoot() + "/sample/img/sample_crowd.jpg");
    GRAPH_CHECK(!image.empty());
    if (image.empty()) return;
    std::vector<cv::Mat> frames;
    for (int k = 0; k < 4; ++k) {
        const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 8 * k, 0, 1, 0);
        cv::Mat moved;
        cv::warpAffine(image, moved, shift, image.size());
        frames.push_back(moved);
    }

    try {
        GraphSpec spec = ParseGraphText(json, "real.json");
        ValidateGraph(spec, registry);
        StageGraph graph;
        graph.Build(spec, registry, ModelDir(), true);

        SyncExecutor sync_executor;
        const FrameReport first = sync_executor.RunFrame(graph, image, 0);
        GRAPH_CHECK(first.error.empty());
        if (!first.error.empty()) {
            std::printf("      %s: sync error: %s\n", test, first.error.c_str());
            return;
        }
        std::map<std::string, StageResult>::const_iterator od =
            first.node_results.find("od");
        GRAPH_CHECK(od != first.node_results.end());
        if (od == first.node_results.end()) return;
        const BoxesData* boxes = dynamic_cast<const BoxesData*>(od->second.data.get());
        // Something was detected, so the comparisons below are not of two
        // empty reports.
        GRAPH_CHECK(boxes != NULL && !boxes->items.empty());
        GRAPH_CHECK(!od->second.origin.from_roi);
        GRAPH_CHECK(NearAffine(od->second.origin.inv_align, to_source));
        if (!NearAffine(od->second.origin.inv_align, to_source)) {
            const cv::Matx23f& m = od->second.origin.inv_align;
            std::printf("      %s: od inv_align = (%g, %g, %g, %g, %g, %g)\n", test,
                        m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2));
        }
        if (expect_crops) {
            GRAPH_CHECK(first.roi_results.count("cls") == 1);
            if (first.roi_results.count("cls") == 1) {
                GRAPH_CHECK(!first.roi_results.find("cls")->second.empty());
            }
        }

        // Self-determinism, so a parity failure below is attributable.
        const FrameReport second = sync_executor.RunFrame(graph, image, 0);
        GRAPH_CHECK(first == second);
        if (!(first == second)) return;

        AsyncExecutor single(Frames(4));
        const FrameReport actual = single.RunFrame(graph, image, 0);
        GRAPH_CHECK(actual.error.empty());
        GRAPH_CHECK(first == actual);

        std::vector<FrameReport> expected;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            expected.push_back(sync_executor.RunFrame(graph, frames[i], i));
        }
        AsyncOptions options;
        options.max_frames_in_flight = 4;
        AsyncExecutor executor(options);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            executor.Submit(graph, frames[i], i);
        }
        executor.Finish(graph);
        GRAPH_CHECK(executor.peak_frames_in_flight() > 1);
        for (std::size_t i = 0; i < frames.size(); ++i) {
            FrameReport report;
            GRAPH_CHECK(executor.TryNext(&report));
            GRAPH_CHECK(report.error.empty());
            GRAPH_CHECK(expected[i].error.empty());
            GRAPH_CHECK(expected[i] == report);
        }
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
        std::printf("      %s: %s\n", test, failure.c_str());
    }
}

}  // namespace

// Real stages, real hand-off: denoise (512x512 out, whatever the frame)
// and, when its file is present, x2 super-resolution. od runs on the
// handed-off image, its origin maps back to the source frame, and the
// pipelined executor matches sync frame by frame.
void TestRealStageHandOffMatchesSync() {
    const cv::Mat image = cv::imread(ProjectRoot() + "/sample/img/sample_crowd.jpg");
    GRAPH_CHECK(!image.empty());
    if (image.empty()) return;
    const float w = static_cast<float>(image.cols);
    const float h = static_cast<float>(image.rows);

    std::vector<std::string> denoise_models;
    denoise_models.push_back("dncnn_color_blind");
    denoise_models.push_back("yolov5n");
    CheckRealHandOff("TestRealStageHandOffMatchesSync[denoise]", RealDenoiseHandOffJson(),
                     denoise_models, Affine(w / 512.f, 0.f, 0.f, 0.f, h / 512.f, 0.f),
                     false);

    std::vector<std::string> sr_models;
    sr_models.push_back("realesrgan_x2");
    sr_models.push_back("yolov8n");
    sr_models.push_back("resnet50");
    CheckRealHandOff("TestRealStageHandOffMatchesSync[sr]", RealSuperResolutionHandOffJson(),
                     sr_models, Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f), true);
}

// =============================================================================
// Payload hand-off, final whole-branch review fixes: the renderer draws an
// image as the backdrop (I1), and async/sync regressions around hand-off
// fan-out, chains, warped crops, rendering and error precedence (M2).
// =============================================================================

namespace {

/// The hand-off registry plus a segmenter and a depth model, each producing
/// a map in SR (1280x960) pixels: every label is class 3; depth is 5 on the
/// left half and 0 on the right.
FakeModelRegistry BuildHandOffMapRegistry() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    std::shared_ptr<LabelMapData> labels(new LabelMapData());
    labels->labels = cv::Mat(960, 1280, CV_32S, cv::Scalar(3));
    registry.AddModel(
        MakeFullFrameInfo("bisenetv2", "semantic_segmentation", Shape::kLabelMap), labels);
    std::shared_ptr<DenseMapData> depth(new DenseMapData());
    depth->values = cv::Mat(960, 1280, CV_32F, cv::Scalar(0));
    depth->values(cv::Rect(0, 0, 640, 960)).setTo(cv::Scalar(5.f));
    registry.AddModel(
        MakeFullFrameInfo("fastdepth_1", "depth_estimation", Shape::kDenseMap), depth);
    return registry;
}

/// cam -> sr -> seg and sr -> depth: both maps run on the SR image.
std::string HandOffMapJson() {
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
           "{\"id\":\"seg\",\"model\":\"bisenetv2\"},"
           "{\"id\":\"depth\",\"model\":\"fastdepth_1\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"seg\"},"
           "{\"from\":\"sr\",\"to\":\"depth\"}]}";
}

/// `report` with only the node results named (an empty name is skipped).
FrameReport OnlyNodes(const FrameReport& report, const std::string& first,
                      const std::string& second = std::string()) {
    FrameReport out;
    out.frame_index = report.frame_index;
    const std::string ids[] = {first, second};
    for (std::size_t i = 0; i < 2; ++i) {
        if (ids[i].empty()) continue;
        std::map<std::string, StageResult>::const_iterator it =
            report.node_results.find(ids[i]);
        if (it != report.node_results.end()) out.node_results[ids[i]] = it->second;
    }
    return out;
}

/// 255 where any channel of `bgr` is non-zero, else 0.
cv::Mat AnyChannel(const cv::Mat& bgr) {
    std::vector<cv::Mat> channels;
    cv::split(bgr, channels);
    cv::Mat out = cv::max(channels[0], channels[1]);
    out = cv::max(out, channels[2]);
    return out != 0;
}

/// Pixels (not channels) where `a` and `b` differ.
int ChangedPixels(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    return cv::countNonZero(AnyChannel(diff));
}

/// The point `m` maps to `target`: m^-1 * target.
cv::Point2f Preimage(const cv::Matx23f& m, const cv::Point2f& target) {
    cv::Mat inverse;
    cv::invertAffineTransform(cv::Mat(m), inverse);
    const cv::Matx23f n(inverse);
    return cv::Point2f(n(0, 0) * target.x + n(0, 1) * target.y + n(0, 2),
                       n(1, 0) * target.x + n(1, 1) * target.y + n(1, 2));
}

bool NearPoint(const cv::Point2f& p, float x, float y) {
    return std::fabs(p.x - x) < 1e-3f && std::fabs(p.y - y) < 1e-3f;
}

/// The moving hand-off scene, plus a second detector (reversed scene) and
/// a classifier, all fed by sr.
FakeModelRegistry BuildFanOutHandOffRegistry() {
    FakeModelRegistry registry = BuildMovingHandOffRegistry();
    ModelInfo second = MakeFullFrameInfo("yolov5n", "object_detection", Shape::kBoxes);
    second.input_width = 640;
    second.input_height = 640;
    registry.AddModel(second, SrPeopleAt(true, true));
    registry.SetScriptByFirstPixel("yolov5n", 0, SrPeopleAt(false, true));
    registry.SetScriptByFirstPixel("yolov5n", 1, SrPeopleAt(false, false));
    registry.SetScriptByFirstPixel("yolov5n", 2, SrPeopleAt(true, false));
    registry.SetScriptByFirstPixel("yolov5n", 3, SrPeopleAt(true, true));
    return registry;
}

/// cam -> sr, then sr fans out on plain edges to od, od2 and cls; od's
/// people are cropped into "crop".
std::string FanOutHandOffJson() {
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},"
           "{\"id\":\"od2\",\"model\":\"yolov5n\"},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"},"
           "{\"id\":\"crop\",\"model\":\"resnet50\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"sr\",\"to\":\"od2\"},"
           "{\"from\":\"sr\",\"to\":\"cls\"},"
           "{\"from\":\"od\",\"to\":\"crop\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// A w x h BGR image whose pixel (0,0) channel 0 carries `key`.
std::shared_ptr<ImageData> KeyedImage(int w, int h, int key) {
    std::shared_ptr<ImageData> data(new ImageData());
    data->image = cv::Mat(h, w, CV_8UC3, cv::Scalar(10, 20, 30));
    data->image.at<cv::Vec3b>(0, 0)[0] = static_cast<unsigned char>(key);
    return data;
}

/// One person, 100x100 at (200 + 200k, 200) in the x4 image's pixels.
std::shared_ptr<BoxesData> ChainPerson(int k) {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(
        MakeItem(200.f + 200.f * static_cast<float>(k), 200, 100, 100, 0.9f, "person"));
    return boxes;
}

/// sr (x2, key k) -> sr4 (x2 again: 2560x1920, key k) -> od (ChainPerson(k)).
FakeModelRegistry BuildChainHandOffRegistry() {
    FakeModelRegistry registry = BuildMovingHandOffRegistry();
    registry.AddModel(ImageModel("sr_x2b"), KeyedImage(2560, 1920, 0));
    ModelInfo detector = MakeFullFrameInfo("yolov5n", "object_detection", Shape::kBoxes);
    detector.input_width = 640;
    detector.input_height = 640;
    registry.AddModel(detector, ChainPerson(0));
    for (int k = 0; k < 4; ++k) {
        registry.SetScriptByFirstPixel("sr_x2b", k, KeyedImage(2560, 1920, k));
        registry.SetScriptByFirstPixel("yolov5n", k, ChainPerson(k));
    }
    return registry;
}

std::string ChainHandOffJson() {
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
           "{\"id\":\"sr4\",\"model\":\"sr_x2b\"},"
           "{\"id\":\"od\",\"model\":\"yolov5n\"},"
           "{\"id\":\"cls\",\"model\":\"resnet50\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"sr4\"},"
           "{\"from\":\"sr4\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"cls\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// Frame 0 of the chain: od's origin is x4 down, and the crop's inv_align
/// folds its (200,200) corner into that scale.
void CheckChainFrameZero(const FrameReport& report) {
    GRAPH_CHECK(report.error.empty());
    std::map<std::string, StageResult>::const_iterator od = report.node_results.find("od");
    GRAPH_CHECK(od != report.node_results.end());
    if (od != report.node_results.end()) {
        GRAPH_CHECK(NearAffine(od->second.origin.inv_align,
                              Affine(0.25f, 0.f, 0.f, 0.f, 0.25f, 0.f)));
    }
    std::map<std::string, std::vector<StageResult> >::const_iterator cls =
        report.roi_results.find("cls");
    GRAPH_CHECK(cls != report.roi_results.end() && cls->second.size() == 1);
    if (cls == report.roi_results.end() || cls->second.size() != 1) return;
    const RoiRef& crop = cls->second[0].origin;
    GRAPH_CHECK(NearAffine(crop.inv_align, Affine(0.25f, 0.f, 50.f, 0.f, 0.25f, 50.f)));
    GRAPH_CHECK(NearRect(crop.src_box, cv::Rect2f(50, 50, 25, 25)));
    GRAPH_CHECK(crop.crop_size == cv::Size(100, 100));
}

/// cam -> sr (an empty image) -> od, and cam -> zz, whose stage fails.
/// `zz_first` puts zz before sr and od in node order, and so in the
/// topological order both executors resolve report.error in.
std::string HandOffErrorAndNodeErrorJson(bool zz_first) {
    const std::string zz = "{\"id\":\"zz\",\"model\":\"resnet50\"},";
    return "{\"version\":1,\"name\":\"h\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"}," +
           (zz_first ? zz : std::string()) +
           "{\"id\":\"sr\",\"model\":\"sr_empty\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"}" +
           (zz_first ? std::string() : "," + zz.substr(0, zz.size() - 1)) +
           "],\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"cam\",\"to\":\"zz\"}]}";
}

}  // namespace

// I1: a hand-off graph always reports its image producer's output, and the
// renderer copies an image opaquely - so the image must be the backdrop,
// with the label map and dense map that ran on it blended over it. Drawn
// the other way round, cam -> sr -> seg rendered as the bare SR image.
void TestRenderReportBlendsMapsOverAHandedOffImage() {
    try {
        const cv::Mat source(480, 640, CV_8UC3, cv::Scalar(200, 200, 200));
        FakeModelRegistry registry = BuildHandOffMapRegistry();
        GraphSpec spec = ParseGraphText(HandOffMapJson(), "h.json");
        ValidateGraph(spec, registry);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        SyncExecutor executor;
        const FrameReport report = executor.RunFrame(graph, source, 0);
        GRAPH_CHECK(report.error.empty());

        std::map<std::string, StageResult>::const_iterator sr = report.node_results.find("sr");
        std::map<std::string, StageResult>::const_iterator seg = report.node_results.find("seg");
        GRAPH_CHECK(sr != report.node_results.end() && seg != report.node_results.end());
        GRAPH_CHECK(report.node_results.count("depth") == 1);
        if (sr == report.node_results.end() || seg == report.node_results.end() ||
            report.node_results.count("depth") != 1) {
            return;
        }
        const ImageData* image = dynamic_cast<const ImageData*>(sr->second.data.get());
        GRAPH_CHECK(image != NULL);
        if (image == NULL) return;
        // seg really ran on the handed-off image.
        GRAPH_CHECK(NearAffine(seg->second.origin.inv_align,
                              Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));

        // The SR image alone is the backdrop, resized onto the source canvas.
        cv::Mat backdrop;
        cv::resize(image->image, backdrop, source.size(), 0, 0, cv::INTER_LINEAR);
        const cv::Mat image_only = RenderReport(source, OnlyNodes(report, "sr"));
        GRAPH_CHECK(cv::norm(image_only, backdrop, cv::NORM_INF) == 0);

        // sr -> seg: class 3's colour blended over that backdrop, exactly,
        // and so visible at every pixel.
        const cv::Vec3b colour = SEGMENTATION_COLORS[3];
        cv::Mat expected;
        cv::addWeighted(backdrop, 0.5,
                        cv::Mat(source.size(), CV_8UC3,
                                cv::Scalar(colour[0], colour[1], colour[2])),
                        0.5, 0, expected);
        const cv::Mat with_seg = RenderReport(source, OnlyNodes(report, "sr", "seg"));
        GRAPH_CHECK(cv::norm(with_seg, expected, cv::NORM_INF) == 0);
        GRAPH_CHECK(ChangedPixels(with_seg, image_only) == source.rows * source.cols);

        // sr -> depth as well: the depth colours blend over the segmented
        // backdrop, on both halves of the depth map.
        const cv::Mat full = RenderReport(source, report);
        const cv::Mat layered = RenderReport(with_seg, OnlyNodes(report, "depth"));
        GRAPH_CHECK(cv::norm(full, layered, cv::NORM_INF) == 0);
        const cv::Rect left(0, 0, 320, 480);
        const cv::Rect right(320, 0, 320, 480);
        GRAPH_CHECK(ChangedPixels(full(left), with_seg(left)) > 0);
        GRAPH_CHECK(ChangedPixels(full(right), with_seg(right)) > 0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// M2 (a): one image producer fanned out on plain edges to two detectors
// and a classifier, every stage delivering in reverse: each child runs on
// the SR image, and the pipelined executor matches sync frame by frame.
void TestAsyncHandOffFanOutMatchesSyncUnderReversedDelivery() {
    try {
        const std::vector<int> keys = SceneKeys(8);
        const std::vector<FrameReport> expected =
            SyncReportsOn(FanOutHandOffJson(), BuildFanOutHandOffRegistry(), keys);
        GRAPH_CHECK(expected.size() == 8);
        if (expected.size() != 8) return;
        for (std::size_t i = 0; i < expected.size(); ++i) GRAPH_CHECK(expected[i].error.empty());
        // Not vacuous: every child of sr maps back from the x2 image, the
        // scene moves, and od's people are cropped.
        const char* children[] = {"od", "od2", "cls"};
        for (std::size_t c = 0; c < 3; ++c) {
            std::map<std::string, StageResult>::const_iterator it =
                expected[0].node_results.find(children[c]);
            GRAPH_CHECK(it != expected[0].node_results.end());
            if (it == expected[0].node_results.end()) continue;
            GRAPH_CHECK(!it->second.origin.from_roi);
            GRAPH_CHECK(NearAffine(it->second.origin.inv_align,
                                  Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));
        }
        GRAPH_CHECK(!(expected[0] == expected[1]));
        GRAPH_CHECK(CropsIn(expected, "crop") > 0);

        GraphSpec spec = ParseGraphText(FanOutHandOffJson(), "h.json");
        FakeModelRegistry registry = BuildFanOutHandOffRegistry();
        const char* models[] = {"sr_x2", "yolov8n", "yolov5n", "resnet50"};
        for (std::size_t m = 0; m < 4; ++m) {
            registry.SetDeliveryOrder(models[m], FakeModelRegistry::kReverse);
        }
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunPipelined(graph, keys, Frames(4), false);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(run.peak == 4);
        GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// M2 (b): two hand-offs in a chain under the pipelined executor, reversed:
// od's origin is the composed x4 scale, and so is its crop's.
void TestAsyncHandOffChainComposesScalesLikeSync() {
    try {
        const std::vector<int> keys = SceneKeys(8);
        const std::vector<FrameReport> expected =
            SyncReportsOn(ChainHandOffJson(), BuildChainHandOffRegistry(), keys);
        GRAPH_CHECK(expected.size() == 8);
        if (expected.size() != 8) return;
        CheckChainFrameZero(expected[0]);
        GRAPH_CHECK(!(expected[0] == expected[1]));

        GraphSpec spec = ParseGraphText(ChainHandOffJson(), "h.json");
        FakeModelRegistry registry = BuildChainHandOffRegistry();
        const char* models[] = {"sr_x2", "sr_x2b", "yolov5n", "resnet50"};
        for (std::size_t m = 0; m < 4; ++m) {
            registry.SetDeliveryOrder(models[m], FakeModelRegistry::kReverse);
        }
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunPipelined(graph, keys, Frames(4), false);
        GRAPH_CHECK(run.failure.empty());
        GRAPH_CHECK(run.peak == 4);
        GRAPH_CHECK(CountMismatches(expected, run.reports) == 0);
        GRAPH_CHECK(!run.reports.empty());
        if (!run.reports.empty()) CheckChainFrameZero(run.reports[0]);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// M2 (c): an OBB crop and a face5 crop cut from a x2 view. A point the
// view-space crop sees at view (x, y) restores to source (x/2, y/2), through
// RestorePointWarped and RestorePoint alike, and src_box is the view window
// in source pixels. crop_size and the crop pixels stay the view's.
void TestWarpedCropsFromAScaledViewRestoreToTheSource() {
    cv::Mat big(960, 1280, CV_8UC3);
    cv::randu(big, cv::Scalar::all(0), cv::Scalar::all(255));
    FrameView view;
    view.image = big;
    view.to_source = Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f);
    FrameView identity;
    identity.image = big;

    struct Case {
        const char* name;
        Shape shape;
        const char* align;
    };
    const Case cases[] = {{"obb", Shape::kObBoxes, ""}, {"face5", Shape::kBoxes, "face5"}};
    for (std::size_t c = 0; c < 2; ++c) {
        BoxesData boxes(cases[c].shape);
        std::vector<cv::Point2f> probes;
        if (cases[c].shape == Shape::kObBoxes) {
            BoxItem item = MakeItem(600, 400, 160, 120, 0.9f, "o");
            item.angle = 0.5f;
            boxes.items.push_back(item);
            probes.push_back(cv::Point2f(690, 470));
            probes.push_back(cv::Point2f(620, 430));
        } else {
            BoxItem item = MakeItem(600, 400, 160, 200, 0.9f, "face");
            const float lx[] = {640, 720, 680, 644, 716};
            const float ly[] = {450, 450, 500, 550, 550};
            for (int k = 0; k < 5; ++k) {
                item.landmarks.push_back(Keypoint(lx[k], ly[k]));
                probes.push_back(cv::Point2f(lx[k], ly[k]));
            }
            boxes.items.push_back(item);
        }
        RoiSpec spec;
        spec.present = true;
        spec.align = cases[c].align;
        const std::vector<RoiCrop> scaled = RouteRois(boxes, view, spec, "od", NULL, NULL);
        const std::vector<RoiCrop> plain = RouteRois(boxes, identity, spec, "od", NULL, NULL);
        GRAPH_CHECK(scaled.size() == 1 && plain.size() == 1);
        if (scaled.size() != 1 || plain.size() != 1) continue;
        const RoiRef& ref = scaled[0].ref;
        GRAPH_CHECK(ref.from_roi);

        for (std::size_t p = 0; p < probes.size(); ++p) {
            // The crop-local point the view-space crop sees at probes[p].
            const cv::Point2f local = Preimage(plain[0].ref.inv_align, probes[p]);
            GRAPH_CHECK(NearPoint(RestorePointWarped(local, plain[0].ref),
                                  probes[p].x, probes[p].y));
            const cv::Point2f warped = RestorePointWarped(local, ref);
            GRAPH_CHECK(NearPoint(warped, probes[p].x * 0.5f, probes[p].y * 0.5f));
            const Keypoint restored = RestorePoint(Keypoint(local.x, local.y), ref);
            GRAPH_CHECK(NearPoint(cv::Point2f(restored.x, restored.y),
                                  probes[p].x * 0.5f, probes[p].y * 0.5f));
            if (!NearPoint(warped, probes[p].x * 0.5f, probes[p].y * 0.5f)) {
                std::printf("      %s: view (%g, %g) -> (%g, %g)\n", cases[c].name,
                            probes[p].x, probes[p].y, warped.x, warped.y);
            }
        }

        const cv::Rect2f& window = plain[0].ref.src_box;
        GRAPH_CHECK(NearRect(ref.src_box, cv::Rect2f(window.x * 0.5f, window.y * 0.5f,
                                                     window.width * 0.5f,
                                                     window.height * 0.5f)));
        GRAPH_CHECK(ref.src_box.x >= 0.f && ref.src_box.y >= 0.f &&
                    ref.src_box.x + ref.src_box.width <= 640.f &&
                    ref.src_box.y + ref.src_box.height <= 480.f);
        GRAPH_CHECK(ref.crop_size == plain[0].ref.crop_size);
        GRAPH_CHECK(scaled[0].image.size() == plain[0].image.size());
        if (scaled[0].image.size() == plain[0].image.size()) {
            GRAPH_CHECK(cv::norm(scaled[0].image, plain[0].image, cv::NORM_INF) == 0);
        }
    }
}

// M2 (d): a detector after x2 super-resolution reports its box in SR
// pixels; RenderReport draws it at the box's place in the source frame.
void TestRenderReportDrawsAHandOffChildsBoxAtSourceCoordinates() {
    try {
        FakeModelRegistry registry = BuildHandOffRegistry();
        GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");
        ValidateGraph(spec, registry);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        SyncExecutor executor;
        const cv::Mat source = cv::Mat::zeros(480, 640, CV_8UC3);
        const FrameReport report = executor.RunFrame(graph, source, 0);
        GRAPH_CHECK(report.error.empty());

        // od alone: the SR backdrop and cls's text would cover the pixels
        // checked below.
        const FrameReport od_only = OnlyNodes(report, "od");
        std::map<std::string, StageResult>::const_iterator od = od_only.node_results.find("od");
        GRAPH_CHECK(od != od_only.node_results.end());
        if (od == od_only.node_results.end()) return;
        GRAPH_CHECK(NearAffine(od->second.origin.inv_align,
                              Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f)));
        const cv::Mat drawn = AnyChannel(RenderReport(source, od_only));

        // od's first box is (100,100,80,120) in SR pixels, so (50,50,40,60)
        // in the source. Row 80 crosses its left and right edges only.
        const cv::Mat row = drawn.row(80);
        GRAPH_CHECK(cv::countNonZero(row.colRange(46, 55)) > 0);
        GRAPH_CHECK(cv::countNonZero(row.colRange(86, 95)) > 0);
        GRAPH_CHECK(cv::countNonZero(row.colRange(55, 86)) == 0);
        GRAPH_CHECK(cv::countNonZero(row.colRange(95, 640)) == 0);
        // Column 70 crosses its bottom edge at y = 110, and nothing below.
        const cv::Mat column = drawn.col(70);
        GRAPH_CHECK(cv::countNonZero(column.rowRange(106, 115)) > 0);
        GRAPH_CHECK(cv::countNonZero(column.rowRange(115, 480)) == 0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// M2 (e): one frame with a hand-off error on od and a stage error on zz.
// report.error holds one message - the later node's, in topological order -
// and both executors keep the same one, whichever of the two comes last.
void TestHandOffErrorAndANodeErrorLeaveTheSameSurvivor() {
    struct Case {
        bool zz_first;
        const char* survivor;
    };
    const Case cases[] = {{false, "node \"zz\": boom"},
                          {true, "node \"od\": \"sr\" handed off an empty image"}};
    for (std::size_t c = 0; c < 2; ++c) {
        try {
            const std::string json = HandOffErrorAndNodeErrorJson(cases[c].zz_first);
            GraphSpec spec = ParseGraphText(json, "h.json");
            const cv::Mat frame(480, 640, CV_8UC3, cv::Scalar::all(0));

            FakeModelRegistry sync_registry = BuildHandOffRegistry();
            sync_registry.SetFailure("resnet50", "boom");
            ValidateGraph(spec, sync_registry);
            StageGraph sync_graph;
            sync_graph.Build(spec, sync_registry, "/models", false);
            SyncExecutor sync_executor;
            const FrameReport sync_report = sync_executor.RunFrame(sync_graph, frame, 0);

            FakeModelRegistry async_registry = BuildHandOffRegistry();
            async_registry.SetFailure("resnet50", "boom");
            StageGraph async_graph;
            async_graph.Build(spec, async_registry, "/models", false);
            AsyncExecutor async_executor(Frames(2));
            const FrameReport async_report = async_executor.RunFrame(async_graph, frame, 0);

            GRAPH_CHECK(sync_report.error == cases[c].survivor);
            GRAPH_CHECK(async_report.error == cases[c].survivor);
            GRAPH_CHECK(sync_report == async_report);
            // Both errors happened: od ran as an empty consumer, and zz
            // left no result.
            const FrameReport* reports[] = {&sync_report, &async_report};
            for (std::size_t r = 0; r < 2; ++r) {
                std::map<std::string, std::vector<StageResult> >::const_iterator od =
                    reports[r]->roi_results.find("od");
                GRAPH_CHECK(od != reports[r]->roi_results.end() && od->second.empty());
                GRAPH_CHECK(reports[r]->node_results.count("zz") == 0);
            }
            if (sync_report.error != cases[c].survivor ||
                async_report.error != cases[c].survivor) {
                std::printf("      zz_first=%d: sync '%s', async '%s'\n",
                            static_cast<int>(cases[c].zz_first), sync_report.error.c_str(),
                            async_report.error.c_str());
            }
        } catch (const std::exception& error) {
            const std::string failure = error.what();
            GRAPH_CHECK(failure.empty());
        }
    }
}

// =============================================================================
// Spec C6: a consumer's first recorded error is kept.
// =============================================================================

namespace {

/// A detector stage whose poll() throws once, then holds its jobs for one
/// more poll(), then delivers. run() and every delivery return an empty
/// kBoxes payload, so a SyncExecutor over the same stage is an exact oracle.
/// The one-poll hold is what lets the NEXT frame's zero-crop run of this
/// node be harvested while the escaped poll() error is still pending.
class PollThrowsThenHoldsStage : public IStage {
 public:
    PollThrowsThenHoldsStage() : polls_(0) {}

    StageResult run(const StageInput& input) { return Payload(input); }

    void submit(const StageInput& input, StageCallback callback) {
        pending_.push_back(std::make_pair(input, callback));
    }

    void flush() { DeliverAll(); }

    void poll() {
        const int call = polls_++;
        if (call == 0) throw std::runtime_error("poll exploded");
        if (call == 1) return;
        DeliverAll();
    }

    Shape outputShape() const { return Shape::kBoxes; }
    InputContract inputContract() const { return InputContract::kFullFrame; }

 private:
    static StageResult Payload(const StageInput& input) {
        StageResult result;
        result.data = std::make_shared<BoxesData>(Shape::kBoxes);
        result.origin = input.origin;
        return result;
    }

    void DeliverAll() {
        std::vector<std::pair<StageInput, StageCallback> > due;
        due.swap(pending_);
        for (std::size_t i = 0; i < due.size(); ++i) {
            due[i].second(Payload(due[i].first), std::string());
        }
    }

    int polls_;
    std::vector<std::pair<StageInput, StageCallback> > pending_;
};

/// The hand-off registry with sr_x2 handing off a gray (non-BGR) image
/// for source key 2, and its usual x2 image otherwise.
FakeModelRegistry BuildGrayOnKeyTwoHandOffRegistry() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    std::shared_ptr<ImageData> gray(new ImageData());
    gray->image = cv::Mat(960, 1280, CV_8UC1, cv::Scalar(5));
    registry.SetScriptByFirstPixel("sr_x2", 2, gray);
    return registry;
}

}  // namespace

// Frame 0 hands od a good image; od's poll() throws while frame 0's job is
// out. Frame 1 hands od a gray image, so od records a hand-off error and
// runs as a zero-crop consumer - harvested while the escaped poll() error
// is still pending. That harvest must keep the hand-off error (sync
// reports it), and the poll() error must not be dropped: it stays with od
// and lands in od's next harvest, frame 0's.
void TestAsyncHarvestKeepsAConsumersHandOffError() {
    const std::string handoff = "node \"od\": \"sr\" handed off a non-BGR image";
    try {
        GraphSpec spec = ParseGraphText(HandOffJson("sr_x2"), "h.json");

        FakeModelRegistry sync_registry = BuildGrayOnKeyTwoHandOffRegistry();
        ValidateGraph(spec, sync_registry);
        StageGraph sync_graph;
        sync_graph.Build(spec, sync_registry, "/models", false);
        const std::size_t sync_od = NodeIndexById(sync_graph, "od");
        GRAPH_CHECK(sync_od < sync_graph.nodes().size());
        if (sync_od >= sync_graph.nodes().size()) return;
        sync_graph.mutable_nodes()[sync_od].stage.reset(new PollThrowsThenHoldsStage());
        SyncExecutor sync_executor;
        const FrameReport sync0 = sync_executor.RunFrame(sync_graph, FrameWithKey(0), 0);
        const FrameReport sync1 = sync_executor.RunFrame(sync_graph, FrameWithKey(2), 1);
        GRAPH_CHECK(sync0.error.empty());
        GRAPH_CHECK(sync1.error == handoff);

        FakeModelRegistry async_registry = BuildGrayOnKeyTwoHandOffRegistry();
        StageGraph async_graph;
        async_graph.Build(spec, async_registry, "/models", false);
        const std::size_t od = NodeIndexById(async_graph, "od");
        GRAPH_CHECK(od < async_graph.nodes().size());
        if (od >= async_graph.nodes().size()) return;
        async_graph.mutable_nodes()[od].stage.reset(new PollThrowsThenHoldsStage());
        AsyncExecutor executor(Frames(2));
        FrameReport async0;
        FrameReport async1;
        executor.Submit(async_graph, FrameWithKey(0), 0);
        GRAPH_CHECK(!executor.TryNext(&async0));  // od's job is out; its poll threw
        executor.Submit(async_graph, FrameWithKey(2), 1);
        executor.Finish(async_graph);
        GRAPH_CHECK(executor.TryNext(&async0));
        GRAPH_CHECK(executor.TryNext(&async1));

        GRAPH_CHECK(async1.error == handoff);
        GRAPH_CHECK(async1 == sync1);
        if (async1.error != handoff) {
            std::printf("      frame 1: async '%s'\n", async1.error.c_str());
        }
        // Frame 0 carries the escaped poll() error; otherwise it is sync's.
        GRAPH_CHECK(async0.error == "node \"od\": poll exploded");
        FrameReport async0_without_error = async0;
        async0_without_error.error.clear();
        GRAPH_CHECK(async0_without_error == sync0);
    } catch (const std::exception& error) {
        const std::string failure = error.what();
        GRAPH_CHECK(failure.empty());
    }
}

// =============================================================================
// Task 5 (C2, C3): an instance's box and label; dense ROI results placed on
// rotated and aligned crops through their own inv_align.
// =============================================================================

// C4: instance segmentation on the NPU, through both executors. Since C1
// the shared comparator compares every instance's mask pixel for pixel,
// so this is real mask parity, not just box parity.
void TestRealStageInstanceSegSyncAsyncParity() {
    const char* kTest = "TestRealStageInstanceSegSyncAsyncParity";
    StaticModelRegistry registry;
    std::vector<std::string> models;
    models.push_back("yolov8n_seg");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) {
        Skip(kTest, missing);
        return;
    }
    const std::string image_path = ProjectRoot() + "/sample/img/sample_people.jpg";
    const cv::Mat frame = cv::imread(image_path);
    // NOT a skip: a tracked sample image that will not read is a broken
    // checkout (same reasoning as TestRealStageSyncAsyncParity).
    GRAPH_CHECK(!frame.empty());
    if (frame.empty()) {
        std::printf("      %s: could not read %s\n", kTest, image_path.c_str());
        return;
    }

    GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"name\":\"seg\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"sample.jpg\"},"
        "{\"id\":\"seg\",\"model\":\"yolov8n_seg\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"seg\"}]}",
        "seg.json");
    StageGraph graph;
    std::string build_error;
    try {
        ValidateGraph(spec, registry);
        graph.Build(spec, registry, ModelDir(), true);
    } catch (const std::exception& error) {
        build_error = error.what();
    }
    // NOT a skip: the .dxnn is present, so a build failure is a regression.
    GRAPH_CHECK(build_error.empty());
    if (!build_error.empty()) {
        std::printf("      %s: the .dxnn is present but the graph would not build: %s\n",
                    kTest, build_error.c_str());
        return;
    }

    SyncExecutor sync_executor;
    const FrameReport first = sync_executor.RunFrame(graph, frame, 3);
    GRAPH_CHECK(first.error.empty());
    std::map<std::string, StageResult>::const_iterator seg = first.node_results.find("seg");
    GRAPH_CHECK(seg != first.node_results.end());
    if (!first.error.empty() || seg == first.node_results.end()) return;
    const BoxesData* instances = dynamic_cast<const BoxesData*>(seg->second.data.get());
    GRAPH_CHECK(instances != NULL && instances->shape() == Shape::kInstances);
    if (instances == NULL) return;
    // Not vacuous: people found, each with a mask that has pixels on.
    GRAPH_CHECK(instances->items.size() >= 2);
    for (std::size_t i = 0; i < instances->items.size(); ++i) {
        const cv::Mat& mask = instances->items[i].mask;
        GRAPH_CHECK(!mask.empty() && cv::countNonZero(mask.reshape(1)) > 0);
    }

    const FrameReport second = sync_executor.RunFrame(graph, frame, 3);
    GRAPH_CHECK(first == second);
    if (!(first == second)) {
        std::printf("      two identical synchronous runs disagreed - the stage is not "
                    "reproducible, so the parity comparison below cannot mean anything\n");
        return;
    }
    AsyncExecutor async_executor(StageJobs(4));
    const FrameReport actual = async_executor.RunFrame(graph, frame, 3);
    GRAPH_CHECK(actual.error.empty());
    GRAPH_CHECK(first == actual);
    AsyncExecutor throttled(StageJobs(1));
    GRAPH_CHECK(first == throttled.RunFrame(graph, frame, 3));

    // The comparison sees the masks: one pixel of one mask flipped in a
    // copy of the async report makes it unequal.
    std::shared_ptr<BoxesData> tampered(new BoxesData(*instances));
    tampered->items[0].mask = tampered->items[0].mask.clone();
    tampered->items[0].mask.ptr<uchar>(0)[0] ^= 1;  // first byte, any type
    FrameReport changed = actual;
    changed.node_results["seg"].data = tampered;
    GRAPH_CHECK(!(first == changed));
    std::printf("      %s: %zu instances, masks equal sync == async\n", kTest,
                instances->items.size());
}

// C2: an instance is drawn with its box and label, like a detection - the
// outline on the box's edge restored through the origin, in the class
// colour, above the (fully "on") mask.
void TestRenderReportDrawsAnInstancesBoxOnItsRestoredEdge() {
    std::shared_ptr<BoxesData> instances(new BoxesData(Shape::kInstances));
    BoxItem item = MakeItem(10.f, 10.f, 30.f, 40.f, 0.8f, "person");
    item.class_id = 2;
    item.mask = cv::Mat(60, 50, CV_8UC1, cv::Scalar(255));  // the whole crop
    instances->items.push_back(item);

    RoiRef origin;
    origin.from_roi = true;
    origin.src_box = cv::Rect2f(20.f, 30.f, 50.f, 60.f);
    origin.inv_align = cv::Matx23f(1.f, 0.f, 20.f, 0.f, 1.f, 30.f);
    origin.crop_size = cv::Size(50, 60);
    origin.parent_node = "od";
    origin.parent_index = 0;
    origin.roi_index = 0;

    const cv::Mat source = cv::Mat::zeros(120, 160, CV_8UC3);
    const cv::Mat rendered =
        RenderReport(source, MakeRoiReport("seg", origin, instances));
    GRAPH_CHECK(rendered.size() == source.size());
    if (rendered.size() != source.size()) return;

    // Restored box: (30, 40, 30, 40). Its left edge is x = 30 on rows
    // 40..80, clear of the label (drawn above the box, rows < 40).
    const cv::Scalar colour = dxapp::getClassColor(2);
    const cv::Vec3b expected(static_cast<uchar>(colour[0]), static_cast<uchar>(colour[1]),
                             static_cast<uchar>(colour[2]));
    GRAPH_CHECK(rendered.at<cv::Vec3b>(60, 30) == expected);
    // Right edge: x = 59, the last column of a box that ends at x = 60.
    GRAPH_CHECK(rendered.at<cv::Vec3b>(60, 59) == expected);
    // Inside the box, off the outline: the mask's blend, not the outline.
    GRAPH_CHECK(rendered.at<cv::Vec3b>(60, 45) != expected);
    GRAPH_CHECK(rendered.at<cv::Vec3b>(60, 45) != cv::Vec3b(0, 0, 0));
    // The label: a filled class-colour background above the box.
    GRAPH_CHECK(HasNonZeroNear(rendered, 35, 26, 1));
    // Outside the crop and the label: untouched.
    GRAPH_CHECK(rendered.at<cv::Vec3b>(100, 120) == cv::Vec3b(0, 0, 0));
}

namespace {

/// The crop's pixel footprint in the source frame: the [-0.5, size - 0.5]
/// square of its pixel centres' cells, through ref.inv_align, as a closed
/// polygon.
std::vector<cv::Point2f> CropFootprint(const RoiRef& ref) {
    const std::vector<cv::Point2f> c = RestoreBoxCorners(
        cv::Rect2f(-0.5f, -0.5f, static_cast<float>(ref.crop_size.width),
                   static_cast<float>(ref.crop_size.height)),
        ref);
    // RestoreBoxCorners: (x0,y0), (x0,y1), (x1,y0), (x1,y1).
    std::vector<cv::Point2f> quad;
    quad.push_back(c[0]);
    quad.push_back(c[1]);
    quad.push_back(c[3]);
    quad.push_back(c[2]);
    return quad;
}

/// Pixels `rendered` changed against `baseline` must lie in `quad` (one
/// pixel of tolerance for the edge) and cover at least 95 % of it.
void CheckChangedPixelsFillFootprint(const char* name, const cv::Mat& rendered,
                                     const cv::Mat& baseline,
                                     const std::vector<cv::Point2f>& quad) {
    GRAPH_CHECK(rendered.size() == baseline.size());
    if (rendered.size() != baseline.size()) return;
    cv::Mat diff;
    cv::absdiff(rendered, baseline, diff);
    const cv::Mat changed = AnyChannel(diff);
    int outside = 0;
    int inside = 0;
    int inside_changed = 0;
    for (int y = 0; y < rendered.rows; ++y) {
        for (int x = 0; x < rendered.cols; ++x) {
            const double d = cv::pointPolygonTest(
                quad, cv::Point2f(static_cast<float>(x), static_cast<float>(y)), true);
            const bool on = changed.at<uchar>(y, x) != 0;
            if (on && d < -1.0) ++outside;
            if (d >= 0.0) {
                ++inside;
                if (on) ++inside_changed;
            }
        }
    }
    GRAPH_CHECK(inside > 500);  // a real footprint, not a sliver
    GRAPH_CHECK(outside == 0);
    GRAPH_CHECK(inside_changed >= 0.95 * inside);
    if (outside != 0 || inside_changed < 0.95 * inside) {
        std::printf("      %s: %d changed outside the footprint, %d of %d inside\n", name,
                    outside, inside_changed, inside);
    }
}

/// Every array-shaped result drawn on one warped crop: a white image, a
/// labelmap, a densemap and an instance mask, each of a size unlike the
/// crop's so the resize runs too. The canvas is pure green, a colour no
/// palette entry and no magma entry has, so every blended pixel changes.
void CheckDenseResultsFillWarpedCrop(const char* name, const RoiRef& ref,
                                     const cv::Size& canvas_size) {
    GRAPH_CHECK(ref.from_roi);
    // Not vacuous: this crop's alignment rotates, so no rectangle fits it.
    GRAPH_CHECK(ref.inv_align(0, 1) != 0.f || ref.inv_align(1, 0) != 0.f);
    const std::vector<cv::Point2f> quad = CropFootprint(ref);
    const cv::Mat canvas(canvas_size, CV_8UC3, cv::Scalar(0, 255, 0));
    const cv::Size odd(ref.crop_size.width / 2 + 3, ref.crop_size.height / 2 + 5);

    std::shared_ptr<ImageData> white(new ImageData());
    white->image = cv::Mat(odd, CV_8UC3, cv::Scalar::all(255));
    CheckChangedPixelsFillFootprint(
        (std::string(name) + " image").c_str(),
        RenderReport(canvas, MakeRoiReport("img", ref, white)), canvas, quad);

    std::shared_ptr<LabelMapData> labels(new LabelMapData());
    labels->labels = cv::Mat(odd, CV_32S, cv::Scalar(3));
    CheckChangedPixelsFillFootprint(
        (std::string(name) + " labelmap").c_str(),
        RenderReport(canvas, MakeRoiReport("seg", ref, labels)), canvas, quad);

    std::shared_ptr<DenseMapData> dense(new DenseMapData());
    dense->values = cv::Mat(odd, CV_32F);
    for (int y = 0; y < odd.height; ++y)
        for (int x = 0; x < odd.width; ++x) dense->values.at<float>(y, x) = float(x + y);
    CheckChangedPixelsFillFootprint(
        (std::string(name) + " densemap").c_str(),
        RenderReport(canvas, MakeRoiReport("depth", ref, dense)), canvas, quad);

    // The instance's own box and label (C2) are drawn with or without its
    // mask, so the mask's pixels are what a mask-less render lacks.
    std::shared_ptr<BoxesData> masked(new BoxesData(Shape::kInstances));
    BoxItem item = MakeItem(2.f, 2.f, 6.f, 6.f, 0.5f, "i");
    item.mask = cv::Mat(odd, CV_8UC1, cv::Scalar(255));
    masked->items.push_back(item);
    std::shared_ptr<BoxesData> bare(new BoxesData(Shape::kInstances));
    item.mask = cv::Mat();
    bare->items.push_back(item);
    CheckChangedPixelsFillFootprint(
        (std::string(name) + " instance mask").c_str(),
        RenderReport(canvas, MakeRoiReport("inst", ref, masked)),
        RenderReport(canvas, MakeRoiReport("inst", ref, bare)), quad);
}

}  // namespace

// C3: a real OBB crop from RouteRois (the fixture of TestRestoreBoxCorner-
// RoundTripsThroughRealObbCrop). Before, its dense results were pasted as
// the un-rotated window at src_box's corner.
void TestRenderReportWarpsDenseResultsOntoARealObbCrop() {
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    BoxItem obb = MakeItem(300.f, 200.f, 80.f, 60.f, 0.9f, "obj");
    obb.angle = 0.5f;
    boxes.items.push_back(obb);
    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;
    const std::vector<RoiCrop> crops =
        RouteRois(boxes, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(40)), spec, "od", NULL,
                  NULL);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    CheckDenseResultsFillWarpedCrop("obb", crops[0].ref, cv::Size(640, 480));
}

// C3: a real face5 crop of a tilted face. Its pixels are a side x side warp
// of the frame, not a window of it; before, they were pasted at src_box's
// corner.
void TestRenderReportWarpsDenseResultsOntoARealFace5Crop() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    BoxItem face = MakeItem(290.f, 190.f, 100.f, 120.f, 0.9f, "face");
    const float ox[] = {-20.f, 20.f, 0.f, -18.f, 18.f};
    const float oy[] = {-25.f, -25.f, 0.f, 25.f, 25.f};
    const float tilt = 0.4f;
    for (int k = 0; k < 5; ++k) {
        face.landmarks.push_back(
            Keypoint(340.f + ox[k] * std::cos(tilt) - oy[k] * std::sin(tilt),
                     250.f + ox[k] * std::sin(tilt) + oy[k] * std::cos(tilt)));
    }
    boxes.items.push_back(face);
    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;
    spec.align = "face5";
    const std::vector<RoiCrop> crops =
        RouteRois(boxes, cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(40)), spec, "od", NULL,
                  NULL);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].ref.crop_size == cv::Size(120, 120));
    CheckDenseResultsFillWarpedCrop("face5", crops[0].ref, cv::Size(640, 480));
}

// C3: an OBB crop cut from a 2x (handed-off) view, drawn on the source:
// inv_align carries the view's 0.5 scale on top of the rotation. Before,
// the crop's view-sized pixels were pasted at source scale.
void TestRenderReportWarpsDenseResultsOntoAnObbCropFromAScaledView() {
    FrameView view;
    view.image = cv::Mat(960, 1280, CV_8UC3, cv::Scalar::all(40));
    view.to_source = Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f);
    BoxesData boxes = MakeBoxes(Shape::kObBoxes);
    BoxItem obb = MakeItem(600.f, 400.f, 160.f, 120.f, 0.9f, "obj");
    obb.angle = 0.5f;
    boxes.items.push_back(obb);
    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;
    const std::vector<RoiCrop> crops = RouteRois(boxes, view, spec, "od", NULL, NULL);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    GRAPH_CHECK(crops[0].ref.crop_size == cv::Size(160, 120));
    CheckDenseResultsFillWarpedCrop("obb over a 2x view", crops[0].ref, cv::Size(640, 480));
}

// A rotated crop whose footprint runs off the canvas. WarpOntoCanvas clamps
// its region to the canvas and shifts the warp by the clamped region's
// corner: off the top-left the corner clamps to (0, 0), off the
// bottom-right the corner is inside the canvas and the far side is cut.
// Either way the part of the footprint on the canvas is coloured and
// nothing else changes. A footprint wholly off the canvas draws nothing.
void TestRenderReportWarpsARotatedCropThatCrossesTheCanvasEdge() {
    const cv::Size canvas_size(320, 240);
    struct EdgeCase {
        const char* name;
        float angle;
        float tx;
        float ty;
    };
    // A 100x80 crop rotated by `angle` about its (0, 0) corner, which lands
    // at (tx, ty) - on the canvas, so the instance's label (drawn above its
    // box, near that corner) stays off the footprint.
    //   angle 2.0 at (100, 20): x in [-14, 100], y in [-13, 111] - off the
    //   top and the left; the region's corner clamps to (0, 0).
    //   angle 0.5 at (280, 180): x in [242, 368], y in [180, 298] - off the
    //   right and the bottom; the corner is inside, the far side is cut.
    const EdgeCase cases[] = {
        {"rotated crop off the top-left", 2.0f, 100.f, 20.f},
        {"rotated crop off the bottom-right", 0.5f, 280.f, 180.f},
    };
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float c = std::cos(cases[i].angle);
        const float s = std::sin(cases[i].angle);
        RoiRef ref;
        ref.from_roi = true;
        ref.inv_align = cv::Matx23f(c, -s, cases[i].tx, s, c, cases[i].ty);
        ref.crop_size = cv::Size(100, 80);
        ref.src_box = RestoreBox(cv::Rect2f(0.f, 0.f, 100.f, 80.f), ref);
        ref.parent_node = "od";
        ref.parent_index = 0;
        ref.roi_index = 0;
        // Not vacuous: the footprint really crosses the canvas edge.
        const cv::Rect2f bounds = ref.src_box;
        const bool low = bounds.x < 0.f && bounds.y < 0.f;
        const bool high = bounds.x + bounds.width > canvas_size.width &&
                          bounds.y + bounds.height > canvas_size.height;
        GRAPH_CHECK(i == 0 ? low : high);
        CheckDenseResultsFillWarpedCrop(cases[i].name, ref, canvas_size);
    }

    // Wholly off the canvas: no crash, no pixel changed.
    RoiRef away;
    away.from_roi = true;
    away.inv_align = cv::Matx23f(std::cos(0.5f), -std::sin(0.5f), 900.f, std::sin(0.5f),
                                 std::cos(0.5f), 900.f);
    away.crop_size = cv::Size(100, 80);
    away.src_box = RestoreBox(cv::Rect2f(0.f, 0.f, 100.f, 80.f), away);
    const cv::Mat canvas(canvas_size, CV_8UC3, cv::Scalar(0, 255, 0));
    std::shared_ptr<ImageData> white(new ImageData());
    white->image = cv::Mat(40, 50, CV_8UC3, cv::Scalar::all(255));
    GRAPH_CHECK(cv::norm(RenderReport(canvas, MakeRoiReport("img", away, white)),
                         canvas, cv::NORM_INF) == 0);
}

// Review focus 4: a dense result on a plain crop from RouteRois keeps the
// old formula pixel for pixel - resized into src_box and blended there.
void TestRenderReportKeepsPlainCropDenseResultsBitExact() {
    BoxesData boxes = MakeBoxes(Shape::kBoxes);
    boxes.items.push_back(MakeItem(100.f, 80.f, 90.f, 70.f, 0.9f, "p"));
    RoiSpec spec;
    spec.present = true;
    spec.pad = 0.f;
    const cv::Mat canvas(240, 320, CV_8UC3, cv::Scalar(10, 120, 200));
    const std::vector<RoiCrop> crops = RouteRois(boxes, canvas, spec, "od", NULL, NULL);
    GRAPH_CHECK(crops.size() == 1);
    if (crops.size() != 1) return;
    const RoiRef& ref = crops[0].ref;
    const cv::Rect window(100, 80, 90, 70);
    GRAPH_CHECK(ref.crop_size == window.size());

    std::shared_ptr<LabelMapData> labels(new LabelMapData());
    labels->labels = cv::Mat(23, 31, CV_32S);
    for (int y = 0; y < 23; ++y)
        for (int x = 0; x < 31; ++x) labels->labels.at<int>(y, x) = (x / 4 + y / 5) % 7;
    cv::Mat colored(labels->labels.size(), CV_8UC3);
    for (int y = 0; y < 23; ++y)
        for (int x = 0; x < 31; ++x)
            colored.at<cv::Vec3b>(y, x) =
                SEGMENTATION_COLORS[labels->labels.at<int>(y, x) % SEGMENTATION_COLORS.size()];
    cv::Mat resized;
    cv::resize(colored, resized, window.size(), 0, 0, cv::INTER_NEAREST);
    cv::Mat expected = canvas.clone();
    cv::Mat blended;
    cv::addWeighted(expected(window), 0.5, resized, 0.5, 0, blended);
    blended.copyTo(expected(window));
    GRAPH_CHECK(cv::norm(RenderReport(canvas, MakeRoiReport("seg", ref, labels)),
                         expected, cv::NORM_INF) == 0);

    std::shared_ptr<ImageData> textured(new ImageData());
    textured->image = cv::Mat(41, 57, CV_8UC3);
    cv::randu(textured->image, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::resize(textured->image, resized, window.size(), 0, 0, cv::INTER_LINEAR);
    expected = canvas.clone();
    resized.copyTo(expected(window));
    GRAPH_CHECK(cv::norm(RenderReport(canvas, MakeRoiReport("sr", ref, textured)),
                         expected, cv::NORM_INF) == 0);

    // A plain crop from a 2x (handed-off) view: a pure scale, still resized
    // into its source rectangle (90x70 at (100,80)), not warped - a 0.5
    // warp samples every other pixel where the resize averages.
    FrameView view;
    view.image = cv::Mat(480, 640, CV_8UC3, cv::Scalar::all(40));
    view.to_source = Affine(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f);
    BoxesData view_boxes = MakeBoxes(Shape::kBoxes);
    view_boxes.items.push_back(MakeItem(200.f, 160.f, 180.f, 140.f, 0.9f, "p"));
    const std::vector<RoiCrop> scaled = RouteRois(view_boxes, view, spec, "od", NULL, NULL);
    GRAPH_CHECK(scaled.size() == 1);
    if (scaled.size() != 1) return;
    GRAPH_CHECK(scaled[0].ref.crop_size == cv::Size(180, 140));
    textured->image = cv::Mat(140, 180, CV_8UC3);
    cv::randu(textured->image, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::resize(textured->image, resized, window.size(), 0, 0, cv::INTER_LINEAR);
    expected = canvas.clone();
    resized.copyTo(expected(window));
    GRAPH_CHECK(cv::norm(RenderReport(canvas, MakeRoiReport("sr", scaled[0].ref, textured)),
                         expected, cv::NORM_INF) == 0);
}

// =====================================================================
// SP1 Task 5: the streamed --report and non-finite numbers (U-04, U-11)
// =====================================================================
namespace {

std::string ScratchPath(const std::string& name) {
    const char* dir = std::getenv("TMPDIR");
    // The pid keeps two concurrent runs (two build trees, a CI matrix) apart.
    return std::string(dir != NULL && *dir != '\0' ? dir : "/tmp") + "/graph_engine_test_" +
           std::to_string(static_cast<long long>(::getpid())) + "_" + name;
}

/// Silences OpenCV's own log lines for one scope and restores the previous
/// level on exit. Used only around a failure a test provokes on purpose (a
/// video writer that cannot open, an image that is not there): OpenCV
/// reports those on stderr by itself, which made a passing run look broken.
class ScopedOpenCvLogLevel {
 public:
    explicit ScopedOpenCvLogLevel(cv::utils::logging::LogLevel level)
        : previous_(cv::utils::logging::setLogLevel(level)) {}
    ~ScopedOpenCvLogLevel() { cv::utils::logging::setLogLevel(previous_); }
    ScopedOpenCvLogLevel(const ScopedOpenCvLogLevel&) = delete;
    ScopedOpenCvLogLevel& operator=(const ScopedOpenCvLogLevel&) = delete;

 private:
    cv::utils::logging::LogLevel previous_;
};

std::string ReadWholeFile(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

nlohmann::json WriterFrame(unsigned index, const std::string& error) {
    nlohmann::json frame = nlohmann::json::object();
    frame["index"] = index;
    frame["error"] = error;
    frame["skipped_out_of_bounds"] = 0;
    frame["nodes"] = nlohmann::json::object();
    frame["roi_nodes"] = nlohmann::json::object();
    return frame;
}

}  // namespace

/// Review Focus 4: the silence ends with its scope.
void TestScopedOpenCvLogLevelRestoresTheLevel() {
    const cv::utils::logging::LogLevel before = cv::utils::logging::getLogLevel();
    {
        ScopedOpenCvLogLevel quiet(cv::utils::logging::LOG_LEVEL_SILENT);
        GRAPH_CHECK(cv::utils::logging::getLogLevel() == cv::utils::logging::LOG_LEVEL_SILENT);
    }
    GRAPH_CHECK(cv::utils::logging::getLogLevel() == before);
}

/// Review Focus #1: the streamed file is byte for byte what the whole-file
/// writer produced - {"graph":..,"frames":[..]}.dump(2) + "\n".
void TestReportFileWriterWritesWhatTheWholeFileDumpWrote() {
    std::vector<nlohmann::json> frames;
    frames.push_back(WriterFrame(0, ""));
    nlohmann::json second = WriterFrame(1, "line one\nline \"two\" \xc3\xa9");
    second["nodes"]["od"]["payload"]["items"] = nlohmann::json::array();
    second["nodes"]["od"]["payload"]["values"] = {0.5, -1.25};
    second["roi_nodes"]["reid"] = nlohmann::json::array();
    frames.push_back(second);
    const char* names[] = {"person-reid", "", "odd \"name\"\n\xc3\xa9"};
    for (std::size_t n = 0; n < 3; ++n) {
        for (std::size_t count = 0; count <= frames.size(); ++count) {
            nlohmann::json whole = nlohmann::json::object();
            whole["graph"] = names[n];
            whole["frames"] = nlohmann::json::array();
            for (std::size_t i = 0; i < count; ++i) whole["frames"].push_back(frames[i]);
            const std::string expected = whole.dump(2) + "\n";
            const std::string path = ScratchPath("writer.json");
            std::string error;
            {
                cli::ReportFileWriter writer;
                GRAPH_CHECK(writer.Open(path, names[n], &error));
                for (std::size_t i = 0; i < count; ++i) {
                    GRAPH_CHECK(writer.Append(frames[i], &error));
                }
                GRAPH_CHECK(writer.frames_written() == count);
                GRAPH_CHECK(writer.Finish(&error));
                GRAPH_CHECK(!writer.is_open());
            }
            const std::string written = ReadWholeFile(path);
            GRAPH_CHECK(written == expected);
            if (written != expected) {
                std::printf("      name %u, %u frame(s):\n%s---\n%s", static_cast<unsigned>(n),
                            static_cast<unsigned>(count), written.c_str(), expected.c_str());
            }
            std::remove(path.c_str());
        }
    }
}

/// Review Focus #2: while the run goes on the file is exactly what a
/// SIGKILL would leave - a prefix of the final text ending on a frame's
/// closing brace - and a writer destroyed without Finish() still finalizes.
void TestReportFileWriterLeavesAFramePrefixAndFinishesOnDestruction() {
    const std::string path = ScratchPath("prefix.json");
    nlohmann::json whole = nlohmann::json::object();
    whole["graph"] = "g";
    whole["frames"] = nlohmann::json::array();
    whole["frames"].push_back(WriterFrame(0, ""));
    whole["frames"].push_back(WriterFrame(1, ""));
    const std::string expected = whole.dump(2) + "\n";
    {
        cli::ReportFileWriter writer;
        std::string error;
        GRAPH_CHECK(writer.Open(path, "g", &error));
        GRAPH_CHECK(writer.Append(whole["frames"][0], &error));
        GRAPH_CHECK(writer.Append(whole["frames"][1], &error));
        const std::string partial = ReadWholeFile(path);
        GRAPH_CHECK(!partial.empty() && partial[partial.size() - 1] == '}');
        GRAPH_CHECK(expected.compare(0, partial.size(), partial) == 0);
        GRAPH_CHECK(partial + "\n  ],\n  \"graph\": \"g\"\n}\n" == expected);
    }  // no Finish(): the destructor finalizes
    GRAPH_CHECK(ReadWholeFile(path) == expected);
    std::remove(path.c_str());
}

void TestReportFileWriterReportsAPathItCannotOpen() {
    cli::ReportFileWriter writer;
    std::string error;
    GRAPH_CHECK(!writer.Open("/definitely/not/a/dir/report.json", "g", &error));
    GRAPH_CHECK(error == "could not write /definitely/not/a/dir/report.json");
    GRAPH_CHECK(!writer.is_open());
    GRAPH_CHECK(writer.Finish(&error));  // nothing open: a no-op
}

/// U-11: NaN, +Inf and -Inf are three different strings, no report holds
/// null, and a strict parser (nlohmann's own) reads the text back.
void TestReportJsonSpellsNonFiniteNumbersDistinctly() {
    FrameReport report;
    std::shared_ptr<VectorData> vector(new VectorData());
    vector->values.push_back(std::numeric_limits<float>::quiet_NaN());
    vector->values.push_back(std::numeric_limits<float>::infinity());
    vector->values.push_back(-std::numeric_limits<float>::infinity());
    vector->values.push_back(0.5f);
    StageResult embedding;
    embedding.data = vector;
    report.node_results["emb"] = embedding;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(
        MakeItem(1, 2, 3, 4, std::numeric_limits<float>::quiet_NaN(), "person"));
    StageResult detector;
    detector.data = boxes;
    report.node_results["od"] = detector;

    const nlohmann::json out = consumer::ReportToJson(report, false);
    const nlohmann::json& values = out.at("nodes").at("emb").at("payload").at("values");
    GRAPH_CHECK(values.at(0) == "NaN");
    GRAPH_CHECK(values.at(1) == "Infinity");
    GRAPH_CHECK(values.at(2) == "-Infinity");
    GRAPH_CHECK(values.at(3) == 0.5);
    GRAPH_CHECK(out.at("nodes").at("od").at("payload").at("items").at(0).at("score") == "NaN");
    const std::string text = out.dump();
    GRAPH_CHECK(text.find("null") == std::string::npos);
    GRAPH_CHECK(nlohmann::json::parse(text) == out);
}

// =====================================================================
// SP1 Task 6: output kind, live sources, video and display (U-07)
// =====================================================================

/// Review Focus #4 and #5 (the pure half).
void TestCliOutputRules() {
    using cli::OutputKind;
    GRAPH_CHECK(cli::ClassifyOutput("") == OutputKind::kNone);
    GRAPH_CHECK(cli::ClassifyOutput("out.mp4") == OutputKind::kVideo);
    GRAPH_CHECK(cli::ClassifyOutput("OUT.AVI") == OutputKind::kVideo);
    GRAPH_CHECK(cli::ClassifyOutput("run.v1/out.mkv") == OutputKind::kVideo);
    GRAPH_CHECK(cli::ClassifyOutput("out.png") == OutputKind::kImagePerFrame);
    GRAPH_CHECK(cli::ClassifyOutput("out") == OutputKind::kImagePerFrame);
    GRAPH_CHECK(cli::ClassifyOutput("dir.mp4/out") == OutputKind::kImagePerFrame);
    GRAPH_CHECK(cli::ClassifyOutput("out.mp4.png") == OutputKind::kImagePerFrame);

    GRAPH_CHECK(cli::IsLiveUri("camera:0"));
    GRAPH_CHECK(cli::IsLiveUri("rtsp://host/stream"));
    GRAPH_CHECK(!cli::IsLiveUri("sample/img/sample_people.jpg"));
    GRAPH_CHECK(!cli::IsLiveUri("camera.mp4"));
    GRAPH_CHECK(!cli::IsLiveUri(""));

    const std::string refused = cli::OutputProblem("out.png", "camera:0", 0);
    GRAPH_CHECK(refused ==
                "--output out.png writes one image per frame, and \"camera:0\" is a "
                "live source that never ends: write a video instead (--output "
                "<stem>.mkv, .avi or .mp4) or stop after N frames (--frames N)");
    GRAPH_CHECK(!cli::OutputProblem("out", "rtsp://h/s", 0).empty());
    GRAPH_CHECK(!cli::OutputProblem("OUT.PNG", "camera:1", 0).empty());
    GRAPH_CHECK(cli::OutputProblem("out.png", "camera:0", 5).empty());
    GRAPH_CHECK(cli::OutputProblem("out.mp4", "rtsp://h/s", 0).empty());
    GRAPH_CHECK(cli::OutputProblem("out.png", "clip.mp4", 0).empty());
    GRAPH_CHECK(cli::OutputProblem("", "camera:0", 0).empty());

    GRAPH_CHECK(cli::VideoFourcc("a.avi") == cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    GRAPH_CHECK(cli::VideoFourcc("a.MP4") == cv::VideoWriter::fourcc('m', 'p', '4', 'v'));
    GRAPH_CHECK(cli::VideoFourcc("a.mkv") == cv::VideoWriter::fourcc('m', 'p', '4', 'v'));
    GRAPH_CHECK(cli::FourccText(cv::VideoWriter::fourcc('m', 'p', '4', 'v')) == "mp4v");

    GRAPH_CHECK(cli::VideoFps(25.0) == 25.0);
    GRAPH_CHECK(cli::VideoFps(240.0) == 240.0);
    GRAPH_CHECK(cli::VideoFps(0.0) == 30.0);        // images, many cameras
    GRAPH_CHECK(cli::VideoFps(0.5) == 30.0);
    GRAPH_CHECK(cli::VideoFps(90000.0) == 30.0);    // RTSP's 90 kHz clock
    GRAPH_CHECK(cli::VideoFps(std::numeric_limits<double>::quiet_NaN()) == 30.0);

    GRAPH_CHECK(cli::IsStopKey('q'));
    GRAPH_CHECK(cli::IsStopKey('Q'));
    GRAPH_CHECK(cli::IsStopKey(27));
    GRAPH_CHECK(cli::IsStopKey(0x100000 | 'q'));    // modifier bits above the low byte
    GRAPH_CHECK(!cli::IsStopKey(-1));               // waitKey: no key
    GRAPH_CHECK(!cli::IsStopKey('a'));
}

/// Review Focus #5: cv::VideoWriter drops a frame of another size silently;
/// VideoOutput resizes it, so every frame lands, at the first frame's size.
void TestVideoOutputKeepsEveryFrameAtTheFirstFramesSize() {
    const char* extensions[] = {".avi", ".mp4", ".mkv"};
    for (std::size_t e = 0; e < 3; ++e) {
        const std::string path = ScratchPath(std::string("video") + extensions[e]);
        {
            cli::VideoOutput video(path, 10.0);
            std::string error;
            for (int k = 0; k < 5; ++k) {
                const cv::Size size = k == 3 ? cv::Size(32, 24) : cv::Size(64, 48);
                GRAPH_CHECK(video.Write(cv::Mat(size, CV_8UC3, cv::Scalar(40 * k, 0, 0)),
                                        &error));
            }
            GRAPH_CHECK(video.frames_written() == 5);
        }  // the destructor finalizes the container
        cv::VideoCapture capture(path);
        GRAPH_CHECK(capture.isOpened());
        int count = 0;
        cv::Mat frame;
        while (capture.read(frame)) {
            ++count;
            GRAPH_CHECK(frame.cols == 64 && frame.rows == 48);
        }
        GRAPH_CHECK(count == 5);
        GRAPH_CHECK(std::fabs(capture.get(cv::CAP_PROP_FPS) - 10.0) < 0.5);
        if (count != 5) std::printf("      %s: read back %d frame(s)\n", extensions[e], count);
        std::remove(path.c_str());
    }
    {
        ScopedOpenCvLogLevel quiet(cv::utils::logging::LOG_LEVEL_SILENT);
        cli::VideoOutput nowhere("/definitely/not/a/dir/out.mp4", 30.0);
        std::string error;
        GRAPH_CHECK(!nowhere.Write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(0, 0, 0)), &error));
        GRAPH_CHECK(error.find("/definitely/not/a/dir/out.mp4") != std::string::npos);
        GRAPH_CHECK(error.find("mp4v") != std::string::npos);
    }
}

// =====================================================================
// SP1 fix wave (final review)
// =====================================================================

/// I3: a writer that will not open blames the codec only as one of two
/// causes, and offers ".avi" only when .avi is not what just failed. The
/// path is an existing directory, so the open fails with the directory in
/// place and the codec available.
void TestVideoOutputOpenFailureOffersAviOnlyWhenItIsNotAvi() {
    const char* extensions[] = {".avi", ".mp4", ".mkv"};
    for (std::size_t e = 0; e < 3; ++e) {
        ScopedOpenCvLogLevel quiet(cv::utils::logging::LOG_LEVEL_SILENT);
        const std::string path = ScratchPath(std::string("is_a_dir") + extensions[e]);
        ::mkdir(path.c_str(), 0700);
        std::string error;
        {
            cli::VideoOutput video(path, 30.0);
            GRAPH_CHECK(!video.Write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(0, 0, 0)), &error));
        }
        ::rmdir(path.c_str());
        GRAPH_CHECK(error.find("could not open a video writer for " + path) == 0);
        GRAPH_CHECK(error.find("cannot be created there") != std::string::npos);
        const bool offers_avi = error.find("try .avi") != std::string::npos;
        GRAPH_CHECK(offers_avi == (e != 0));
        if (offers_avi != (e != 0)) std::printf("      %s\n", error.c_str());
    }
}

/// I3: the directory check Main runs before PrepareGraph.
void TestCliOutputDirectoryIsCheckedUpFront() {
    GRAPH_CHECK(cli::ParentDirectory("out.png") == ".");
    GRAPH_CHECK(cli::ParentDirectory("/out.png") == "/");
    GRAPH_CHECK(cli::ParentDirectory("a/b/out.mp4") == "a/b");
    GRAPH_CHECK(cli::ParentDirectory("run.v1/out") == "run.v1");

    GRAPH_CHECK(cli::OutputDirectoryProblem("").empty());
    GRAPH_CHECK(cli::OutputDirectoryProblem("out.png").empty());  // the working directory
    GRAPH_CHECK(cli::OutputDirectoryProblem(ScratchPath("out.mkv")).empty());
    GRAPH_CHECK(cli::OutputDirectoryProblem("/definitely/not/a/dir/out.mp4") ==
                "could not write /definitely/not/a/dir/out.mp4: directory "
                "/definitely/not/a/dir does not exist");
    const std::string file = ScratchPath("plain_file");
    { std::ofstream touch(file.c_str()); }
    GRAPH_CHECK(cli::OutputDirectoryProblem(file + "/out.png") ==
                "could not write " + file + "/out.png: " + file + " is not a directory");
    std::remove(file.c_str());

    const int mjpg = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    const int mp4v = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    GRAPH_CHECK(cli::VideoOpenError("o.avi", mjpg) ==
                "could not open a video writer for o.avi (fourcc MJPG): the file cannot "
                "be created there, or this OpenCV build cannot encode it");
    GRAPH_CHECK(cli::VideoOpenError("o.MKV", mp4v) ==
                "could not open a video writer for o.MKV (fourcc mp4v): the file cannot "
                "be created there, or this OpenCV build cannot encode it; try .avi");
}

/// M4: a header that cannot be written leaves the writer closed, so neither
/// Append nor the destructor's Finish writes into it. /dev/full opens and
/// then fails every write with ENOSPC.
void TestReportFileWriterClosesWhenTheHeaderCannotBeWritten() {
#ifdef __linux__
    cli::ReportFileWriter writer;
    std::string error;
    GRAPH_CHECK(!writer.Open("/dev/full", "g", &error));
    GRAPH_CHECK(error == "could not write /dev/full");
    GRAPH_CHECK(!writer.is_open());
    error.clear();
    GRAPH_CHECK(!writer.Append(WriterFrame(0, ""), &error));
    GRAPH_CHECK(writer.frames_written() == 0);
    GRAPH_CHECK(writer.Finish(&error));  // nothing open: a no-op
#endif
}

/// M3: a stall timeout too large for a nanosecond duration waits, like any
/// other long timeout, instead of wrapping negative and stalling at once.
/// Every stage delivers from a worker 10-20 ms later, so the executor sits
/// in its wait loop, where the timeout is checked, on every job.
void TestAsyncHugeStallTimeoutDoesNotStallAtOnce() {
    const std::size_t huge[] = {std::numeric_limits<std::size_t>::max(),
                                static_cast<std::size_t>(10000000000000ULL)};  // 1e13 ms
    for (std::size_t h = 0; h < 2; ++h) {
        GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
        FakeModelRegistry registry = BuildCascadeRegistry();
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        ThreadEveryStage(graph, 77, 10, 20);
        AsyncOptions options;
        options.stall_timeout_ms = huge[h];
        AsyncExecutor executor(options);
        std::string failure;
        try {
            const FrameReport report =
                executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
            failure = report.error;
        } catch (const std::exception& error) {
            failure = error.what();
        }
        GRAPH_CHECK(failure.empty());
        if (!failure.empty()) std::printf("      %s\n", failure.c_str());
    }
}

// =====================================================================
// SP2 Task 1: one stream per source node in validation (U-01)
// =====================================================================
namespace {

/// cam1 -> sr -> od and cam2 -> od, plus `extra_edge` (",{...}" or "").
std::string TwoStreamHandOffJson(const std::string& extra_edge) {
    return "{\"version\":1,\"name\":\"h2\",\"nodes\":["
           "{\"id\":\"cam1\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
           "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
           "\"edges\":[{\"from\":\"cam1\",\"to\":\"sr\"},"
           "{\"from\":\"sr\",\"to\":\"od\"},"
           "{\"from\":\"cam2\",\"to\":\"od\"}" + extra_edge + "]}";
}

}  // namespace

void TestListStreamsHasOneStreamPerSource() {
    const std::vector<StreamSpec> streams =
        ListStreams(ParseGraphText(TwoStreamHandOffJson(""), "h2.json"));
    GRAPH_CHECK(streams.size() == 2);
    if (streams.size() != 2) return;
    std::vector<std::string> first;
    first.push_back("cam1");
    first.push_back("sr");
    first.push_back("od");
    std::vector<std::string> second;
    second.push_back("cam2");
    second.push_back("od");
    GRAPH_CHECK(streams[0].source == "cam1" && streams[0].nodes == first);
    GRAPH_CHECK(streams[1].source == "cam2" && streams[1].nodes == second);

    // One source: one stream holding every node, in declaration order.
    const std::vector<StreamSpec> one =
        ListStreams(ParseGraphText(HandOffJson("sr_x2"), "h.json"));
    GRAPH_CHECK(one.size() == 1);
    if (one.size() != 1) return;
    GRAPH_CHECK(one[0].source == "cam" && one[0].nodes.size() == 4);
    GRAPH_CHECK(one[0].nodes.size() == 4 && one[0].nodes[3] == "cls");
}

void TestValidateCountsFullFrameImagesPerStream() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    // od runs on sr's image in stream cam1 and on cam2's frame in stream
    // cam2: one image per stream. Rejected before SP2 ("more than one
    // image": sr is not a source).
    GRAPH_CHECK(ValidationError(TwoStreamHandOffJson(""), registry).empty());

    // cam1 -> od as well: stream cam1 now brings od two images.
    const std::string message = ValidationError(
        TwoStreamHandOffJson(",{\"from\":\"cam1\",\"to\":\"od\"}"), registry);
    const std::string expected =
        "ERROR [GRAPH_EDGE] node \"od\": receives full-frame input from more "
        "than one image in stream \"cam1\": \"sr\", \"cam1\"\n"
        "  -> keep one plain edge into this node, or split it into one node "
        "per input";
    GRAPH_CHECK(message == expected);
    if (message != expected) std::printf("      got: %s\n", message.c_str());
}

// =====================================================================
// SP2 Task 2: streams in the built graph and the sync executor
// =====================================================================
namespace {

/// camA and camB both feed one tracked detector, which crops people for
/// one reid: two streams through two shared nodes. Node indices: camA 0,
/// camB 1, od 2, reid 3.
std::string TwoStreamJson() {
    return "{\"version\":1,\"name\":\"two\",\"nodes\":["
           "{\"id\":\"camA\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"camB\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"track\":{\"algo\":\"iou\"}},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"camA\",\"to\":\"od\"},"
           "{\"from\":\"camB\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// Two independent detector branches from two sources, both cropping into
/// one reid. Node indices: camA 0, camB 1, odA 2, odB 3, reid 4.
std::string TwoStreamFanInJson(bool tracked) {
    const std::string track = tracked ? ",\"track\":{\"algo\":\"iou\"}" : "";
    return "{\"version\":1,\"name\":\"fanin2\",\"nodes\":["
           "{\"id\":\"camA\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"camB\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
           "{\"id\":\"odA\",\"model\":\"yolov8n\"" + track + "},"
           "{\"id\":\"odB\",\"model\":\"yolov8n\"" + track + "},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"camA\",\"to\":\"odA\"},"
           "{\"from\":\"camB\",\"to\":\"odB\"},"
           "{\"from\":\"odA\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}},"
           "{\"from\":\"odB\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

/// (stream index, first-pixel key) per frame, in submission order; key -1
/// is an empty frame.
typedef std::vector<std::pair<std::size_t, int> > StreamFrames;

StreamFrames FramesOf(std::size_t stream, const std::vector<int>& keys) {
    StreamFrames out;
    for (std::size_t i = 0; i < keys.size(); ++i) out.push_back(std::make_pair(stream, keys[i]));
    return out;
}

/// Round robin, as the CLI reads two streams: a0 b0 a1 b1 ..., then the
/// longer one's tail. `a` is stream 0, `b` stream 1.
StreamFrames Interleave(const std::vector<int>& a, const std::vector<int>& b) {
    StreamFrames out;
    for (std::size_t i = 0; i < a.size() || i < b.size(); ++i) {
        if (i < a.size()) out.push_back(std::make_pair(std::size_t(0), a[i]));
        if (i < b.size()) out.push_back(std::make_pair(std::size_t(1), b[i]));
    }
    return out;
}

cv::Mat StreamFrame(int key) { return key < 0 ? cv::Mat() : FrameWithKey(key); }

/// SyncExecutor over `frames` on a fresh graph, numbered per stream.
std::vector<FrameReport> SyncStreamReports(const std::string& json,
                                           FakeModelRegistry registry,
                                           const StreamFrames& frames) {
    GraphSpec spec = ParseGraphText(json, "x.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    std::vector<std::size_t> next(graph.streams().size(), 0);
    std::vector<FrameReport> out;
    for (std::size_t i = 0; i < frames.size(); ++i) {
        const std::size_t stream = frames[i].first;
        out.push_back(executor.RunFrame(graph, stream, StreamFrame(frames[i].second),
                                        next[stream]++));
    }
    return out;
}

/// The reports of stream `stream`, in order.
std::vector<FrameReport> OfStream(const std::vector<FrameReport>& reports,
                                  const std::string& stream) {
    std::vector<FrameReport> out;
    for (std::size_t i = 0; i < reports.size(); ++i) {
        if (reports[i].stream == stream) out.push_back(reports[i]);
    }
    return out;
}

}  // namespace

void TestStageGraphPlansEachStream() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    GRAPH_CHECK(graph.streams().size() == 2);
    if (graph.streams().size() != 2) return;
    const StreamPlan& a = graph.streams()[0];
    const StreamPlan& b = graph.streams()[1];
    GRAPH_CHECK(a.source == 0 && a.source_id == "camA");
    GRAPH_CHECK(b.source == 1 && b.source_id == "camB");
    GRAPH_CHECK(graph.source_indices().size() == 2);
    for (std::size_t k = 0; k < graph.streams().size() && k < graph.source_indices().size(); ++k) {
        GRAPH_CHECK(graph.streams()[k].source == graph.source_indices()[k]);
    }
    GRAPH_CHECK(a.member[0] == 1 && a.member[1] == 0 && a.member[2] == 1 && a.member[3] == 1);
    GRAPH_CHECK(b.member[0] == 0 && b.member[1] == 1 && b.member[2] == 1 && b.member[3] == 1);
    // Review Focus #2: od has two plain in-edges, and each stream brings one.
    GRAPH_CHECK(a.in_edges[2] == 1 && b.in_edges[2] == 1);
    GRAPH_CHECK(a.in_edges[3] == 1 && b.in_edges[3] == 1);
    std::vector<std::size_t> a_order;
    a_order.push_back(0);
    a_order.push_back(2);
    a_order.push_back(3);
    std::vector<std::size_t> b_order;
    b_order.push_back(1);
    b_order.push_back(2);
    b_order.push_back(3);
    GRAPH_CHECK(a.order == a_order && b.order == b_order);
    GRAPH_CHECK(graph.StreamIndex("camB") == 1);
    GRAPH_CHECK(graph.StreamIndex("od") == 2);   // not a source: streams().size()
    const NodeRuntime& od = graph.nodes()[2];
    GRAPH_CHECK(od.trackers.size() == 2);
    GRAPH_CHECK(od.trackers.size() == 2 && od.trackers[0] && od.trackers[1] &&
                od.trackers[0] != od.trackers[1]);
    GRAPH_CHECK(graph.nodes()[3].trackers.empty());

    GraphSpec one = ParseGraphText(CascadeJson(), "t.json");
    StageGraph single;
    single.Build(one, registry, "/models", false);
    GRAPH_CHECK(single.streams().size() == 1);
    GRAPH_CHECK(single.streams().size() == 1 &&
                single.streams()[0].order == single.topological_order());
    GRAPH_CHECK(single.streams().size() == 1 && single.source_indices().size() == 1 &&
                single.streams()[0].source == single.source_indices()[0]);
    GRAPH_CHECK(single.OnlyStream("test") == 0);
    GRAPH_CHECK(single.nodes()[1].trackers.size() == 1 && single.nodes()[1].trackers[0]);
}

void TestSyncStreamRunsOnlyItsOwnSubgraph() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    GraphSpec spec = ParseGraphText(TwoStreamFanInJson(false), "fanin2.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FrameReport a = executor.RunFrame(graph, 0, frame, 0);
    GRAPH_CHECK(a.error.empty() && a.stream == "camA" && a.frame_index == 0);
    GRAPH_CHECK(a.node_results.count("camA") == 1 && a.node_results.count("odA") == 1);
    GRAPH_CHECK(a.node_results.count("camB") == 0 && a.node_results.count("odB") == 0);
    GRAPH_CHECK(a.roi_results["reid"].size() == 2);
    for (std::size_t i = 0; i < a.roi_results["reid"].size(); ++i) {
        GRAPH_CHECK(a.roi_results["reid"][i].origin.parent_node == "odA");
    }

    FrameReport b = executor.RunFrame(graph, 1, frame, 0);
    GRAPH_CHECK(b.error.empty() && b.stream == "camB");
    GRAPH_CHECK(b.node_results.count("odB") == 1 && b.node_results.count("odA") == 0);
    GRAPH_CHECK(b.roi_results["reid"].size() == 2);
    for (std::size_t i = 0; i < b.roi_results["reid"].size(); ++i) {
        GRAPH_CHECK(b.roi_results["reid"][i].origin.parent_node == "odB");
    }
}

// Review Focus #3: interleaving camB's frames must not change camA's
// reports, and camB's first person must be track 0 - with one tracker per
// node it would get the id after camA's left person.
void TestSyncStreamsKeepTheirOwnTrackIds() {
    std::vector<int> a;
    a.push_back(0);  // left
    a.push_back(1);  // both
    a.push_back(2);  // right
    std::vector<int> b;
    b.push_back(2);  // right
    b.push_back(1);  // both
    b.push_back(0);  // left
    const std::vector<FrameReport> both =
        SyncStreamReports(TwoStreamJson(), BuildMovingCascadeRegistry(), Interleave(a, b));
    const std::vector<FrameReport> a_alone =
        SyncStreamReports(TwoStreamJson(), BuildMovingCascadeRegistry(), FramesOf(0, a));
    const std::vector<FrameReport> b_alone =
        SyncStreamReports(TwoStreamJson(), BuildMovingCascadeRegistry(), FramesOf(1, b));
    GRAPH_CHECK(both.size() == 6);
    GRAPH_CHECK(OfStream(both, "camA") == a_alone);
    GRAPH_CHECK(OfStream(both, "camB") == b_alone);
    const std::vector<FrameReport> b_reports = OfStream(both, "camB");
    for (std::size_t k = 0; k < b_reports.size(); ++k) {
        GRAPH_CHECK(b_reports[k].frame_index == k);
    }
    if (both.size() != 6) return;
    std::map<std::string, StageResult>::const_iterator od = both[1].node_results.find("od");
    const BoxesData* first_b = od == both[1].node_results.end()
        ? NULL : dynamic_cast<const BoxesData*>(od->second.data.get());
    GRAPH_CHECK(first_b != NULL && first_b->items.size() == 1);
    GRAPH_CHECK(first_b != NULL && !first_b->items.empty() && first_b->items[0].track_id == 0);
}

void TestExecutorsRefuseAnUnnamedStreamOnSeveralSources() {
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    std::string message;
    try {
        executor.RunFrame(graph, FrameWithKey(0), 0);
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    GRAPH_CHECK(message ==
                "SyncExecutor::RunFrame: the graph has 2 source nodes (camA, camB), so "
                "each frame must name its stream (an index into StageGraph::streams())");
    bool out_of_range = false;
    try {
        executor.RunFrame(graph, 2, FrameWithKey(0), 0);
    } catch (const std::out_of_range&) {
        out_of_range = true;
    }
    GRAPH_CHECK(out_of_range);
    AsyncExecutor pipelined(Frames(2));
    message.clear();
    try {
        pipelined.Submit(graph, FrameWithKey(0), 0);
    } catch (const std::invalid_argument& error) {
        message = error.what();
    }
    GRAPH_CHECK(message.find("AsyncExecutor::Submit: the graph has 2 source nodes") == 0);
}

void TestReportsNameTheirStream() {
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    SyncExecutor executor;
    GRAPH_CHECK(executor.RunFrame(graph, FrameWithKey(0), 0).stream == "cam");
    const FrameReport empty = executor.RunFrame(graph, cv::Mat(), 1);
    GRAPH_CHECK(empty.stream == "cam" && empty.error == "source produced an empty frame");
    AsyncExecutor pipelined(Frames(2));
    GRAPH_CHECK(pipelined.RunFrame(graph, FrameWithKey(0), 0).stream == "cam");

    FrameReport x;
    x.stream = "camA";
    FrameReport y = x;
    GRAPH_CHECK(x == y);
    y.stream = "camB";
    GRAPH_CHECK(!(x == y));
}

// =====================================================================
// SP2 Task 3: streams in the async executor
// =====================================================================
namespace {

/// Every frame of `frames` through one AsyncExecutor, numbered per stream.
/// `drain` takes the ready reports after every Submit, as the CLI does;
/// otherwise every frame is submitted first.
PipelinedRun RunStreamsPipelined(StageGraph& graph, const StreamFrames& frames,
                                 const AsyncOptions& options, bool drain) {
    PipelinedRun run;
    try {
        AsyncExecutor executor(options);
        std::vector<std::size_t> next(graph.streams().size(), 0);
        FrameReport report;
        for (std::size_t i = 0; i < frames.size(); ++i) {
            const std::size_t stream = frames[i].first;
            executor.Submit(graph, stream, StreamFrame(frames[i].second), next[stream]++);
            if (!drain) continue;
            while (executor.TryNext(&report)) run.reports.push_back(report);
        }
        executor.Finish(graph);
        while (executor.TryNext(&report)) run.reports.push_back(report);
        run.peak = executor.peak_frames_in_flight();
    } catch (const std::exception& error) {
        run.failure = error.what();
    }
    return run;
}

/// Position by position: the same stream, frame_index and content.
int CountStreamMismatches(const std::vector<FrameReport>& expected,
                          const std::vector<FrameReport>& actual) {
    int bad = expected.size() == actual.size() ? 0 : 1;
    const std::size_t n = expected.size() < actual.size() ? expected.size() : actual.size();
    for (std::size_t i = 0; i < n; ++i) {
        if (!(expected[i] == actual[i])) ++bad;
    }
    return bad;
}

std::vector<int> Reversed(std::vector<int> keys) {
    std::reverse(keys.begin(), keys.end());
    return keys;
}

}  // namespace

// Review Focus #1: the shared tracked detector's completions come back in
// reverse across both streams; each stream must still be tracked in its
// own order, and the reports must come out in submission order.
void TestAsyncStreamsMatchSyncUnderReversedDetectorDelivery() {
    const StreamFrames frames = Interleave(SceneKeys(8), Reversed(SceneKeys(8)));
    const std::vector<FrameReport> expected =
        SyncStreamReports(TwoStreamJson(), BuildMovingCascadeRegistry(), frames);
    GRAPH_CHECK(expected.size() == 16);
    GRAPH_CHECK(CropsIn(expected, "reid") > 0);

    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const PipelinedRun run = RunStreamsPipelined(graph, frames, Frames(6), false);
    GRAPH_CHECK(run.failure.empty());
    if (!run.failure.empty()) std::printf("      %s\n", run.failure.c_str());
    GRAPH_CHECK(run.peak == 6);
    GRAPH_CHECK(CountStreamMismatches(expected, run.reports) == 0);
    const std::vector<FrameReport> b = OfStream(run.reports, "camB");
    for (std::size_t k = 0; k < b.size(); ++k) GRAPH_CHECK(b[k].frame_index == k);
}

// Review Focus #1 and decision 9 ("one stream ending early"): two tracked
// detectors, each reached by one stream only. camB brings an empty frame and
// ends after three frames while camA carries on; neither stream's gate may
// wait for, or be moved by, the other's frames. Frames(4) sets a 2 s stall
// timeout, so a gate that waits forever fails instead of hanging.
void TestAsyncStreamGatesOrderEachStreamOnItsOwnWhenOneEndsEarly() {
    std::vector<int> b;
    b.push_back(2);   // right
    b.push_back(-1);  // an empty frame
    b.push_back(1);   // both - then camB ends
    const StreamFrames frames = Interleave(SceneKeys(10), b);
    const std::vector<FrameReport> expected =
        SyncStreamReports(TwoStreamFanInJson(true), BuildMovingCascadeRegistry(), frames);
    GRAPH_CHECK(expected.size() == 13);
    GRAPH_CHECK(expected.size() == 13 && expected[3].stream == "camB" &&
                expected[3].error == "source produced an empty frame");
    for (int drain = 0; drain < 2; ++drain) {
        GraphSpec spec = ParseGraphText(TwoStreamFanInJson(true), "fanin2.json");
        FakeModelRegistry registry = BuildMovingCascadeRegistry();
        registry.SetDeliveryOrder("yolov8n", FakeModelRegistry::kReverse);
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        const PipelinedRun run = RunStreamsPipelined(graph, frames, Frames(4), drain == 1);
        GRAPH_CHECK(run.failure.empty());
        if (!run.failure.empty()) std::printf("      drain %d: %s\n", drain, run.failure.c_str());
        GRAPH_CHECK(CountStreamMismatches(expected, run.reports) == 0);
    }
}

// Review Focus #2: od is fed by both sources, and a frame of either stream
// must release it with its one in-edge.
void TestAsyncSharedNodeRunsOncePerStreamFrame() {
    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(Frames(2));
    FrameReport a;
    FrameReport b;
    std::string failure;
    try {
        a = executor.RunFrame(graph, 0, FrameWithKey(1), 0);  // both people
        b = executor.RunFrame(graph, 1, FrameWithKey(2), 0);  // the right one
    } catch (const std::exception& error) {
        failure = error.what();
    }
    GRAPH_CHECK(failure.empty());
    GRAPH_CHECK(a.error.empty() && a.stream == "camA");
    if (!a.error.empty()) std::printf("      %s\n", a.error.c_str());
    GRAPH_CHECK(a.node_results.count("od") == 1 && a.node_results.count("camB") == 0);
    GRAPH_CHECK(a.roi_results["reid"].size() == 2);
    GRAPH_CHECK(b.error.empty() && b.stream == "camB");
    GRAPH_CHECK(b.roi_results["reid"].size() == 1);
    bool out_of_range = false;
    try {
        executor.Submit(graph, 2, FrameWithKey(0), 0);
    } catch (const std::out_of_range&) {
        out_of_range = true;
    }
    GRAPH_CHECK(out_of_range);
}

// Spec R2: one window for every stream. A cap per stream would admit six
// frames here.
void TestAsyncFrameWindowIsSharedByEveryStream() {
    const StreamFrames frames = Interleave(SceneKeys(6), Reversed(SceneKeys(6)));
    const std::vector<FrameReport> expected =
        SyncStreamReports(TwoStreamJson(), BuildMovingCascadeRegistry(), frames);
    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    FakeModelRegistry registry = BuildMovingCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const PipelinedRun run = RunStreamsPipelined(graph, frames, Frames(3), false);
    GRAPH_CHECK(run.failure.empty());
    GRAPH_CHECK(run.peak == 3);
    GRAPH_CHECK(CountStreamMismatches(expected, run.reports) == 0);
}

// =====================================================================
// SP2 Task 4: stream-aware JSON, --input binding, the round-robin reader
// =====================================================================
namespace {

/// An input with `frames` frames of first-pixel key `key`; counts getFrame calls.
class CountingInput : public IInputSource {
 public:
    CountingInput(int frames, int key) : left_(frames), key_(key), reads_(0) {}
    bool getFrame(cv::Mat& frame) {
        ++reads_;
        if (left_ <= 0) return false;
        --left_;
        frame = FrameWithKey(key_);
        return true;
    }
    bool isOpened() const { return true; }
    void release() {}
    InputType getType() const { return InputType::VIDEO; }
    int getWidth() const { return 640; }
    int getHeight() const { return 480; }
    double getFPS() const { return 10.0; }
    int getTotalFrames() const { return -1; }
    std::string getDescription() const { return "counting"; }
    bool isLiveSource() const { return false; }
    int reads() const { return reads_; }

 private:
    int left_;
    int key_;
    int reads_;
};

typedef std::vector<std::pair<std::size_t, std::size_t> > ReadOrder;

/// Two counting inputs of `a` and `b` frames; *first/*second observe them.
consumer::StreamReader TwoCounting(int a, int b, std::size_t limit, CountingInput** first,
                                   CountingInput** second) {
    std::vector<consumer::StreamInput> inputs(2);
    inputs[0].source = "camA";
    inputs[0].uri = "a.avi";
    inputs[0].input.reset(new CountingInput(a, 1));
    *first = static_cast<CountingInput*>(inputs[0].input.get());
    inputs[1].source = "camB";
    inputs[1].uri = "b.avi";
    inputs[1].input.reset(new CountingInput(b, 2));
    *second = static_cast<CountingInput*>(inputs[1].input.get());
    return consumer::StreamReader(std::move(inputs), limit);
}

ReadOrder ReadAll(consumer::StreamReader* reader) {
    ReadOrder order;
    std::size_t stream = 0;
    std::size_t index = 0;
    cv::Mat frame;
    while (reader->Next(&stream, &frame, &index)) order.push_back(std::make_pair(stream, index));
    return order;
}

ReadOrder Order(const int (*pairs)[2], std::size_t n) {
    ReadOrder out;
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(std::make_pair(std::size_t(pairs[i][0]), std::size_t(pairs[i][1])));
    }
    return out;
}

}  // namespace

// Review Focus #4.
void TestReportJsonNamesTheStreamOnlyForSeveralSources() {
    FrameReport report;
    report.frame_index = 3;
    report.stream = "camB";
    const nlohmann::json one = consumer::ReportToJson(report, false);
    GRAPH_CHECK(!one.contains("stream"));
    GRAPH_CHECK(one.dump() ==
                "{\"error\":\"\",\"index\":3,\"nodes\":{},\"roi_nodes\":{},"
                "\"skipped_out_of_bounds\":0}");
    const nlohmann::json several = consumer::ReportToJson(report, true);
    GRAPH_CHECK(several.contains("stream") && several.at("stream") == "camB");
    nlohmann::json without = several;
    without.erase("stream");
    GRAPH_CHECK(without == one);

    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph two;
    two.Build(ParseGraphText(TwoStreamJson(), "two.json"), registry, "/models", false);
    StageGraph single;
    single.Build(ParseGraphText(CascadeJson(), "t.json"), registry, "/models", false);
    GRAPH_CHECK(consumer::NamesStreams(two));
    GRAPH_CHECK(!consumer::NamesStreams(single));
}

// Review Focus #5.
void TestParseInputOverridesBindsOnlyKnownSourceIds() {
    const GraphSpec two = ParseGraphText(TwoStreamJson(), "two.json");
    std::map<std::string, std::string> bound;
    std::vector<std::string> inputs(1, "camB=b.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 1 && bound["camB"] == "b.mp4");

    inputs.assign(1, "clip.mp4");
    bound.clear();
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input \"clip.mp4\" does not say which source it replaces: graph "
                "\"two\" has 2 source nodes (camA, camB); write --input "
                "<source_id>=<uri>, e.g. --input camA=clip.mp4");
    // not a source id before '=': a misspelled id is named as one, and the
    // hint never suggests "camA=camX=a.mp4"
    inputs.assign(1, "camX=a.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input \"camX=a.mp4\" does not say which source it replaces: \"camX\" is "
                "not a source node of graph \"two\"; its source nodes: camA, camB");
    // a uri with '=' in it is still a bare uri, and the hint prefixes it
    inputs.assign(1, "rtsp://h/s?a=1");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input \"rtsp://h/s?a=1\" does not say which source it replaces: graph "
                "\"two\" has 2 source nodes (camA, camB); write --input "
                "<source_id>=<uri>, e.g. --input camA=rtsp://h/s?a=1");
    inputs.assign(1, "camA=a.mp4");
    inputs.push_back("camA=b.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input names source \"camA\" twice");
    GRAPH_CHECK(bound.empty());  // an error leaves nothing bound
    inputs.assign(1, "camB=");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input camB= has no uri after \"=\"");
    // the map is replaced, not added to
    bound.clear();
    bound["stale"] = "x.mp4";
    inputs.assign(1, "camA=a.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 1 && bound["camA"] == "a.mp4");
    bound["stale"] = "x.mp4";
    inputs.assign(1, "camA=");
    GRAPH_CHECK(!consumer::ParseInputOverrides(two, inputs, &bound).empty() && bound.empty());

    const GraphSpec one = ParseGraphText(CascadeJson(), "t.json");
    const char* const given[] = {"clip.mp4", "cam=clip.mp4", "rtsp://h/s?a=1"};
    const char* const uri[] = {"clip.mp4", "clip.mp4", "rtsp://h/s?a=1"};
    for (std::size_t k = 0; k < 3; ++k) {
        inputs.assign(1, given[k]);
        bound.clear();
        GRAPH_CHECK(consumer::ParseInputOverrides(one, inputs, &bound).empty());
        GRAPH_CHECK(bound.size() == 1 && bound["cam"] == uri[k]);
    }
    inputs.assign(1, "a.mp4");
    inputs.push_back("b.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(one, inputs, &bound) ==
                "--input names source \"cam\" twice");
    inputs.clear();
    bound.clear();
    GRAPH_CHECK(consumer::ParseInputOverrides(one, inputs, &bound).empty() && bound.empty());
}

void TestResolveStreamsGivesEverySourceItsUri() {
    const GraphSpec two = ParseGraphText(TwoStreamJson(), "two.json");
    std::map<std::string, std::string> overrides;
    overrides["camB"] = "b.mp4";
    std::vector<consumer::StreamUri> streams;
    GRAPH_CHECK(consumer::ResolveStreams(two, overrides, &streams).empty());
    GRAPH_CHECK(streams.size() == 2);
    if (streams.size() != 2) return;
    GRAPH_CHECK(streams[0].source == "camA" && streams[0].uri == "a.jpg");
    GRAPH_CHECK(streams[1].source == "camB" && streams[1].uri == "b.mp4");
    overrides["camX"] = "x.mp4";
    GRAPH_CHECK(consumer::ResolveStreams(two, overrides, &streams) ==
                "\"camX\" is not a source node of graph \"two\"; its source nodes: "
                "camA, camB");
}

void TestOpenStreamKeepsTheOneSourceWording() {
    const GraphSpec one = ParseGraphText(CascadeJson(), "t.json");
    const GraphSpec two = ParseGraphText(TwoStreamJson(), "two.json");
    consumer::StreamUri stream;
    stream.source = "cam";
    std::string message;
    try {
        consumer::OpenStream(one, stream);
    } catch (const GraphError& error) {
        message = error.what();
    }
    GRAPH_CHECK(message == "ERROR [GRAPH_SCHEMA] graph \"t\": the source node has no "
                           "\"uri\"\n  -> add \"uri\", or pass --input");
    stream.source = "camB";
    message.clear();
    try {
        consumer::OpenStream(two, stream);
    } catch (const GraphError& error) {
        message = error.what();
    }
    GRAPH_CHECK(message == "ERROR [GRAPH_SCHEMA] node \"camB\": the source node has no "
                           "\"uri\"\n  -> add \"uri\" to it, or pass --input camB=<uri>");
    stream.uri = "nope/missing.jpg";
    message.clear();
    try {
        ScopedOpenCvLogLevel quiet(cv::utils::logging::LOG_LEVEL_SILENT);
        consumer::OpenStream(two, stream);
    } catch (const GraphError& error) {
        message = error.what();
    }
    GRAPH_CHECK(message.find("ERROR [GRAPH_SCHEMA] source \"nope/missing.jpg\": ") == 0);
    GRAPH_CHECK(message.find("\n  -> check the path, or pass --input camB=<file>") !=
                std::string::npos);
    stream.source = "cam";
    message.clear();
    try {
        ScopedOpenCvLogLevel quiet(cv::utils::logging::LOG_LEVEL_SILENT);
        consumer::OpenStream(one, stream);
    } catch (const GraphError& error) {
        message = error.what();
    }
    GRAPH_CHECK(message.find("\n  -> check the path, or pass --input <file>") !=
                std::string::npos);
}

// Review Focus #5.
void TestStreamReaderReadsInTurnAndStopsEachStreamAtItsLimit() {
    CountingInput* a = NULL;
    CountingInput* b = NULL;
    {
        consumer::StreamReader reader = TwoCounting(3, 1, 0, &a, &b);
        const int expected[][2] = {{0, 0}, {1, 0}, {0, 1}, {0, 2}};
        GRAPH_CHECK(ReadAll(&reader) == Order(expected, 4));
        GRAPH_CHECK(reader.frames_read(0) == 3 && reader.frames_read(1) == 1);
        GRAPH_CHECK(reader.ended(0) && reader.ended(1));
        // one failed read ends a stream, and it is never read again
        GRAPH_CHECK(a->reads() == 4 && b->reads() == 2);
        std::size_t stream = 0;
        std::size_t index = 0;
        cv::Mat frame;
        GRAPH_CHECK(!reader.Next(&stream, &frame, &index));
        GRAPH_CHECK(a->reads() == 4 && b->reads() == 2);
    }
    {
        // --frames 2: two frames each, and no third read (a camera would grab one)
        consumer::StreamReader reader = TwoCounting(5, 5, 2, &a, &b);
        const int expected[][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
        GRAPH_CHECK(ReadAll(&reader) == Order(expected, 4));
        GRAPH_CHECK(a->reads() == 2 && b->reads() == 2);
    }
    {
        // a stream with no frames ends at its first read; the other carries on
        consumer::StreamReader reader = TwoCounting(0, 2, 0, &a, &b);
        const int expected[][2] = {{1, 0}, {1, 1}};
        GRAPH_CHECK(ReadAll(&reader) == Order(expected, 2));
        GRAPH_CHECK(reader.frames_read(0) == 0 && reader.ended(0));
        GRAPH_CHECK(a->reads() == 1);
    }
    {
        // one stream: exactly the loop the CLI had
        std::vector<consumer::StreamInput> inputs(1);
        inputs[0].source = "cam";
        inputs[0].input.reset(new CountingInput(3, 1));
        a = static_cast<CountingInput*>(inputs[0].input.get());
        consumer::StreamReader reader(std::move(inputs), 2);
        const int expected[][2] = {{0, 0}, {0, 1}};
        GRAPH_CHECK(ReadAll(&reader) == Order(expected, 2));
        GRAPH_CHECK(a->reads() == 2 && reader.size() == 1 && reader.input(0).source == "cam");
    }
}

// Review Focus #4.
void TestStreamOutputPaths() {
    GRAPH_CHECK(cli::NumberedPath("out.png", 3) == "out_000003.png");
    GRAPH_CHECK(cli::NumberedPath("out", 3) == "out_000003.png");
    GRAPH_CHECK(cli::TaggedPath("run.v1/out.mp4", "cam1") == "run.v1/out_cam1.mp4");
    GRAPH_CHECK(cli::TaggedPath("dir.x/out", "cam1") == "dir.x/out_cam1");
    GRAPH_CHECK(cli::StreamVideoPath("out.mp4", "cam1", false) == "out.mp4");
    GRAPH_CHECK(cli::StreamVideoPath("out.mp4", "cam1", true) == "out_cam1.mp4");
    GRAPH_CHECK(cli::StreamImagePath("out.png", "cam", false, true, 0) == "out.png");
    GRAPH_CHECK(cli::StreamImagePath("out.png", "cam", false, false, 7) == "out_000007.png");
    GRAPH_CHECK(cli::StreamImagePath("out.png", "cam2", true, true, 0) == "out_cam2_000000.png");
    GRAPH_CHECK(cli::StreamImagePath("out", "cam2", true, false, 4) == "out_cam2_000004.png");
    GRAPH_CHECK(cli::WindowName("cam1", false) == "multi_model_graph");
    GRAPH_CHECK(cli::WindowName("cam1", true) == "multi_model_graph: cam1");
}

void TestFrameExecutorsSubmitToTheNamedStream() {
    GraphSpec spec = ParseGraphText(TwoStreamJson(), "two.json");
    for (int kind = 0; kind < 2; ++kind) {
        FakeModelRegistry registry = BuildMovingCascadeRegistry();
        StageGraph graph;
        graph.Build(spec, registry, "/models", false);
        std::unique_ptr<consumer::FrameExecutor> executor;
        if (kind == 0) {
            executor.reset(new consumer::SyncFrameExecutor());
        } else {
            executor.reset(new consumer::AsyncFrameExecutor(Frames(2)));
        }
        executor->Submit(graph, 1, FrameWithKey(2), 0);
        executor->Submit(graph, 0, FrameWithKey(0), 0);
        executor->Finish(graph);
        FrameReport first;
        FrameReport second;
        GRAPH_CHECK(executor->TryNext(&first, NULL) && first.stream == "camB");
        GRAPH_CHECK(executor->TryNext(&second, NULL) && second.stream == "camA");
    }
}

// =====================================================================
// SP2 final review fix wave
// =====================================================================

// M5: the full-frame rule names the stream the clash is in, a later one too.
void TestValidateNamesTheLaterStreamAFullFrameClashIsIn() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    // Stream cam1 brings od one image (cam1); stream cam2 brings it two
    // (sr's and cam2's own).
    const std::string json =
        "{\"version\":1,\"name\":\"h2\",\"nodes\":["
        "{\"id\":\"cam1\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"sr\",\"model\":\"sr_x2\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam1\",\"to\":\"od\"},"
        "{\"from\":\"cam2\",\"to\":\"sr\"},"
        "{\"from\":\"sr\",\"to\":\"od\"},"
        "{\"from\":\"cam2\",\"to\":\"od\"}]}";
    const std::string message = ValidationError(json, registry);
    const std::string expected =
        "ERROR [GRAPH_EDGE] node \"od\": receives full-frame input from more "
        "than one image in stream \"cam2\": \"sr\", \"cam2\"\n"
        "  -> keep one plain edge into this node, or split it into one node "
        "per input";
    GRAPH_CHECK(message == expected);
    if (message != expected) std::printf("      got: %s\n", message.c_str());
}

// M3: a graph broken twice - an unreachable node that would also break the
// full-frame rule - is reported by its orphan, the root cause. Rule 4a only
// counts the nodes of some stream (the Task 1 ruling).
void TestValidateReportsTheOrphanOfAnUnreachableFullFrameClash() {
    FakeModelRegistry registry = BuildHandOffRegistry();
    const std::string json =
        "{\"version\":1,\"name\":\"h\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},"
        "{\"id\":\"sa\",\"model\":\"sr_x2\"},"
        "{\"id\":\"sb\",\"model\":\"sr_x2\"},"
        "{\"id\":\"x\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"sa\",\"to\":\"x\"},"
        "{\"from\":\"sb\",\"to\":\"x\"}]}";
    const std::string message = ValidationError(json, registry);
    const std::string expected =
        "ERROR [GRAPH_ORPHAN] node \"sa\": unreachable from any source\n"
        "  -> connect it with an edge, or remove it";
    GRAPH_CHECK(message == expected);
    if (message != expected) std::printf("      got: %s\n", message.c_str());
}

// M6: a stream that ends in the middle of the declaration order is skipped
// from then on; the streams on either side keep their turns.
void TestStreamReaderSkipsAMiddleStreamThatEndsEarly() {
    std::vector<consumer::StreamInput> inputs(3);
    const int frames[] = {3, 1, 3};
    CountingInput* counted[3] = {NULL, NULL, NULL};
    for (std::size_t s = 0; s < 3; ++s) {
        inputs[s].source = std::string("cam") + char('A' + s);
        inputs[s].input.reset(new CountingInput(frames[s], int(s) + 1));
        counted[s] = static_cast<CountingInput*>(inputs[s].input.get());
    }
    consumer::StreamReader reader(std::move(inputs), 0);
    const int expected[][2] = {{0, 0}, {1, 0}, {2, 0}, {0, 1}, {2, 1}, {0, 2}, {2, 2}};
    GRAPH_CHECK(ReadAll(&reader) == Order(expected, 7));
    GRAPH_CHECK(reader.frames_read(0) == 3 && reader.frames_read(1) == 1 &&
                reader.frames_read(2) == 3);
    // camB's one failed read ended it; it was never read again
    GRAPH_CHECK(counted[0]->reads() == 4 && counted[1]->reads() == 2 &&
                counted[2]->reads() == 4);
}

// M1: a source id may itself contain '='. "--input <id>=<uri>" binds to the
// longest source id that, followed by '=', begins the value.
void TestParseInputOverridesBindsTheLongestSourceIdBeforeAnEquals() {
    const std::string two_json =
        "{\"version\":1,\"name\":\"yard\",\"nodes\":["
        "{\"id\":\"back\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"back=yard\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"back\",\"to\":\"od\"},"
        "{\"from\":\"back=yard\",\"to\":\"od\"}]}";
    const GraphSpec two = ParseGraphText(two_json, "yard.json");
    std::map<std::string, std::string> bound;
    std::vector<std::string> inputs(1, "back=yard=clip.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 1 && bound["back=yard"] == "clip.mp4");
    inputs.assign(1, "back=x.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 1 && bound["back"] == "x.mp4");
    inputs.assign(1, "back=a.mp4");
    inputs.push_back("back=yard=b.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 2 && bound["back"] == "a.mp4" && bound["back=yard"] == "b.mp4");
    inputs.assign(1, "back=yard=");
    GRAPH_CHECK(consumer::ParseInputOverrides(two, inputs, &bound) ==
                "--input back=yard= has no uri after \"=\"");

    // One source named "back=yard": its "<id>=" form binds as well, not as a
    // bare uri "back=yard=clip.mp4".
    const std::string one_json =
        "{\"version\":1,\"name\":\"yard1\",\"nodes\":["
        "{\"id\":\"back=yard\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"back=yard\",\"to\":\"od\"}]}";
    const GraphSpec one = ParseGraphText(one_json, "yard1.json");
    inputs.assign(1, "back=yard=clip.mp4");
    GRAPH_CHECK(consumer::ParseInputOverrides(one, inputs, &bound).empty());
    GRAPH_CHECK(bound.size() == 1 && bound["back=yard"] == "clip.mp4");
}

// ---------------------------------------------------------------------
// SP4 Task 1: named output ports (U-08)
// ---------------------------------------------------------------------
namespace {

StagePorts DrivePorts() {
    StagePorts ports;
    std::shared_ptr<LabelMapData> mask(new LabelMapData());
    mask->labels = cv::Mat::zeros(480, 640, CV_8U);
    mask->labels(cv::Rect(0, 240, 640, 240)).setTo(1);
    mask->binary_mask = true;
    mask->mask_color = cv::Vec3b(0, 180, 0);
    ports["drivable"] = mask;
    std::shared_ptr<BoxesData> people(new BoxesData(Shape::kBoxes));
    people->items.push_back(MakeItem(100, 100, 50, 100, 0.8f, "person"));
    people->items.push_back(MakeItem(300, 120, 50, 100, 0.7f, "person"));
    ports["people"] = people;
    std::shared_ptr<ImageData> enhanced(new ImageData());
    enhanced->image = cv::Mat(960, 1280, CV_8UC3, cv::Scalar(7, 8, 9));
    ports["enhanced"] = enhanced;
    return ports;
}

/// "drive": one primary box plus three extra outputs (a mask, a second box
/// set, a x2 image). "cls" takes crops; "od2" takes a handed-off image.
FakeModelRegistry BuildPortsRegistry() {
    FakeModelRegistry registry;
    ModelInfo drive = MakeFullFrameInfo("drive", "panoptic_driving_perception", Shape::kBoxes);
    drive.ports.push_back(PortInfo("drivable", Shape::kLabelMap));
    drive.ports.push_back(PortInfo("people", Shape::kBoxes));
    drive.ports.push_back(PortInfo("enhanced", Shape::kImage));
    std::shared_ptr<BoxesData> primary(new BoxesData(Shape::kBoxes));
    primary->items.push_back(MakeItem(10, 10, 40, 60, 0.9f, "vehicle"));
    registry.AddModel(drive, primary);
    registry.SetPorts("drive", DrivePorts());

    ModelInfo cls = MakeFullFrameInfo("cls", "classification", Shape::kScores);
    cls.input_contract = InputContract::kEither;
    std::shared_ptr<ScoresData> scores(new ScoresData());
    ClassificationResult top;
    top.class_id = 1;
    top.class_name = "walker";
    top.confidence = 0.6f;
    scores->items.push_back(top);
    registry.AddModel(cls, scores);

    ModelInfo od2 = MakeFullFrameInfo("od2", "object_detection", Shape::kBoxes);
    std::shared_ptr<BoxesData> found(new BoxesData(Shape::kBoxes));
    found->items.push_back(MakeItem(200, 200, 100, 100, 0.9f, "car"));
    registry.AddModel(od2, found);
    return registry;
}

/// cam -> drive (`drive_extra` is spliced into drive's node object), plus `edges`.
std::string PortsJson(const std::string& drive_extra, const std::string& edges) {
    return "{\"version\":1,\"name\":\"ports\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"drive\",\"model\":\"drive\"" + drive_extra + "},"
           "{\"id\":\"cls\",\"model\":\"cls\"},"
           "{\"id\":\"od2\",\"model\":\"od2\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"drive\"}," + edges + "]}";
}

const char kPortEdges[] =
    "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":\"people\"},"
    "{\"from\":\"drive\",\"to\":\"od2\",\"port\":\"enhanced\"}";
const char kCamToOd2[] = ",{\"from\":\"cam\",\"to\":\"od2\"}";

/// Builds into `graph`, which the caller keeps: the graph owns the stages,
/// so FakeModelRegistry::last_stage() is valid only while it lives.
FrameReport RunPorts(const std::string& json, bool async, FakeModelRegistry* registry,
                     StageGraph* graph) {
    GraphSpec spec = ParseGraphText(json, "p.json");
    ValidateGraph(spec, *registry);
    graph->Build(spec, *registry, "/models", false);
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    if (async) {
        AsyncExecutor executor(StageJobs(4));
        return executor.RunFrame(*graph, frame, 0);
    }
    SyncExecutor executor;
    return executor.RunFrame(*graph, frame, 0);
}

FrameReport RunPorts(const std::string& json, bool async, FakeModelRegistry* registry) {
    StageGraph graph;
    return RunPorts(json, async, registry, &graph);
}

}  // namespace

void TestParseReadsAnEdgePort() {
    const GraphSpec spec = ParseGraphText(PortsJson("", kPortEdges), "p.json");
    GRAPH_CHECK(spec.edges[0].port.empty());
    GRAPH_CHECK(spec.edges[1].port == "people");
    GRAPH_CHECK(spec.edges[2].port == "enhanced");
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        PortsJson("", "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":5}"),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"port\" must be a non-empty string, got 5") != std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(
        PortsJson("", "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":\"\"}"),
        GraphErrorCode::kGraphSchema, &message));
}

void TestValidateEdgePorts() {
    FakeModelRegistry registry = BuildPortsRegistry();
    GRAPH_CHECK(ValidationError(PortsJson("", kPortEdges), registry).empty());
    // The primary's own name selects the primary.
    GRAPH_CHECK(ValidationError(PortsJson("", std::string(
        "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":\"boxes\"}") + kCamToOd2),
        registry).empty());

    std::string m = ValidationError(PortsJson("", std::string(
        "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":\"wings\"}") + kCamToOd2),
        registry);
    GRAPH_CHECK(m.find("ERROR [GRAPH_EDGE] edge \"drive\"->\"cls\": \"drive\" has no output "
                       "port \"wings\"") == 0);
    GRAPH_CHECK(m.find("its outputs: boxes, drivable, people, enhanced") != std::string::npos);

    m = ValidationError(PortsJson("", "{\"from\":\"cam\",\"to\":\"od2\",\"port\":\"frame\"},"
                                      "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{}}"), registry);
    GRAPH_CHECK(m.find("\"cam\" is a source; \"port\" selects one of a model's outputs") !=
                std::string::npos);

    m = ValidationError(PortsJson("", std::string(
        "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{},\"port\":\"drivable\"}") + kCamToOd2),
        registry);
    GRAPH_CHECK(m.find("\"drive\" port \"drivable\" produces labelmap, not boxes") !=
                std::string::npos);

    m = ValidationError(PortsJson("", "{\"from\":\"drive\",\"to\":\"od2\",\"port\":\"drivable\"},"
                                      "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{}}"), registry);
    GRAPH_CHECK(m.find("\"drive\" port \"drivable\" produces labelmap, and only an image can be "
                       "handed off on a plain edge") != std::string::npos);

    m = ValidationError(PortsJson("", std::string(
        "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{\"align\":\"face5\"},"
        "\"port\":\"people\"}") + kCamToOd2),
        registry);
    GRAPH_CHECK(m.find("ERROR [GRAPH_ALIGN]") == 0);
    GRAPH_CHECK(m.find("port \"people\" of \"drive\" carries none") != std::string::npos);

    // Without "port" every message is today's text.
    m = ValidationError(PortsJson("", std::string(
        "{\"from\":\"drive\",\"to\":\"od2\"},"
        "{\"from\":\"drive\",\"to\":\"cls\",\"roi\":{}}")), registry);
    GRAPH_CHECK(m.find("\"drive\" produces boxes, and a plain edge carries only a source frame") !=
                std::string::npos);
}

void TestSyncRoutesEachEdgeFromItsPort() {
    FakeModelRegistry registry = BuildPortsRegistry();
    StageGraph graph;  // keeps od2's stage alive for last_stage() below
    FrameReport report = RunPorts(PortsJson("", kPortEdges), false, &registry, &graph);
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_results["drive"].ports.size() == 3);
    // cls ran once per box of the "people" port (2), not per primary box (1).
    GRAPH_CHECK(report.roi_results["cls"].size() == 2);
    if (report.roi_results["cls"].size() == 2) {
        GRAPH_CHECK(report.roi_results["cls"][1].origin.src_box.x == 300.f);
    }
    // od2 ran on the "enhanced" image, mapped back to the source by 1/2.
    const FakeStage* od2 = registry.last_stage("od2");
    GRAPH_CHECK(od2 != NULL && od2->seen().size() == 1);
    if (od2 != NULL && od2->seen().size() == 1) {
        GRAPH_CHECK(od2->seen()[0].image.cols == 1280);
        GRAPH_CHECK(od2->seen()[0].origin.inv_align(0, 0) == 0.5f);
    }
}

void TestAsyncRoutesPortsLikeSyncUnderReversedDelivery() {
    FakeModelRegistry sync_registry = BuildPortsRegistry();
    const FrameReport expected = RunPorts(PortsJson("", kPortEdges), false, &sync_registry);
    FakeModelRegistry async_registry = BuildPortsRegistry();
    async_registry.SetDeliveryOrder("cls", FakeModelRegistry::kReverse);
    const FrameReport actual = RunPorts(PortsJson("", kPortEdges), true, &async_registry);
    GRAPH_CHECK(expected.roi_results.at("cls").size() == 2);
    GRAPH_CHECK(actual.node_results.at("drive").ports.size() == 3);
    GRAPH_CHECK(expected == actual);
}

// Review Focus 1: tracking rebuilds the producer's result; the ports survive.
void TestTrackedProducerKeepsItsPorts() {
    for (int async = 0; async < 2; ++async) {
        FakeModelRegistry registry = BuildPortsRegistry();
        FrameReport report =
            RunPorts(PortsJson(",\"track\":{}", kPortEdges), async != 0, &registry);
        const StageResult& drive = report.node_results["drive"];
        GRAPH_CHECK(drive.ports.size() == 3);
        const BoxesData* boxes = dynamic_cast<const BoxesData*>(drive.data.get());
        GRAPH_CHECK(boxes != NULL && boxes->items.size() == 1);
        if (boxes != NULL && boxes->items.size() == 1) GRAPH_CHECK(boxes->items[0].track_id >= 0);
        // Crops cut from a non-primary box port are untracked.
        GRAPH_CHECK(report.roi_results["cls"].size() == 2);
        if (report.roi_results["cls"].size() == 2) {
            GRAPH_CHECK(report.roi_results["cls"][0].origin.track_id == -1);
        }
    }
}

// A stage that breaks its ports contract: "people" is declared as boxes but
// is missing, or is not box-shaped. The ROI edge reading it records an
// error in both executors, like a plain edge that gets no image, instead
// of silently cutting no crops.
void TestRoiEdgeFromABrokenPortRecordsAnError() {
    const std::string expected = "node \"cls\": \"drive\" port \"people\" handed off no boxes";
    for (int variant = 0; variant < 2; ++variant) {
        StagePorts broken = DrivePorts();
        if (variant == 0) {
            broken.erase("people");
        } else {
            broken["people"] = broken["enhanced"];
        }
        FrameReport reports[2];
        for (int async = 0; async < 2; ++async) {
            FakeModelRegistry registry = BuildPortsRegistry();
            registry.SetPorts("drive", broken);
            reports[async] = RunPorts(PortsJson("", kPortEdges), async != 0, &registry);
            GRAPH_CHECK(reports[async].error == expected);
            GRAPH_CHECK(reports[async].roi_results.count("cls") == 1 &&
                        reports[async].roi_results["cls"].empty());
        }
        GRAPH_CHECK(reports[0] == reports[1]);
    }
}

void TestReportsDifferingOnlyInAPortAreUnequal() {
    FrameReport a;
    StageResult r;
    r.data = std::shared_ptr<BoxesData>(new BoxesData(Shape::kBoxes));
    r.ports = DrivePorts();
    a.node_results["drive"] = r;
    FrameReport b = a;
    GRAPH_CHECK(a == b);

    const LabelMapData* drivable =
        static_cast<const LabelMapData*>(a.node_results["drive"].ports["drivable"].get());
    std::shared_ptr<LabelMapData> pixel(new LabelMapData(*drivable));
    pixel->labels = drivable->labels.clone();
    pixel->labels.at<uchar>(0, 0) = 1;
    b.node_results["drive"].ports["drivable"] = pixel;
    GRAPH_CHECK(!(a == b));

    FrameReport c = a;
    c.node_results["drive"].ports.erase("people");
    GRAPH_CHECK(!(a == c));

    FrameReport d = a;
    std::shared_ptr<LabelMapData> flag(new LabelMapData(*drivable));
    flag->binary_mask = false;
    d.node_results["drive"].ports["drivable"] = flag;
    GRAPH_CHECK(!(a == d));

    FrameReport e = a;
    std::shared_ptr<DenseMapData> dense(new DenseMapData());
    dense->values = cv::Mat::ones(2, 4, CV_32F);
    std::shared_ptr<DenseMapData> flat(new DenseMapData(*dense));
    flat->spatial = false;
    a.node_results["drive"].ports["d"] = dense;
    e.node_results["drive"].ports["d"] = flat;
    GRAPH_CHECK(!(a == e));

    // The same rule holds for the ports of a ROI result.
    FrameReport f;
    StageResult crop;
    crop.data = std::shared_ptr<ScoresData>(new ScoresData());
    crop.ports["d"] = dense;
    f.roi_results["cls"].push_back(crop);
    FrameReport g = f;
    GRAPH_CHECK(f == g);
    g.roi_results["cls"][0].ports["d"] = flat;
    GRAPH_CHECK(!(f == g));
    FrameReport h = f;
    h.roi_results["cls"][0].ports.clear();
    GRAPH_CHECK(!(f == h));
}

// A ROI consumer whose results carry ports, delivered in reverse under
// async: both executors report the same ports on every crop's result.
void TestRoiResultsKeepTheirPortsUnderReversedDelivery() {
    StagePorts cls_ports;
    std::shared_ptr<DenseMapData> embedding(new DenseMapData());
    embedding->values = cv::Mat::ones(1, 8, CV_32F);
    embedding->spatial = false;
    cls_ports["embedding"] = embedding;

    FakeModelRegistry sync_registry = BuildPortsRegistry();
    sync_registry.SetPorts("cls", cls_ports);
    const FrameReport expected = RunPorts(PortsJson("", kPortEdges), false, &sync_registry);
    FakeModelRegistry async_registry = BuildPortsRegistry();
    async_registry.SetPorts("cls", cls_ports);
    async_registry.SetDeliveryOrder("cls", FakeModelRegistry::kReverse);
    const FrameReport actual = RunPorts(PortsJson("", kPortEdges), true, &async_registry);

    const FrameReport* reports[2] = {&expected, &actual};
    for (int r = 0; r < 2; ++r) {
        const std::vector<StageResult>& crops = reports[r]->roi_results.at("cls");
        GRAPH_CHECK(crops.size() == 2);
        for (std::size_t i = 0; i < crops.size(); ++i) {
            GRAPH_CHECK(crops[i].ports.size() == 1 && crops[i].ports.count("embedding") == 1);
        }
    }
    GRAPH_CHECK(expected == actual);
}

// ---------------------------------------------------------------------
// SP4 Task 2: ports in the report and the render (U-08)
// ---------------------------------------------------------------------

// Review Focus 2: a result without ports is written exactly as before.
void TestReportWritesPortsOnlyWhenThereAreSome() {
    FrameReport report;
    StageResult plain;
    plain.data = std::shared_ptr<BoxesData>(new BoxesData(Shape::kBoxes));
    report.node_results["od"] = plain;
    StageResult drive = plain;
    drive.ports = DrivePorts();
    report.node_results["drive"] = drive;
    report.roi_results["cls"].push_back(drive);

    // Not const: operator[] on a const json with a missing key is undefined
    // behaviour, which would abort the binary instead of failing a check.
    nlohmann::json out = consumer::ReportToJson(report, false);
    nlohmann::json& od = out["nodes"]["od"];
    GRAPH_CHECK(od.size() == 2 && od.count("origin") == 1 && od.count("payload") == 1);
    GRAPH_CHECK(out["nodes"]["drive"].contains("ports"));
    if (!out["nodes"]["drive"].contains("ports")) return;
    nlohmann::json& ports = out["nodes"]["drive"]["ports"];
    GRAPH_CHECK(ports.size() == 3);
    GRAPH_CHECK(ports["drivable"]["shape"] == "labelmap");
    GRAPH_CHECK(ports["drivable"]["labels"]["nonzero"] == 640 * 240);
    GRAPH_CHECK(ports["people"]["items"].size() == 2);
    GRAPH_CHECK(ports["enhanced"]["image"]["cols"] == 1280);
    nlohmann::json& cls = out["roi_nodes"]["cls"];
    GRAPH_CHECK(cls.is_array() && cls.size() == 1 && cls[0].contains("ports"));
    if (!(cls.is_array() && cls.size() == 1 && cls[0].contains("ports"))) return;
    GRAPH_CHECK(cls[0]["ports"].size() == 3);
}

void TestRenderReportDrawsPortsInTheirShapesPass() {
    const cv::Mat source = cv::Mat::zeros(40, 40, CV_8UC3);
    FrameReport report;
    StageResult drive;
    drive.data = std::shared_ptr<BoxesData>(new BoxesData(Shape::kBoxes));
    std::shared_ptr<LabelMapData> mask(new LabelMapData());
    mask->labels = cv::Mat::zeros(40, 40, CV_8U);
    mask->labels(cv::Rect(0, 0, 20, 40)).setTo(1);
    mask->binary_mask = true;
    mask->mask_color = cv::Vec3b(0, 180, 0);
    drive.ports["drivable"] = mask;
    std::shared_ptr<DenseMapData> descriptors(new DenseMapData());
    descriptors->values = cv::Mat(5, 256, CV_32F, cv::Scalar(3.f));
    descriptors->values.at<float>(0, 0) = 9.f;
    descriptors->spatial = false;
    drive.ports["descriptors"] = descriptors;
    report.node_results["drive"] = drive;

    const cv::Mat rendered = RenderReport(source, report);
    GRAPH_CHECK(rendered.at<cv::Vec3b>(20, 5) == cv::Vec3b(0, 90, 0));   // mask, alpha 0.5
    // Label 0 stays undrawn, and the descriptors are not a picture.
    GRAPH_CHECK(rendered.at<cv::Vec3b>(20, 35) == cv::Vec3b(0, 0, 0));

    // The same mask without the flag is today's palette rendering, everywhere.
    mask->binary_mask = false;
    const cv::Mat palette = RenderReport(source, report);
    GRAPH_CHECK(palette.at<cv::Vec3b>(20, 35) != cv::Vec3b(0, 0, 0));
}

// ---------------------------------------------------------------------
// SP4 Task 3: factory graph traits (U-14, B7)
// ---------------------------------------------------------------------
namespace sp4_traits {
struct DeclaresRoi {
    static constexpr dxapp::GraphInput graphInput() { return dxapp::GraphInput::kRoi; }
};
struct DeclaresPorts {
    static constexpr const char* graphPorts() { return "descriptors"; }
};
struct DeclaresNothing {};
}  // namespace sp4_traits

static_assert(dxapp::graph::detail::DeclaredGraphInput<sp4_traits::DeclaresRoi>(0) ==
                  dxapp::GraphInput::kRoi, "a declared contract is seen");
static_assert(dxapp::graph::detail::DeclaredGraphInput<sp4_traits::DeclaresNothing>(0) ==
                  dxapp::GraphInput::kDefault, "no declaration is kDefault");
static_assert(dxapp::graph::detail::SameText(
                  dxapp::graph::detail::DeclaredGraphPorts<sp4_traits::DeclaresPorts>(0),
                  "descriptors"), "declared ports are seen");
static_assert(!dxapp::graph::detail::SameText(
                  dxapp::graph::detail::DeclaredGraphPorts<sp4_traits::DeclaresPorts>(0),
                  "descriptor"), "SameText compares the whole text");
static_assert(dxapp::graph::detail::SameText(
                  dxapp::graph::detail::DeclaredGraphPorts<sp4_traits::DeclaresNothing>(0), ""),
              "no declaration is the empty text");

namespace {
std::string TopDownJson(const std::string& pose_model) {
    return "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\"},{\"id\":\"pose\",\"model\":\"" + pose_model +
           "\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
           "{\"from\":\"od\",\"to\":\"pose\",\"roi\":{\"classes\":[\"person\"]}}]}";
}
}  // namespace

void TestVitPoseIsATopDownRoiConsumer() {
    StaticModelRegistry registry;
    const ModelInfo* vit = registry.find("vitpose-s_256x192");
    GRAPH_CHECK(vit != NULL);
    if (vit == NULL) return;
    GRAPH_CHECK(vit->input_contract == InputContract::kEither);
    const ModelInfo* yolo_pose = registry.find("yolov8s_pose");
    GRAPH_CHECK(yolo_pose != NULL && yolo_pose->input_contract == InputContract::kFullFrame);
    GRAPH_CHECK(ValidationError(TopDownJson("vitpose-s_256x192"), registry).empty());
    GRAPH_CHECK(ValidationError(TopDownJson("yolov8s_pose"), registry)
                    .find("\"pose\" requires a full frame") != std::string::npos);
}

// ---------------------------------------------------------------------
// SP4 Task 4: the models' own ports (U-08)
// ---------------------------------------------------------------------
void TestPanopticConversionKeepsTheBoxesAndAddsBothMasks() {
    PanopticResult frame;
    std::vector<float> box;
    box.push_back(1.f); box.push_back(2.f); box.push_back(11.f); box.push_back(22.f);
    frame.detections.push_back(DetectionResult(box, 0.8f, 3, "vehicle"));
    frame.drivable = cv::Mat::zeros(4, 6, CV_8U);
    frame.drivable.at<uchar>(1, 2) = 1;
    frame.lane = cv::Mat::zeros(4, 6, CV_8U);
    frame.lane.at<uchar>(3, 5) = 1;
    const std::vector<PanopticResult> results(1, frame);

    // Held in locals: a raw pointer into a temporary StageDataPtr dangles.
    const StageDataPtr boxes_data = ToStageData(results);
    const StageDataPtr direct_data = ToStageData(frame.detections);
    const BoxesData* boxes = dynamic_cast<const BoxesData*>(boxes_data.get());
    const BoxesData* direct = dynamic_cast<const BoxesData*>(direct_data.get());
    GRAPH_CHECK(boxes != NULL && direct != NULL && boxes->items.size() == 1);
    if (boxes != NULL && direct != NULL && boxes->items.size() == 1) {
        GRAPH_CHECK(boxes->items[0].box == direct->items[0].box);
        GRAPH_CHECK(boxes->items[0].class_name == "vehicle" && boxes->items[0].score == 0.8f);
    }
    const StagePorts ports = ToStagePorts(results);
    GRAPH_CHECK(ports.size() == 2);
    const LabelMapData* drivable = dynamic_cast<const LabelMapData*>(ports.at("drivable").get());
    const LabelMapData* lane = dynamic_cast<const LabelMapData*>(ports.at("lane").get());
    GRAPH_CHECK(drivable != NULL && drivable->binary_mask &&
                drivable->mask_color == cv::Vec3b(0, 180, 0));
    GRAPH_CHECK(lane != NULL && lane->binary_mask && lane->mask_color == cv::Vec3b(0, 0, 200));
    if (drivable != NULL) {
        GRAPH_CHECK(cv::countNonZero(drivable->labels) == 1 &&
                    drivable->labels.at<uchar>(1, 2) == 1);
    }
    if (lane != NULL) GRAPH_CHECK(lane->labels.at<uchar>(3, 5) == 1);
    // Nothing decoded still fills every port (the PORTS CONTRACT).
    const StagePorts none = ToStagePorts(std::vector<PanopticResult>());
    GRAPH_CHECK(none.size() == 2 && none.at("drivable") && none.at("lane"));
    GRAPH_CHECK(ToStageData(std::vector<PanopticResult>())->shape() == Shape::kBoxes);
}

typedef std::vector<std::vector<float> > DescriptorRows;

/// PoseResult::descriptors as TARGET's SuperPoint sets it: a shared,
/// immutable set of rows (Decision 4).
std::shared_ptr<const DescriptorRows> SharedRows(const DescriptorRows& rows) {
    return std::make_shared<const DescriptorRows>(rows);
}

void TestPoseDescriptorsBecomeOneMatrixRowPerKeypoint() {
    std::vector<PoseResult> results(2);
    results[0].keypoints.resize(2);
    DescriptorRows first(2, std::vector<float>(4, 0.5f));
    first[1][3] = 9.f;
    results[0].descriptors = SharedRows(first);
    results[1].keypoints.resize(1);
    results[1].descriptors = SharedRows(DescriptorRows(1, std::vector<float>(4, 1.f)));
    const StagePorts ports = ToStagePorts(results);
    const DenseMapData* desc = dynamic_cast<const DenseMapData*>(ports.at("descriptors").get());
    GRAPH_CHECK(desc != NULL);
    if (desc == NULL) return;
    GRAPH_CHECK(!desc->spatial);
    GRAPH_CHECK(desc->values.rows == 3 && desc->values.cols == 4 && desc->values.type() == CV_32F);
    GRAPH_CHECK(desc->values.at<float>(1, 3) == 9.f && desc->values.at<float>(2, 0) == 1.f);
    const StageDataPtr kp_data = ToStageData(results);
    const KeypointsData* kp = dynamic_cast<const KeypointsData*>(kp_data.get());
    GRAPH_CHECK(kp != NULL && kp->items.size() == 2 && !kp->items[0].descriptors);
    GRAPH_CHECK(results[0].descriptors != NULL);   // the conversion leaves its input alone
    std::vector<PoseResult> plain(1);
    plain[0].keypoints.resize(17);
    const StagePorts plain_ports = ToStagePorts(plain);
    const DenseMapData* empty =
        dynamic_cast<const DenseMapData*>(plain_ports.at("descriptors").get());
    GRAPH_CHECK(empty != NULL && empty->values.rows == 0);
    // A null set on one item contributes no rows; the others still do.
    std::vector<PoseResult> mixed = results;
    mixed[0].descriptors.reset();
    const StagePorts mixed_ports = ToStagePorts(mixed);
    const DenseMapData* one =
        dynamic_cast<const DenseMapData*>(mixed_ports.at("descriptors").get());
    GRAPH_CHECK(one != NULL && one->values.rows == 1 && one->values.at<float>(0, 0) == 1.f);
    // ragged: a decoder bug, reported, not guessed
    results[1].descriptors = SharedRows(DescriptorRows(1, std::vector<float>(3, 1.f)));
    bool threw = false;
    try { ToStagePorts(results); } catch (const std::runtime_error&) { threw = true; }
    GRAPH_CHECK(threw);
}

// A port converter that throws leaves the stage's result untouched: a failed
// result carries neither data nor ports (TypedStage reports only the error).
void TestAFailedPortConversionLeavesTheResultEmpty() {
    std::vector<PoseResult> ragged(2);
    ragged[0].keypoints.resize(1);
    ragged[0].descriptors = SharedRows(DescriptorRows(1, std::vector<float>(4, 0.5f)));
    ragged[1].keypoints.resize(1);
    ragged[1].descriptors = SharedRows(DescriptorRows(1, std::vector<float>(3, 0.5f)));
    const std::vector<PortInfo> declared(1, PortInfo("descriptors", Shape::kDenseMap));
    RoiRef origin;
    origin.roi_index = 4;
    StageResult result;
    bool threw = false;
    try {
        detail::FillStageResult(ragged, declared, origin, &result);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    GRAPH_CHECK(threw);
    GRAPH_CHECK(!result.data);
    GRAPH_CHECK(result.ports.empty());
    GRAPH_CHECK(result.origin.roi_index != 4);

    ragged[1].descriptors = SharedRows(DescriptorRows(1, std::vector<float>(4, 0.5f)));
    detail::FillStageResult(ragged, declared, origin, &result);
    GRAPH_CHECK(result.data && result.ports.size() == 1 && result.origin.roi_index == 4);
}

void TestHandednessBecomesOneScorePerHand() {
    std::vector<HandLandmarkResult> hands(3);
    hands[0].handedness = "Right"; hands[0].confidence = 0.9f;
    hands[1].handedness = "Left"; hands[1].confidence = 0.8f;
    hands[2].handedness = "Unknown"; hands[2].confidence = 1.f;
    const StagePorts ports = ToStagePorts(hands);
    const ScoresData* s = dynamic_cast<const ScoresData*>(ports.at("handedness").get());
    GRAPH_CHECK(s != NULL && s->items.size() == 3);
    if (s == NULL || s->items.size() != 3) return;
    GRAPH_CHECK(s->items[0].class_id == 1 && s->items[0].class_name == "Right" &&
                s->items[0].confidence == 0.9f);
    GRAPH_CHECK(s->items[1].class_id == 0 && s->items[1].class_name == "Left");
    GRAPH_CHECK(s->items[2].class_id == -1 && s->items[2].top_k.empty());
}

void TestFacePoseBecomesThreeFloatsPerFace() {
    std::vector<FaceAlignmentResult> faces(2);
    faces[0].pose.push_back(10.f); faces[0].pose.push_back(-5.f); faces[0].pose.push_back(2.f);
    faces[1].pose.push_back(1.f); faces[1].pose.push_back(2.f); faces[1].pose.push_back(3.f);
    const StagePorts ports = ToStagePorts(faces);
    const VectorData* v = dynamic_cast<const VectorData*>(ports.at("pose").get());
    GRAPH_CHECK(v != NULL && v->values.size() == 6);
    if (v != NULL && v->values.size() == 6) {
        GRAPH_CHECK(v->values[0] == 10.f && v->values[5] == 3.f);
    }
}

void TestPortProblemNamesAPortTheResultCannotFill() {
    const std::vector<PortInfo> good(1, PortInfo("descriptors", Shape::kDenseMap));
    GRAPH_CHECK(detail::PortProblem<PoseResult>(good).empty());
    GRAPH_CHECK(detail::PortProblem<DetectionResult>(good).find("\"descriptors\"") !=
                std::string::npos);
    const std::vector<PortInfo> wrong(1, PortInfo("descriptors", Shape::kVector));
    GRAPH_CHECK(detail::PortProblem<PoseResult>(wrong).find("densemap") != std::string::npos);
    GRAPH_CHECK(detail::PortProblem<DetectionResult>(std::vector<PortInfo>()).empty());
}

// Review Focus 2: only these models declare ports; yolov8-s-pose shares
// SuperPoint's converter and must not gain one, and an embedding model
// without a gallery has no ranking to port.
void TestRealRegistryDeclaresPortsOnlyWhereTheyExist() {
    StaticModelRegistry registry;
    struct Want { const char* model; const char* port; Shape shape; };
    const Want want[] = {
        {"yolopv2_384x640", "drivable", Shape::kLabelMap},
        {"yolopv2_384x640", "lane", Shape::kLabelMap},
        {"superpoint_480x640", "descriptors", Shape::kDenseMap},
        {"mediapipe-hands-lite_224x224", "handedness", Shape::kScores},
        {"3ddfa-v2_mobilenetv1_120x120", "pose", Shape::kVector},
        {"3ddfa-v2_mobilenet-0.5_120x120", "pose", Shape::kVector},
        {"ppmatting-hrnet-w48-composition_512x512", "alpha", Shape::kDenseMap},
        {"ppmatting-hrnet-w48-distinctions_512x512", "alpha", Shape::kDenseMap},
        {"eigenplaces-resnet18_512x512", "matches", Shape::kScores},
        {"eigenplaces-resnet50_512x512", "matches", Shape::kScores},
        {"pp-shituv2-feature-extraction_224x224", "matches", Shape::kScores},
        {"repvgg-a0-reid_256x128", "matches", Shape::kScores},
        {"clip-img_resnet50_224x224_openai", "matches", Shape::kScores}};
    for (std::size_t i = 0; i < sizeof(want) / sizeof(want[0]); ++i) {
        const ModelInfo* info = registry.find(want[i].model);
        GRAPH_CHECK(info != NULL);
        if (info == NULL) continue;
        bool found = false;
        for (std::size_t p = 0; p < info->ports.size(); ++p) {
            if (info->ports[p].name == want[i].port && info->ports[p].shape == want[i].shape) {
                found = true;
            }
        }
        GRAPH_CHECK(found);
    }
    GRAPH_CHECK(registry.find("yolov8-s-pose_640x640") != NULL &&
                registry.find("yolov8-s-pose_640x640")->ports.empty());
    GRAPH_CHECK(registry.find("clip-text_resnet50_77x512_openai") != NULL &&
                registry.find("clip-text_resnet50_77x512_openai")->ports.empty());
    const std::vector<ModelInfo> all = registry.list();
    int with_ports = 0;
    for (std::size_t m = 0; m < all.size(); ++m) {
        if (!all[m].ports.empty()) ++with_ports;
        for (std::size_t p = 0; p < all[m].ports.size(); ++p) {
            GRAPH_CHECK(all[m].ports[p].name != ToString(all[m].output_shape));
        }
    }
    GRAPH_CHECK(with_ports == static_cast<int>(sizeof(want) / sizeof(want[0])) - 1);
    GRAPH_CHECK(consumer::PortsText(*registry.find("yolopv2_384x640")) ==
                "drivable=labelmap lane=labelmap");
    GRAPH_CHECK(consumer::PortsText(*registry.find("yolov8-n_640x640")).empty());
}

void TestRealPortModelsReportTheirPortsAndAgreeAcrossExecutors() {
    const char* kTest = "TestRealPortModelsReportTheirPortsAndAgreeAcrossExecutors";
    struct Case {
        const char* image;
        const char* nodes;
        const char* edges;
        const char* node;
        const char* port;
        bool roi;
    };
    const Case cases[] = {
        {"sample_street.jpg", "{\"id\":\"drive\",\"model\":\"yolopv2_384x640\"}",
         "{\"from\":\"cam\",\"to\":\"drive\"}", "drive", "lane", false},
        {"sample_street.jpg", "{\"id\":\"sp\",\"model\":\"superpoint_480x640\"}",
         "{\"from\":\"cam\",\"to\":\"sp\"}", "sp", "descriptors", false},
        {"sample_face.jpg",
         "{\"id\":\"face\",\"model\":\"scrfd500m\"},"
         "{\"id\":\"tddfa\",\"model\":\"3ddfa_v2_mobilnetv1_120x120\"}",
         "{\"from\":\"cam\",\"to\":\"face\"},{\"from\":\"face\",\"to\":\"tddfa\",\"roi\":{}}",
         "tddfa", "pose", true},
        {"sample_hand.jpg",
         "{\"id\":\"hands\",\"model\":\"mediapipe-hand-detector_192x192\"},"
         "{\"id\":\"lmk\",\"model\":\"handlandmarklite_1\"}",
         "{\"from\":\"cam\",\"to\":\"hands\"},{\"from\":\"hands\",\"to\":\"lmk\",\"roi\":{}}",
         "lmk", "handedness", true}};
    StaticModelRegistry registry;
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const std::string json =
            std::string("{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\","
                        "\"uri\":\"x\"},") +
            cases[i].nodes + "],\"edges\":[" + cases[i].edges + "]}";
        const GraphSpec spec = ParseGraphText(json, "real_ports.json");
        std::vector<std::string> models;
        for (std::size_t n = 1; n < spec.nodes.size(); ++n) models.push_back(spec.nodes[n].model);
        const std::string missing = MissingArtifacts(registry, models);
        if (!missing.empty()) { Skip(kTest, missing); return; }
        const cv::Mat frame = cv::imread(ProjectRoot() + "/sample/img/" + cases[i].image);
        GRAPH_CHECK(!frame.empty());
        if (frame.empty()) return;
        ValidateGraph(spec, registry);
        StageGraph sync_graph;
        sync_graph.Build(spec, registry, ModelDir());
        SyncExecutor sync_executor;
        const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);
        StageGraph async_graph;
        async_graph.Build(spec, registry, ModelDir());
        AsyncExecutor async_executor(StageJobs(4));
        const FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);
        GRAPH_CHECK(expected.error.empty());
        GRAPH_CHECK(expected == actual);
        const StageResult* first = NULL;
        if (cases[i].roi) {
            std::map<std::string, std::vector<StageResult> >::const_iterator it =
                expected.roi_results.find(cases[i].node);
            GRAPH_CHECK(it != expected.roi_results.end() && !it->second.empty());
            if (it != expected.roi_results.end() && !it->second.empty()) first = &it->second[0];
        } else {
            std::map<std::string, StageResult>::const_iterator it =
                expected.node_results.find(cases[i].node);
            GRAPH_CHECK(it != expected.node_results.end());
            if (it != expected.node_results.end()) first = &it->second;
        }
        if (first == NULL) continue;
        GRAPH_CHECK(first->ports.count(cases[i].port) == 1);
        if (std::string(cases[i].port) == "lane") {
            const LabelMapData* lane =
                dynamic_cast<const LabelMapData*>(first->ports.at("lane").get());
            GRAPH_CHECK(lane != NULL && lane->labels.size() == frame.size() &&
                        cv::countNonZero(lane->labels) > 0);
        }
        if (std::string(cases[i].port) == "descriptors") {
            const DenseMapData* d =
                dynamic_cast<const DenseMapData*>(first->ports.at("descriptors").get());
            const KeypointsData* k = dynamic_cast<const KeypointsData*>(first->data.get());
            GRAPH_CHECK(d != NULL && k != NULL && d->values.cols == 256);
            if (d != NULL && k != NULL && !k->items.empty()) {
                GRAPH_CHECK(d->values.rows == static_cast<int>(k->items[0].keypoints.size()));
            }
        }
        if (std::string(cases[i].port) == "pose") {
            const VectorData* v = dynamic_cast<const VectorData*>(first->ports.at("pose").get());
            GRAPH_CHECK(v != NULL && v->values.size() == 3 && std::isfinite(v->values[0]));
        }
        if (std::string(cases[i].port) == "handedness") {
            const ScoresData* s =
                dynamic_cast<const ScoresData*>(first->ports.at("handedness").get());
            GRAPH_CHECK(s != NULL && s->items.size() == 1);
        }
    }
}

// ---------------------------------------------------------------------
// SP4 Task 5: text and list params (U-62)
// ---------------------------------------------------------------------
namespace {
FakeModelRegistry BuildParamsRegistry() {
    FakeModelRegistry registry;
    ModelInfo od = MakeFullFrameInfo("yolov8n", "object_detection", Shape::kBoxes);
    od.text_params.push_back(ParamInfo("class_names", ParamInfo::kTextList));
    od.text_params.push_back(ParamInfo("label_file", ParamInfo::kText));
    registry.AddModel(od, std::shared_ptr<BoxesData>(new BoxesData(Shape::kBoxes)));
    registry.AddModel(MakeFullFrameInfo("resnet50", "classification", Shape::kScores),
                      std::shared_ptr<ScoresData>(new ScoresData()));
    return registry;
}

std::string ParamsGraph(const std::string& model, const std::string& params) {
    return "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"" + model + "\",\"params\":" + params + "}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}";
}
}  // namespace

void TestParseAcceptsTextAndListParams() {
    const GraphSpec spec = ParseGraphText(ParamsGraph("yolov8n",
        "{\"score_threshold\":0.4,\"label_file\":\"a.txt\",\"class_names\":[\"car\",\"bus\"]}"),
        "p.json");
    const StageParams& p = spec.nodes[1].params;
    GRAPH_CHECK(p.numeric.size() == 1 && p.numeric.at("score_threshold") == 0.4);
    GRAPH_CHECK(p.text.size() == 1 && p.text.at("label_file") == "a.txt");
    GRAPH_CHECK(p.lists.size() == 1 && p.lists.at("class_names").size() == 2);
    if (p.lists.count("class_names") != 0) GRAPH_CHECK(p.lists.at("class_names")[1] == "bus");
}

void TestParseRejectsParamsThatAreNeitherNumberTextNorList() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(ParamsGraph("yolov8n", "{\"a\":true,\"b\":[1,2],\"c\":{}}"),
                               GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("3 invalid values") != std::string::npos);
    GRAPH_CHECK(message.find("\"a\" must be a number, a string or an array of strings, got true") !=
                std::string::npos);
    GRAPH_CHECK(message.find("\"b\" must be a number, a string or an array of strings, "
                             "got [1,2]") != std::string::npos);
    GRAPH_CHECK(message.find("write a switch as 1 or 0") != std::string::npos);
}

void TestValidateChecksEachParamAgainstTheModelsKinds() {
    FakeModelRegistry registry = BuildParamsRegistry();
    GRAPH_CHECK(ValidationError(ParamsGraph("yolov8n",
        "{\"class_names\":[\"a\"],\"label_file\":\"x\",\"top_k\":5,\"anything_at_all\":2}"),
        registry).empty());
    std::string m =
        ValidationError(ParamsGraph("yolov8n", "{\"class_names\":\"person\"}"), registry);
    GRAPH_CHECK(m.find("ERROR [GRAPH_SCHEMA] node \"od\" \"params\": "
                       "\"class_names\" must be an array of "
                       "strings for model \"yolov8n\", got \"person\"") == 0);
    m = ValidationError(ParamsGraph("yolov8n", "{\"label_file\":[\"a\"]}"), registry);
    GRAPH_CHECK(m.find("\"label_file\" must be a string for model \"yolov8n\", got [\"a\"]") !=
                std::string::npos);
    // A number for a text key is shown as written: 5, not 5.0.
    m = ValidationError(ParamsGraph("yolov8n", "{\"class_names\":5,\"label_file\":0.25}"),
                        registry);
    GRAPH_CHECK(m.find("\"class_names\" must be an array of strings for model \"yolov8n\", "
                       "got 5;") != std::string::npos);
    GRAPH_CHECK(m.find("\"label_file\" must be a string for model \"yolov8n\", got 0.25") !=
                std::string::npos);
    m = ValidationError(ParamsGraph("yolov8n", "{\"score_threshold\":\"0.5\"}"), registry);
    GRAPH_CHECK(m.find("\"score_threshold\" must be a number, got \"0.5\"") != std::string::npos);
    GRAPH_CHECK(m.find("model \"yolov8n\" reads text only for: class_names, label_file") !=
                std::string::npos);
    m = ValidationError(ParamsGraph("resnet50", "{\"class_names\":[\"a\"]}"), registry);
    GRAPH_CHECK(m.find("\"class_names\" must be a number, got [\"a\"]") != std::string::npos);
    GRAPH_CHECK(m.find("model \"resnet50\" reads only numbers") != std::string::npos);
    // U-77: ModelConfig now reads JSON strings as JSON does, so no character
    // is refused in a text value or a list entry.
    GRAPH_CHECK(ValidationError(ParamsGraph("yolov8n",
        "{\"class_names\":[\"a\\\"b\",\"c]\",\"d[\"],\"label_file\":\"e\\\\f[g]\"}"),
        registry).empty());
}

void TestBuildHandsTextAndListParamsToTheRegistry() {
    FakeModelRegistry registry = BuildParamsRegistry();
    const GraphSpec spec = ParseGraphText(ParamsGraph("yolov8n",
        "{\"class_names\":[\"pedestrian\"],\"score_threshold\":0.3}"), "p.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const StageParams* seen = registry.last_params("yolov8n");
    GRAPH_CHECK(seen != NULL && seen->lists.count("class_names") == 1 &&
                seen->numeric.count("score_threshold") == 1);
}

// Review Focus 4: a numeric-only overlay is today's text, byte for byte.
void TestParamsToJsonKeepsTheNumericBytesAndAddsTextAndLists() {
    StageParams numeric_only;
    numeric_only.numeric["score_threshold"] = 0.7;
    numeric_only.numeric["num_classes"] = 3;
    const std::map<std::string, std::string> no_arrays;
    GRAPH_CHECK(detail::ParamsToJson(numeric_only, no_arrays) ==
                "{\"num_classes\": 3, \"score_threshold\": 0.69999999999999996}");
    GRAPH_CHECK(detail::ParamsToJson(numeric_only.numeric) ==
                detail::ParamsToJson(numeric_only, no_arrays));
    StageParams mixed = numeric_only;
    mixed.text["label_file"] = "labels.txt";
    mixed.lists["class_names"].push_back("car");
    mixed.lists["class_names"].push_back("bus");
    ModelConfig config(detail::ParamsToJson(mixed, no_arrays), ConfigSource::kText);
    GRAPH_CHECK(config.get<std::string>("label_file", "") == "labels.txt");
    const std::vector<std::string> names = config.get_string_list("class_names");
    GRAPH_CHECK(names.size() == 2 && names[0] == "car" && names[1] == "bus");
    GRAPH_CHECK(config.get<int>("num_classes", 0) == 3);
}

// Review Focus 4: the node overlay must not wipe config.json's lists.
void TestParamsOverlayKeepsConfigJsonArrays() {
    const ModelConfig file("{\"class_names\": [\"a\", \"b\"], \"score_threshold\": 0.2}",
                           ConfigSource::kText);
    StageParams params;
    params.numeric["score_threshold"] = 0.5;
    const ModelConfig node(detail::ParamsToJson(params, file.rawArrays()), ConfigSource::kText);
    GRAPH_CHECK(node.get_string_list("class_names").size() == 2);
    StageParams replace;
    replace.lists["class_names"].push_back("z");
    const ModelConfig node2(detail::ParamsToJson(replace, file.rawArrays()), ConfigSource::kText);
    GRAPH_CHECK(node2.get_string_list("class_names").size() == 1);
    GRAPH_CHECK(detail::ParamsToJson(StageParams(), file.rawArrays()).empty());
}

// U-77: every character survives ParamsToJson -> ModelConfig, and a node
// list still replaces config.json's list of the same key.
void TestParamsToJsonRoundTripsEveryCharacterThroughModelConfig() {
    const ModelConfig file("{\"class_names\": [\"a\", \"b\"], \"colors\": [\"x]\"]}",
                           ConfigSource::kText);
    StageParams params;
    params.numeric["score_threshold"] = 0.5;
    params.text["label_file"] = "x\"y\\z[w]{v}:u,t\ttab\nline \xc3\xa9";
    params.lists["class_names"].push_back("c]");
    params.lists["class_names"].push_back("d\"e");
    params.lists["class_names"].push_back("f\\g");
    const ModelConfig node(detail::ParamsToJson(params, file.rawArrays()), ConfigSource::kText);
    GRAPH_CHECK(node.get<float>("score_threshold", 0.f) == 0.5f);
    GRAPH_CHECK(node.get<std::string>("label_file", "") == params.text["label_file"]);
    GRAPH_CHECK(node.get_string_list("class_names") == params.lists["class_names"]);
    const std::vector<std::string> colors = node.get_string_list("colors");  // copied from the file
    GRAPH_CHECK(colors.size() == 1 && colors[0] == "x]");
}

// U-77: a control character with no short escape is written as \u00XX and
// read back as itself, and a key that needs escaping survives the overlay.
void TestParamsToJsonEscapesControlCharactersAndKeys() {
    GRAPH_CHECK(detail::JsonQuote(std::string("a\x01" "b\x1f", 4)) == "\"a\\u0001b\\u001f\"");
    StageParams params;
    params.text["label_file"] = std::string("x\x01y", 3);
    params.numeric["a\"b"] = 2;
    const ModelConfig node(detail::ParamsToJson(params, std::map<std::string, std::string>()),
                           ConfigSource::kText);
    GRAPH_CHECK(node.get<std::string>("label_file", "") == params.text["label_file"]);
    GRAPH_CHECK(node.get<int>("a\"b", 0) == 2);
}

void TestParseGivesTheSwitchHintOnce() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(ParamsGraph("yolov8n", "{\"a\":true,\"b\":false}"),
                               GraphErrorCode::kGraphSchema, &message));
    const std::string hint = "write a switch as 1 or 0";
    const std::size_t first = message.find(hint);
    GRAPH_CHECK(first != std::string::npos &&
                message.find(hint, first + hint.size()) == std::string::npos);
}

void TestRealRegistryKnowsWhichKeysAreText() {
    StaticModelRegistry registry;
    const ModelInfo* od = registry.find("yolov8n");
    GRAPH_CHECK(od != NULL && od->text_params.size() == 1);
    if (od != NULL && od->text_params.size() == 1) {
        GRAPH_CHECK(od->text_params[0].name == "class_names" &&
                    od->text_params[0].kind == ParamInfo::kTextList);
    }
    GRAPH_CHECK(registry.find("resnet50") != NULL &&
                registry.find("resnet50")->text_params.empty());
}

// Review Focus 5: the tiled stage keeps the IStage contract.
void TestRealTiledSrStageHonoursTheStageContract() {
    const char* kTest = "TestRealTiledSrStageHonoursTheStageContract";
    StaticModelRegistry registry;
    std::vector<std::string> models(1, "espcn_x4");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) { Skip(kTest, missing); return; }
    const cv::Mat frame = cv::imread(ProjectRoot() + "/sample/img/sample_lowres275x150.png");
    GRAPH_CHECK(!frame.empty());
    if (frame.empty()) return;
    const ModelInfo* info = registry.find("espcn_x4");
    const std::string path = ModelDir() + "/" + info->dxnn_file;
    std::unique_ptr<IStage> stage = registry.createStage("espcn_x4", path, StageParams());
    StageInput input;
    input.image = frame;
    const StageResult ran = stage->run(input);
    const ImageData* image = dynamic_cast<const ImageData*>(ran.data.get());
    GRAPH_CHECK(image != NULL);
    if (image == NULL) return;
    GRAPH_CHECK(image->image.cols == 1100 && image->image.rows == 600 &&
                image->image.type() == CV_8UC3);

    std::mutex lock;
    int delivered = 0;
    bool same = true;
    for (int i = 0; i < 3; ++i) {
        stage->submit(input, [&](const StageResult& r, const std::string& error) {
            const ImageData* d = dynamic_cast<const ImageData*>(r.data.get());
            const bool match = error.empty() && d != NULL &&
                               cv::norm(d->image, image->image, cv::NORM_INF) == 0;
            // Still inside the callback, and not holding `lock`: a flush()
            // that returned before this callback finished reads 2, not 3.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            std::lock_guard<std::mutex> guard(lock);
            if (!match) same = false;
            ++delivered;
        });
    }
    stage->flush();
    {
        std::lock_guard<std::mutex> guard(lock);
        GRAPH_CHECK(delivered == 3);   // flush() returned only after every callback
        GRAPH_CHECK(same);
    }

    int late = 0;
    for (int i = 0; i < 4; ++i) {
        stage->submit(input, [&](const StageResult&, const std::string&) {
            std::lock_guard<std::mutex> guard(lock);
            ++late;
        });
    }
    stage.reset();                 // clause (5): the destructor delivers, then returns
    GRAPH_CHECK(late == 4);

    StageInput empty;
    std::unique_ptr<IStage> again = registry.createStage("espcn_x4", path, StageParams());
    bool threw = false;
    try { again->run(empty); } catch (const std::runtime_error& e) {
        threw = std::string(e.what()) == "stage received an empty image";
    }
    GRAPH_CHECK(threw);

    // submit() never throws: the same failure reaches the callback, with no data.
    std::string submit_error;
    bool submit_data = true;
    again->submit(empty, [&](const StageResult& r, const std::string& error) {
        std::lock_guard<std::mutex> guard(lock);
        submit_error = error;
        submit_data = static_cast<bool>(r.data);
    });
    again->flush();
    {
        std::lock_guard<std::mutex> guard(lock);
        GRAPH_CHECK(submit_error == "stage received an empty image");
        GRAPH_CHECK(!submit_data);
    }

    StageParams bad;
    bad.numeric["sr_tile_halo"] = 9;
    std::string text;
    try {
        registry.createStage("espcn_x4", path, bad);
    } catch (const std::runtime_error& e) {
        text = e.what();
    }
    GRAPH_CHECK(text.find("SR tile halo 9") != std::string::npos);
}

void TestRealTiledSrHandOffMatchesSync() {
    const char* kTest = "TestRealTiledSrHandOffMatchesSync";
    StaticModelRegistry registry;
    std::vector<std::string> models;
    models.push_back("espcn_x4");
    models.push_back("yolov8n");
    const std::string missing = MissingArtifacts(registry, models);
    if (!missing.empty()) { Skip(kTest, missing); return; }
    const cv::Mat frame = cv::imread(ProjectRoot() + "/sample/img/sample_lowres275x150.png");
    GRAPH_CHECK(!frame.empty());
    if (frame.empty()) return;
    const GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"x\"},"
        "{\"id\":\"sr\",\"model\":\"espcn_x4\"},{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"sr\"},{\"from\":\"sr\",\"to\":\"od\"}]}", "sr.json");
    ValidateGraph(spec, registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, registry, ModelDir());
    SyncExecutor sync_executor;
    const FrameReport expected = sync_executor.RunFrame(sync_graph, frame, 0);
    StageGraph async_graph;
    async_graph.Build(spec, registry, ModelDir());
    AsyncExecutor async_executor(StageJobs(4));
    const FrameReport actual = async_executor.RunFrame(async_graph, frame, 0);
    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    const ImageData* sr = dynamic_cast<const ImageData*>(expected.node_results.at("sr").data.get());
    GRAPH_CHECK(sr != NULL && sr->image.cols == 1100 && sr->image.rows == 600);
    GRAPH_CHECK(expected.node_results.at("od").origin.inv_align(0, 0) == 0.25f);
    GRAPH_CHECK(expected.node_results.at("od").origin.inv_align(1, 1) == 0.25f);
}

// U-36: the double-delivery guard's id rules, without an engine.
struct FakePendingJob {
    std::uintptr_t id;
    bool delivered;
    FakePendingJob() : id(0), delivered(false) {}
};

void TestPendingJobsIssueEachIdOnceAndClaimEachJobOnce() {
    detail::PendingJobs<FakePendingJob> jobs;
    std::shared_ptr<FakePendingJob> a(new FakePendingJob());
    std::shared_ptr<FakePendingJob> b(new FakePendingJob());
    GRAPH_CHECK(jobs.Add(a) == 1u);
    GRAPH_CHECK(jobs.Add(b) == 2u);
    GRAPH_CHECK(a->id == 1u && b->id == 2u);
    GRAPH_CHECK(!jobs.Claim(0));        // a null userArg never matches
    GRAPH_CHECK(jobs.Claim(1) == a);    // the first fire claims the job
    GRAPH_CHECK(!jobs.Claim(1));        // a second fire for it is dropped
    GRAPH_CHECK(!jobs.Abandon(1));      // claimed: the completion path owns it
    jobs.Erase(1);                      // finished
    GRAPH_CHECK(!jobs.Claim(1));        // a late fire finds nothing
    std::shared_ptr<FakePendingJob> c(new FakePendingJob());
    GRAPH_CHECK(jobs.Add(c) == 3u);     // 1 is free, and still never reissued
    GRAPH_CHECK(jobs.Abandon(2));       // never claimed: removed
    GRAPH_CHECK(!jobs.Claim(2));
    GRAPH_CHECK(jobs.size() == 1u);
}

// =====================================================================
// Release port (C5): the per-variant registry, its aliases and resources,
// the variant config path, the container version, and the result fields
// the 8d0b748 tree added.
// =====================================================================
namespace {

/// A file of the given bytes under ScratchPath; returns its path.
std::string WriteScratchBytes(const std::string& name, const std::string& bytes) {
    const std::string path = ScratchPath(name);
    std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    return path;
}

std::string ContainerHeader(unsigned version) {
    std::string bytes("DXNN");
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<char>((version >> (8 * i)) & 0xffu));
    return bytes + "{\"version\":1}";
}

}  // namespace

void TestReadDxnnContainerVersion() {
    uint32_t version = 0;
    std::string error;
    const std::string v8 = WriteScratchBytes("v8.dxnn", ContainerHeader(8));
    GRAPH_CHECK(ReadDxnnContainerVersion(v8, &version, &error) && version == 8 && error.empty());
    const std::string v9 = WriteScratchBytes("v9.dxnn", ContainerHeader(9));
    GRAPH_CHECK(ReadDxnnContainerVersion(v9, &version, &error) && version == 9);
    // Little-endian: byte 5 is the second byte of the number.
    const std::string big = WriteScratchBytes("big.dxnn", ContainerHeader(0x0102u));
    GRAPH_CHECK(ReadDxnnContainerVersion(big, &version, &error) && version == 0x0102u);

    version = 77;
    const std::string magic =
        WriteScratchBytes("magic.dxnn", std::string("ONNX\x08\0\0\0 not a dxnn", 19));
    GRAPH_CHECK(!ReadDxnnContainerVersion(magic, &version, &error));
    GRAPH_CHECK(error.find("DXNN") != std::string::npos && version == 77);
    const std::string short_file = WriteScratchBytes("short.dxnn", std::string("DXNN\x08\0", 6));
    error.clear();
    GRAPH_CHECK(!ReadDxnnContainerVersion(short_file, &version, &error));
    GRAPH_CHECK(error.find("short") != std::string::npos && version == 77);
    error.clear();
    GRAPH_CHECK(!ReadDxnnContainerVersion(ScratchPath("absent.dxnn"), &version, &error));
    GRAPH_CHECK(error.find("absent.dxnn") != std::string::npos);
    std::remove(v8.c_str());
    std::remove(v9.c_str());
    std::remove(big.c_str());
    std::remove(magic.c_str());
    std::remove(short_file.c_str());
}

// Spec section 4: the one text a v9 file on DX-RT < 3.5.0 gets, from bytes
// 4-7 and the runtime's own version string, before any engine exists.
void TestContainerSupportRefusesV9BelowDxrt350() {
    const std::string v9_on_341 =
        ".dxnn container v9 needs DX-RT >= 3.5.0, but this runtime is 3.4.1. "
        "Use the v8 file (dxnn/2_4_0) or upgrade DX-RT.";
    GRAPH_CHECK(ContainerSupportError(8, "v3.4.1").empty());
    GRAPH_CHECK(ContainerSupportError(6, "3.4.1").empty());
    GRAPH_CHECK(ContainerSupportError(9, "v3.4.1") == v9_on_341);
    // dxrt::Configuration::GetVersion() prints "3.4.1"; dxrt-cli "v3.4.1+baec914".
    GRAPH_CHECK(ContainerSupportError(9, "3.4.1") == v9_on_341);
    GRAPH_CHECK(ContainerSupportError(9, "v3.4.1+baec914") == v9_on_341);
    GRAPH_CHECK(ContainerSupportError(9, "3.4.9").find("needs DX-RT >= 3.5.0") != std::string::npos);
    GRAPH_CHECK(ContainerSupportError(9, "2.9.0").find("needs DX-RT >= 3.5.0") != std::string::npos);
    GRAPH_CHECK(ContainerSupportError(9, "v3.5.0").empty());
    GRAPH_CHECK(ContainerSupportError(9, "3.10.0").empty());  // numeric, not text order
    GRAPH_CHECK(ContainerSupportError(9, "4.0").empty());
    // A container newer than v9: no DX-RT floor is known yet, so nothing is
    // refused here on any runtime; dxrt decides.
    GRAPH_CHECK(ContainerSupportError(10, "v3.5.0").empty());
    GRAPH_CHECK(ContainerSupportError(10, "3.4.1").empty());
    GRAPH_CHECK(ContainerSupportError(10, "4.0").empty());
    GRAPH_CHECK(ContainerRequirement(10, "3.4.1").empty());
    // A runtime string this cannot read: no verdict; dxrt decides.
    GRAPH_CHECK(ContainerSupportError(9, "").empty());
    GRAPH_CHECK(ContainerSupportError(9, "unknown").empty());
    GRAPH_CHECK(ContainerSupportError(9, "v3").empty());
    GRAPH_CHECK(ContainerSupportError(9, "3.x.1").empty());

    // The short form --list-models and --check print next to the file.
    GRAPH_CHECK(ContainerRequirement(9, "3.4.1") == "needs DX-RT >= 3.5.0");
    GRAPH_CHECK(ContainerRequirement(8, "3.4.1").empty());
    GRAPH_CHECK(ContainerRequirement(9, "3.5.0").empty());

    // On a file: "<file>: <text>"; a v8 file, or one the header reader
    // refuses (dxrt names that problem itself), gives "".
    const std::string v9 = WriteScratchBytes("support_v9.dxnn", ContainerHeader(9));
    const std::string v8 = WriteScratchBytes("support_v8.dxnn", ContainerHeader(8));
    GRAPH_CHECK(ContainerLoadError(v9, "3.4.1") == v9 + ": " + v9_on_341);
    GRAPH_CHECK(ContainerLoadError(v9, "3.5.0").empty());
    GRAPH_CHECK(ContainerLoadError(v8, "3.4.1").empty());
    GRAPH_CHECK(ContainerLoadError(ScratchPath("support_absent.dxnn"), "3.4.1").empty());
    std::remove(v9.c_str());
    std::remove(v8.c_str());
}

// A registry whose stages check the .dxnn container against a fixed DX-RT
// version ("3.4.1") instead of the running one, before the real registry
// makes the stage: Build's MODEL_LOAD path for a v9 file is then the same on
// every runtime, and no engine is opened on the header-only stub.
namespace {
class PinnedRuntimeRegistry : public IModelRegistry {
 public:
    PinnedRuntimeRegistry(const IModelRegistry& inner, const std::string& runtime)
        : inner_(inner), runtime_(runtime) {}
    const ModelInfo* find(const std::string& model_name) const override {
        return inner_.find(model_name);
    }
    std::vector<ModelInfo> list() const override { return inner_.list(); }
    std::vector<ModelAlias> aliases() const override { return inner_.aliases(); }
    std::unique_ptr<IStage> createStage(const std::string& model_name,
                                        const std::string& model_path,
                                        const StageParams& params) const override {
        return inner_.createStage(
            model_name, dxapp::graph::detail::LoadableModelPath(model_path, runtime_), params);
    }

 private:
    const IModelRegistry& inner_;
    std::string runtime_;
};
}  // namespace

// A v9 file is refused before an engine is opened (spec section 4, R12).
// LoadableModelPath is checked against DX-RT 3.4.1 on every runtime; the
// real stage makers - a typed stage (yolov8-n) and a restoration stage
// (espcn-x2, the tiled-SR maker) - only where the running DX-RT refuses v9
// itself, since on 3.5.0 they would open an engine on the header-only stub.
void TestAStageRefusesAV9FileBeforeOpeningAnEngine() {
    const std::string pinned = "3.4.1";
    const std::string v9_on_341 = ContainerSupportError(9, pinned);
    const std::string v9_hint = "use the v8 file (dxnn/2_4_0) or upgrade DX-RT to >= 3.5.0";
    GRAPH_CHECK(!v9_on_341.empty());
    const std::string v9 = WriteScratchBytes("stage_v9.dxnn", ContainerHeader(9));
    const std::string v8 = WriteScratchBytes("stage_v8.dxnn", ContainerHeader(8));
    {
        std::string message;
        std::string hint;
        try {
            dxapp::graph::detail::LoadableModelPath(v9, pinned);
        } catch (const ModelContainerError& error) {
            message = error.what();
            hint = error.hint();
        }
        GRAPH_CHECK(message == v9 + ": " + v9_on_341);
        GRAPH_CHECK(hint == v9_hint);
        GRAPH_CHECK(dxapp::graph::detail::LoadableModelPath(v9, "3.5.0") == v9);
        GRAPH_CHECK(dxapp::graph::detail::LoadableModelPath(v8, pinned) == v8);
    }
    std::remove(v8.c_str());

    StaticModelRegistry registry;
    const std::string running = dxrt::Configuration::GetInstance().GetVersion();
    const std::string expected = ContainerSupportError(9, running);
    if (!expected.empty()) {
        const char* models[] = {"yolov8-n_640x640", "espcn-x2_17x17"};
        for (std::size_t m = 0; m < sizeof(models) / sizeof(models[0]); ++m) {
            std::string container_message;
            std::string hint;
            try {
                registry.createStage(models[m], v9, StageParams());
            } catch (const ModelContainerError& error) {
                // I3: the registry passes the container refusal through as
                // its own type, so Build can tell it from a device failure.
                container_message = error.what();
                hint = error.hint();
            } catch (const std::exception&) {
            }
            GRAPH_CHECK(container_message == v9 + ": " + expected);
            GRAPH_CHECK(hint == v9_hint);
        }
    }
    std::remove(v9.c_str());

    // StageGraph::Build turns it into MODEL_LOAD with that hint and no
    // download line: downloading again would fetch the same v9 file.
    const std::string dir = ScratchPath("v9_models");
    ::mkdir(dir.c_str(), 0700);
    const std::string file = dir + "/yolov8-n_640x640.dxnn";
    {
        std::ofstream out(file.c_str(), std::ios::binary | std::ios::trunc);
        const std::string header = ContainerHeader(9);
        out.write(header.data(), static_cast<std::streamsize>(header.size()));
    }
    const GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"name\":\"v9\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"x.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}",
        "v9.json");
    const PinnedRuntimeRegistry on_341(registry, pinned);
    std::string built;
    GraphErrorCode code = GraphErrorCode::kGraphSchema;
    try {
        StageGraph graph;
        graph.Build(spec, on_341, dir);
    } catch (const GraphError& error) {
        built = error.what();
        code = error.code();
    }
    GRAPH_CHECK(code == GraphErrorCode::kModelLoad);
    GRAPH_CHECK(built.find("ERROR [MODEL_LOAD] node \"od\": model \"yolov8n\" "
                           "(yolov8-n_640x640.dxnn) could not be loaded: " + file + ": " +
                           v9_on_341) == 0);
    GRAPH_CHECK(built.find("\n  -> " + v9_hint) != std::string::npos);
    GRAPH_CHECK(built.find("setup.sh") == std::string::npos);
    GRAPH_CHECK(built.find("download_models") == std::string::npos);
    GRAPH_CHECK(built.find("dxrt-cli -s") == std::string::npos);
    std::remove(file.c_str());
    ::rmdir(dir.c_str());
}

// R6: the model key is the variant. A legacy model_name and an alias_of row
// both resolve to the variant's own ModelInfo; neither is listed as a model.
void TestRegistryResolvesOldNamesAndAliasesToTheVariant() {
    StaticModelRegistry registry;
    const ModelInfo* variant = registry.find("yolov8-n_640x640");
    GRAPH_CHECK(variant != NULL);
    if (variant == NULL) return;
    GRAPH_CHECK(registry.find("yolov8n") == variant);
    GRAPH_CHECK(variant->model_name == "yolov8-n_640x640" && variant->variant == variant->model_name);
    GRAPH_CHECK(variant->family == "yolov8" && variant->task == "object_detection");
    const ModelInfo* deit = registry.find("deit_base384_distilled");
    GRAPH_CHECK(deit != NULL && deit->model_name == "deit-b_384x384_distilled");
    GRAPH_CHECK(registry.find("no_such_model_anywhere") == NULL);

    const std::vector<ModelInfo> all = registry.list();
    std::map<std::string, const ModelInfo*> keys;
    for (std::size_t i = 0; i < all.size(); ++i) {
        GRAPH_CHECK(all[i].model_name == all[i].variant);
        keys[all[i].model_name] = &all[i];
    }
    GRAPH_CHECK(keys.size() == all.size());
    const std::vector<ModelAlias> aliases = registry.aliases();
    GRAPH_CHECK(aliases.size() > 400);
    int alias_of = 0;
    bool deit_listed = false;
    bool yolov8n_legacy = false;
    for (std::size_t i = 0; i < aliases.size(); ++i) {
        GRAPH_CHECK(keys.count(aliases[i].name) == 0);        // never a second model
        GRAPH_CHECK(keys.count(aliases[i].variant) == 1);     // always names a model
        const ModelInfo* found = registry.find(aliases[i].name);
        GRAPH_CHECK(found != NULL && found->model_name == aliases[i].variant);
        if (aliases[i].kind == ModelAlias::kAliasOf) ++alias_of;
        if (aliases[i].name == "deit_base384_distilled") {
            deit_listed = aliases[i].kind == ModelAlias::kAliasOf &&
                          aliases[i].variant == "deit-b_384x384_distilled";
        }
        if (aliases[i].name == "yolov8n") yolov8n_legacy = aliases[i].kind == ModelAlias::kLegacyName;
    }
    GRAPH_CHECK(alias_of >= 1 && deit_listed && yolov8n_legacy);

    // An alias builds its variant's stage, and the refusal names the alias.
    // A legacy name of a not-ready variant, so it is refused before any
    // engine: the refusal comes from the variant's row, under the alias.
    const char* alias = "efficientad_m_teacher_256x256";
    const ModelInfo* teacher = registry.find(alias);
    GRAPH_CHECK(teacher != NULL && teacher->model_name == "efficientad-m-teacher_256x256" &&
                !teacher->ready);
    std::string refusal;
    try {
        registry.createStage(alias, "/nonexistent.dxnn", StageParams());
    } catch (const std::runtime_error& error) {
        refusal = error.what();
    }
    GRAPH_CHECK(refusal.find(std::string("model \"") + alias + "\" is not usable in a graph: ") ==
                0);
    GRAPH_CHECK(teacher == NULL || refusal.find(teacher->not_ready_reason) != std::string::npos);
}

// ModelInfo carries the registry's published flag and R9's resources.
void TestRegistryCarriesPublishedAndResources() {
    StaticModelRegistry registry;
    const ModelInfo* vpr = registry.find("eigenplaces-resnet18_512x512");
    GRAPH_CHECK(vpr != NULL && vpr->ready && vpr->published);
    if (vpr != NULL) {
        GRAPH_CHECK(vpr->resources.size() == 1 && vpr->resources[0].kind == ResourceInfo::kGallery &&
                    vpr->resources[0].value == "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin");
    }
    const ModelInfo* teacher = registry.find("efficientad-m-teacher_256x256");
    GRAPH_CHECK(teacher != NULL && !teacher->ready);
    if (teacher != NULL) {
        int companions = 0;
        for (std::size_t i = 0; i < teacher->resources.size(); ++i) {
            if (teacher->resources[i].kind == ResourceInfo::kCompanion) ++companions;
        }
        GRAPH_CHECK(companions == 2);
    }
    const std::vector<ModelInfo> all = registry.list();
    int zero_shot = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (all[i].task != "zero_shot_image_classification") continue;
        ++zero_shot;
        GRAPH_CHECK(all[i].resources.size() == 1 && all[i].resources[0].kind == ResourceInfo::kNote &&
                    all[i].resources[0].value.find("Python-only") != std::string::npos);
    }
    GRAPH_CHECK(zero_shot > 0);
    const ModelInfo* plain = registry.find("yolov8-n_640x640");
    GRAPH_CHECK(plain != NULL && plain->resources.empty());
}

// A published model's missing file is offered a download. vit-l-p16_512x512_swag
// used to be the unpublished exception; the zoo now serves it.
void TestBuildOffersADownloadOfAPublishedModel() {
    StaticModelRegistry registry;
    const ModelInfo* swag = registry.find("vit-l-p16_512x512_swag");
    GRAPH_CHECK(swag != NULL && swag->published);
    const char* models[] = {"vit-l-p16_512x512_swag", "yolov8n"};
    const char* downloads[] = {"vit_l_p16_512x512_swag", "YoloV8N"};
    for (std::size_t m = 0; m < 2; ++m) {
        const GraphSpec spec = ParseGraphText(
            std::string("{\"version\":1,\"name\":\"m\",\"nodes\":["
                        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"x.jpg\"},"
                        "{\"id\":\"n\",\"model\":\"") +
                models[m] + "\"}],\"edges\":[{\"from\":\"cam\",\"to\":\"n\"}]}",
            "m.json");
        std::string message;
        try {
            StageGraph graph;
            graph.Build(spec, registry, "/nonexistent_model_dir");
        } catch (const GraphError& error) {
            GRAPH_CHECK(error.code() == GraphErrorCode::kModelMissing);
            message = error.what();
        }
        GRAPH_CHECK(message.find(std::string("\n  -> python3 scripts/download_models.py --models ") +
                                 downloads[m] + " --output /nonexistent_model_dir") !=
                    std::string::npos);
        GRAPH_CHECK(message.find("is not published") == std::string::npos);
    }
}

// TypedStage and TiledSrStage read <task>/<family>/<variant>/config.json,
// through C3's nested "config" overlay.
void TestStageConfigIsTheVariantsConfigJson() {
    StaticModelRegistry registry;
    const ModelInfo* pose = registry.find("yolov8-s-pose_640x640");
    GRAPH_CHECK(pose != NULL);
    if (pose == NULL) return;
    const std::string path = detail::StageConfigPath(*pose);
    GRAPH_CHECK(path == ProjectRoot() +
                            "/src/cpp_example/pose_estimation/yolov8_pose/"
                            "yolov8-s-pose_640x640/config.json");
    const ModelConfig config = LoadOptionalConfig(path);
    GRAPH_CHECK(config.isLoaded());
    GRAPH_CHECK(std::fabs(config.get<float>("score_threshold", -1.f) - 0.3f) < 1e-6f);
    GRAPH_CHECK(std::fabs(config.get<float>("nms_threshold", -1.f) - 0.45f) < 1e-6f);

    const std::vector<ModelInfo> all = registry.list();
    int ready = 0;
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (!all[i].ready) continue;
        ++ready;
        const std::string config_path = detail::StageConfigPath(all[i]);
        if (!FileIsReadable(config_path)) std::printf("no config: %s\n", config_path.c_str());
        GRAPH_CHECK(FileIsReadable(config_path));
    }
    GRAPH_CHECK(ready > 400);
}

// M6: a ModelInfo without family or variant names no config.json. Reading
// "<task>//config.json" would silently leave the factory on its defaults,
// so StageConfigPath refuses it.
void TestStageConfigPathRefusesAModelInfoWithoutFamilyOrVariant() {
    ModelInfo info;
    info.model_name = "hand_built";
    info.task = "object_detection";
    info.family = "yolov8";
    info.variant = "yolov8-n_640x640";
    GRAPH_CHECK(detail::StageConfigPath(info) ==
                ProjectRoot() + "/src/cpp_example/object_detection/yolov8/yolov8-n_640x640/config.json");
    const std::string expected =
        "ModelInfo for \"hand_built\" has no family/variant: cannot locate its config.json";
    for (int missing = 0; missing < 2; ++missing) {
        ModelInfo partial = info;
        (missing == 0 ? partial.family : partial.variant).clear();
        std::string message;
        try {
            detail::StageConfigPath(partial);
        } catch (const std::runtime_error& error) {
            message = error.what();
        }
        GRAPH_CHECK(message == expected);
    }
}

// I2: a stage reads a relative "gallery" against the repository, as --check
// does, so a run from any working directory opens the gallery --check found.
// The real EigenPlaces factory, configured the way TypedStage configures it,
// from a working directory that is not the repository.
void TestARelativeGalleryIsReadAgainstTheRepository() {
    StaticModelRegistry registry;
    const ModelInfo* info = registry.find("eigenplaces-resnet18_512x512");
    GRAPH_CHECK(info != NULL);
    if (info == NULL) return;
    const std::string gallery = "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin";
    GRAPH_CHECK(info->resources.size() == 1 && info->resources[0].value == gallery);

    // The overlay itself: a config.json value is added, a node param is made
    // absolute, an absolute path and other params are kept.
    const ModelConfig file_config("{\"gallery\": \"" + gallery + "\", \"top_k\": 5}",
                                  ConfigSource::kText);
    const ModelConfig no_gallery("{\"top_k\": 5}", ConfigSource::kText);
    StageParams none;
    GRAPH_CHECK(detail::GalleryAgainstRoot(none, file_config, "/repo").text.at("gallery") ==
                "/repo/" + gallery);
    GRAPH_CHECK(detail::GalleryAgainstRoot(none, no_gallery, "/repo").text.empty());
    StageParams relative;
    relative.text["gallery"] = "my/g.bin";
    relative.numeric["top_k"] = 3;
    const StageParams made = detail::GalleryAgainstRoot(relative, file_config, "/repo");
    GRAPH_CHECK(made.text.at("gallery") == "/repo/my/g.bin" && made.numeric.at("top_k") == 3);
    StageParams absolute;
    absolute.text["gallery"] = "/data/g.bin";
    GRAPH_CHECK(detail::GalleryAgainstRoot(absolute, file_config, "/repo").text.at("gallery") ==
                "/data/g.bin");
    const ModelConfig absolute_file("{\"gallery\": \"/data/g.bin\"}", ConfigSource::kText);
    GRAPH_CHECK(detail::GalleryAgainstRoot(none, absolute_file, "/repo").text.empty());
    GRAPH_CHECK(detail::IsAbsolutePath("C:\\g.bin") && !detail::IsAbsolutePath("g.bin"));

    char cwd[4096];
    GRAPH_CHECK(::getcwd(cwd, sizeof(cwd)) != NULL);
    const std::string elsewhere = ScratchPath("gallery_cwd");
    ::mkdir(elsewhere.c_str(), 0700);
    GRAPH_CHECK(::chdir(elsewhere.c_str()) == 0);

    const char* given[] = {NULL, "sample/gallery/vpr_eigenplaces-resnet18_512x512.bin"};
    for (std::size_t g = 0; g < 2; ++g) {
        StageParams params;
        if (given[g] != NULL) params.text["gallery"] = given[g];  // a node param
        v_eigenplaces_resnet18_512x512::EigenplacesFactory factory;
        detail::ConfigureFactory(&factory, *info, params);
        const PostprocessorPtr<EmbeddingResult> post = factory.createPostprocessor(512, 512);
        const GalleryRetrievalPostprocessor* retrieval =
            dynamic_cast<const GalleryRetrievalPostprocessor*>(post.get());
        GRAPH_CHECK(retrieval != NULL);
        if (retrieval == NULL) continue;
        if (!retrieval->galleryError().empty()) {
            std::printf("gallery: %s\n", retrieval->galleryError().c_str());
        }
        GRAPH_CHECK(retrieval->galleryError().empty());
    }
    GRAPH_CHECK(::chdir(cwd) == 0);
    ::rmdir(elsewhere.c_str());
}

// I18: --check and the stage overlay read a repository-relative path by one
// rule (common/utility/repo_path.hpp): absolute as given, relative joined to
// PROJECT_ROOT_DIR, empty left empty.
void TestResolveRepoRelativeKeepsAbsoluteAndJoinsRelative() {
    const std::string root = PROJECT_ROOT_DIR;
    GRAPH_CHECK(ResolveRepoRelative("/data/g.bin") == "/data/g.bin");
    GRAPH_CHECK(ResolveRepoRelative("C:\\data\\g.bin") == "C:\\data\\g.bin");
    GRAPH_CHECK(ResolveRepoRelative("sample/gallery/g.bin") == root + "/sample/gallery/g.bin");
    GRAPH_CHECK(ResolveRepoRelative("").empty());
    GRAPH_CHECK(ResolveAgainstRoot("/repo", "my/g.bin") == "/repo/my/g.bin");
    GRAPH_CHECK(ResolveAgainstRoot("/repo", "/data/g.bin") == "/data/g.bin");
    GRAPH_CHECK(ResolveAgainstRoot("/repo", "").empty());
}

// PP-Matting's soft alpha matte is the "alpha" port: a spatial CV_32F map.
// The primary output stays the thresholded class map.
void TestMattingAlphaBecomesADenseMapPort() {
    std::vector<SegmentationResult> results(1);
    results[0].width = 3;
    results[0].height = 2;
    results[0].mask.assign(6, 1);
    results[0].alpha = cv::Mat(2, 3, CV_32F, cv::Scalar(0.25f));
    results[0].alpha.at<float>(1, 2) = 1.f;
    const StagePorts ports = ToStagePorts(results);
    GRAPH_CHECK(ports.size() == 1);
    const DenseMapData* alpha = dynamic_cast<const DenseMapData*>(ports.at("alpha").get());
    GRAPH_CHECK(alpha != NULL);
    if (alpha == NULL) return;
    GRAPH_CHECK(alpha->spatial);
    GRAPH_CHECK(alpha->values.rows == 2 && alpha->values.cols == 3 &&
                alpha->values.type() == CV_32F);
    GRAPH_CHECK(alpha->values.at<float>(1, 2) == 1.f && alpha->values.at<float>(0, 0) == 0.25f);
    GRAPH_CHECK(alpha->values.data != results[0].alpha.data);   // a copy, not a view
    GRAPH_CHECK(ToStageData(results[0])->shape() == Shape::kLabelMap);

    std::vector<SegmentationResult> class_map(1);
    const StagePorts class_map_ports = ToStagePorts(class_map);
    const DenseMapData* none =
        dynamic_cast<const DenseMapData*>(class_map_ports.at("alpha").get());
    GRAPH_CHECK(none != NULL && none->values.empty());
    GRAPH_CHECK(ToStagePorts(std::vector<SegmentationResult>()).at("alpha") != NULL);
}

// A payload outlives the stage's output tensors: a report is serialized
// after later frames reuse them, and a dx_graph report after its graph (and
// every engine) is closed. 8d0b748's FastDepthPostprocessor returns a
// depth_map that still points into the dxrt tensor (its MatExpr assignment
// writes in place), so the conversion must copy.
void TestDepthConversionOwnsItsValues() {
    float tensor[4] = {0.f, 0.25f, 0.5f, 1.f};
    DepthResult result;
    result.depth_map = cv::Mat(2, 2, CV_32F, tensor);
    const StageDataPtr data = ToStageData(result);
    const DenseMapData* dense = dynamic_cast<const DenseMapData*>(data.get());
    GRAPH_CHECK(dense != NULL);
    if (dense == NULL) return;
    GRAPH_CHECK(dense->values.data != reinterpret_cast<uchar*>(tensor));
    tensor[3] = 9.f;   // the runtime reuses its buffer
    GRAPH_CHECK(dense->values.at<float>(1, 1) == 1.f);
}

// A gallery model's ranking is the "matches" port, one score per match,
// best first; the primary output stays the embedding.
void TestGalleryMatchesBecomeAScoresPort() {
    std::vector<EmbeddingResult> results(1);
    results[0].embedding.assign(4, 0.5f);
    GalleryMatch place;
    place.rank = 1;
    place.score = 0.9f;
    place.path = "sample/vpr/database/a.jpg";
    GalleryMatch person;
    person.rank = 2;
    person.score = 0.4f;
    person.path = "sample/reid/gallery/7/b.jpg";
    person.label = "7";
    results[0].matches.push_back(place);
    results[0].matches.push_back(person);
    const StagePorts ports = ToStagePorts(results);
    GRAPH_CHECK(ports.size() == 1);
    const ScoresData* matches = dynamic_cast<const ScoresData*>(ports.at("matches").get());
    GRAPH_CHECK(matches != NULL && matches->items.size() == 2);
    if (matches == NULL || matches->items.size() != 2) return;
    GRAPH_CHECK(matches->items[0].class_id == 1 && matches->items[0].confidence == 0.9f &&
                matches->items[0].class_name == "sample/vpr/database/a.jpg");
    GRAPH_CHECK(matches->items[1].class_id == 2 && matches->items[1].class_name == "7");
    const StageDataPtr primary = ToStageData(results[0]);
    const VectorData* vector = dynamic_cast<const VectorData*>(primary.get());
    GRAPH_CHECK(vector != NULL && vector->values.size() == 4);

    std::vector<EmbeddingResult> plain(1);
    const StagePorts plain_ports = ToStagePorts(plain);
    const ScoresData* none = dynamic_cast<const ScoresData*>(plain_ports.at("matches").get());
    GRAPH_CHECK(none != NULL && none->items.empty());
    GRAPH_CHECK(ToStagePorts(std::vector<EmbeddingResult>()).at("matches") != NULL);
}

bool CpuValidateThrows(const std::string& text, const IModelRegistry& registry,
                       GraphErrorCode expected, std::string* message) {
    try {
        const GraphSpec spec = ParseGraphText(text, "cpu.json");
        ValidateGraph(spec, registry);
    } catch (const GraphError& error) {
        *message = error.what();
        return error.code() == expected;
    } catch (...) {
        *message = "non-GraphError exception";
        return false;
    }
    *message = "no exception";
    return false;
}

ModelInfo CpuFixtureModel(const std::string& name, Shape shape) {
    ModelInfo info;
    info.model_name = name;
    info.task = "object_detection";
    info.dxnn_file = name + ".dxnn";
    info.output_shape = shape;
    info.input_contract = InputContract::kFullFrame;
    info.ready = true;
    info.published = true;
    return info;
}

const RecordsData* RecordsOf(const FrameReport& report, const std::string& id) {
    std::map<std::string, StageResult>::const_iterator it = report.node_results.find(id);
    if (it == report.node_results.end() || !it->second.data) return NULL;
    return dynamic_cast<const RecordsData*>(it->second.data.get());
}

bool NamedNumber(const RecordItem& item, const std::string& name, double* value) {
    for (std::size_t i = 0; i < item.numbers.size(); ++i) {
        if (item.numbers[i].first != name) continue;
        *value = item.numbers[i].second;
        return true;
    }
    return false;
}

FrameReport RunSync(const std::string& text, FakeModelRegistry* registry, const cv::Mat& frame) {
    const GraphSpec spec = ParseGraphText(text, "cpu.json");
    ValidateGraph(spec, *registry);
    StageGraph graph;
    graph.Build(spec, *registry, "/models", false);
    return SyncExecutor().RunFrame(graph, frame, 0);
}

FrameReport RunAsync(const std::string& text, FakeModelRegistry* registry, const cv::Mat& frame) {
    const GraphSpec spec = ParseGraphText(text, "cpu.json");
    StageGraph graph;
    graph.Build(spec, *registry, "/models", false);
    return AsyncExecutor(StageJobs(4)).RunFrame(graph, frame, 0);
}

void TestCpuNodeRejectsAGenericFuseAndABadResultEdge() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"tilt\"}],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown cpu op") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\",\"track\":{\"algo\":\"iou\"}}],"
        "\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("cannot track") != std::string::npos);

    FakeModelRegistry registry;
    registry.AddModel(CpuFixtureModel("face", Shape::kBoxes),
                      StageDataPtr(new BoxesData(Shape::kBoxes)));
    GRAPH_CHECK(CpuValidateThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\",\"params\":{\"scale_factor\":1}}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"}]}",
        registry, GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("takes no params") != std::string::npos);

    GRAPH_CHECK(CpuValidateThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},"
        "{\"id\":\"od\",\"model\":\"face\"},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"},"
        "{\"from\":\"od\",\"to\":\"pose\",\"carry\":\"result\"}]}",
        registry, GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("one boxes result") != std::string::npos);

    GRAPH_CHECK(CpuValidateThrows(
        "{\"version\":1,\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},"
        "{\"id\":\"od\",\"model\":\"face\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"od\",\"carry\":\"result\"}]}",
        registry, GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("only by a cpu node") != std::string::npos);
}

void TestHeadPoseMatchesSyncAndAsync() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem face;
    face.box = cv::Rect2f(200.f, 120.f, 80.f, 100.f);
    face.class_id = 0;
    face.class_name = "face";
    face.landmarks.push_back(Keypoint(220.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(260.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(240.f, 180.f, 1.f));
    face.landmarks.push_back(Keypoint(225.f, 200.f, 1.f));
    face.landmarks.push_back(Keypoint(255.f, 200.f, 1.f));
    boxes->items.push_back(face);

    const std::string text =
        "{\"version\":1,\"name\":\"headpose\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"}]}";
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry;
    sync_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport expected = RunSync(text, &sync_registry, frame);

    FakeModelRegistry async_registry;
    async_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport actual = RunAsync(text, &async_registry, frame);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    const RecordsData* records = RecordsOf(expected, "pose");
    GRAPH_CHECK(records != NULL && records->items.size() == 1);
    if (records == NULL || records->items.empty()) return;
    double pitch = 0.0;
    double yaw = 0.0;
    double roll = 0.0;
    GRAPH_CHECK(NamedNumber(records->items[0], "pitch", &pitch));
    GRAPH_CHECK(NamedNumber(records->items[0], "yaw", &yaw));
    GRAPH_CHECK(NamedNumber(records->items[0], "roll", &roll));
    GRAPH_CHECK(std::isfinite(pitch) && std::isfinite(yaw) && std::isfinite(roll));
}

void TestVolumeProxyMatchesSyncAndAsync() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem package;
    package.box = cv::Rect2f(0.f, 0.f, 4.f, 4.f);
    package.class_id = 24;
    package.class_name = "backpack";
    boxes->items.push_back(package);
    BoxItem person;
    person.box = cv::Rect2f(10.f, 10.f, 4.f, 4.f);
    person.class_id = 0;
    person.class_name = "person";
    boxes->items.push_back(person);

    std::shared_ptr<BoxesData> instances(new BoxesData(Shape::kInstances));
    BoxItem segment = package;
    segment.mask = cv::Mat::zeros(4, 4, CV_8UC1);
    segment.mask.at<unsigned char>(0, 0) = 255;
    segment.mask.at<unsigned char>(0, 1) = 255;
    segment.mask.at<unsigned char>(1, 0) = 255;
    segment.mask.at<unsigned char>(1, 1) = 255;
    instances->items.push_back(segment);

    std::shared_ptr<DenseMapData> depth(new DenseMapData());
    depth->values = cv::Mat(4, 4, CV_32FC1, cv::Scalar(3.0f));

    const std::string text =
        "{\"version\":1,\"name\":\"volume\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"det\",\"model\":\"det\"},"
        "{\"id\":\"seg\",\"model\":\"seg\"},"
        "{\"id\":\"depth\",\"model\":\"depth\"},"
        "{\"id\":\"volume\",\"type\":\"cpu\",\"op\":\"volume\","
        "\"params\":{\"scale_factor\":2.0}}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"det\"},{\"from\":\"cam\",\"to\":\"seg\"},"
        "{\"from\":\"cam\",\"to\":\"depth\"},"
        "{\"from\":\"det\",\"to\":\"volume\",\"carry\":\"result\"},"
        "{\"from\":\"seg\",\"to\":\"volume\",\"carry\":\"result\"},"
        "{\"from\":\"depth\",\"to\":\"volume\",\"carry\":\"result\"}]}";
    // A full-frame instance mask is a raster of the frame: 4x4 here, as the mask.
    const cv::Mat frame = cv::Mat::zeros(4, 4, CV_8UC3);

    FakeModelRegistry sync_registry;
    sync_registry.AddModel(CpuFixtureModel("det", Shape::kBoxes), boxes);
    sync_registry.AddModel(CpuFixtureModel("seg", Shape::kInstances), instances);
    sync_registry.AddModel(CpuFixtureModel("depth", Shape::kDenseMap), depth);
    const FrameReport expected = RunSync(text, &sync_registry, frame);

    FakeModelRegistry async_registry;
    async_registry.AddModel(CpuFixtureModel("det", Shape::kBoxes), boxes);
    async_registry.AddModel(CpuFixtureModel("seg", Shape::kInstances), instances);
    async_registry.AddModel(CpuFixtureModel("depth", Shape::kDenseMap), depth);
    const FrameReport actual = RunAsync(text, &async_registry, frame);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    const RecordsData* records = RecordsOf(expected, "volume");
    GRAPH_CHECK(records != NULL && records->items.size() == 1);
    if (records == NULL || records->items.size() != 1) return;
    double area = 0.0;
    double median = 0.0;
    double volume = 0.0;
    GRAPH_CHECK(NamedNumber(records->items[0], "mask_area_px", &area) && area == 4.0);
    GRAPH_CHECK(NamedNumber(records->items[0], "median_depth", &median) && median == 3.0);
    GRAPH_CHECK(NamedNumber(records->items[0], "volume_proxy", &volume) && volume == 24.0);
    GRAPH_CHECK(records->items[0].text.size() == 1 &&
                records->items[0].text[0].second == "backpack");
}

// Phase 4: the depth model's raster (768x768, a stretch of the frame) and a segmenter's mask (the frame's own
// 500x333) both cover the frame. volume reads the depth at each mask pixel's place in the frame, as the overlay
// places both; it used to give no volume_proxy unless the two had one size, which no real model pair has.
const char kVolumeGraph[] =
    "{\"version\":1,\"name\":\"volume\",\"nodes\":["
    "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
    "{\"id\":\"det\",\"model\":\"det\"},"
    "{\"id\":\"seg\",\"model\":\"seg\"},"
    "{\"id\":\"depth\",\"model\":\"depth\"},"
    "{\"id\":\"volume\",\"type\":\"cpu\",\"op\":\"volume\"}],"
    "\"edges\":[{\"from\":\"cam\",\"to\":\"det\"},{\"from\":\"cam\",\"to\":\"seg\"},"
    "{\"from\":\"cam\",\"to\":\"depth\"},"
    "{\"from\":\"det\",\"to\":\"volume\",\"carry\":\"result\"},"
    "{\"from\":\"seg\",\"to\":\"volume\",\"carry\":\"result\"},"
    "{\"from\":\"depth\",\"to\":\"volume\",\"carry\":\"result\"}]}";

/// A 768x768 depth raster over the frame: 2 on its left half, 9 on its right.
std::shared_ptr<DenseMapData> SplitDepth768() {
    std::shared_ptr<DenseMapData> depth(new DenseMapData());
    depth->values = cv::Mat(768, 768, CV_32FC1, cv::Scalar(9.0f));
    depth->values(cv::Rect(0, 0, 384, 768)).setTo(2.0f);
    return depth;
}

/// A suitcase at `box` and its instance: a mask raster of `mask_size` with `box_in_mask` on.
void SuitcaseFixture(const cv::Rect2f& box, const cv::Size& mask_size, const cv::Rect& box_in_mask,
                     std::shared_ptr<BoxesData>* boxes, std::shared_ptr<BoxesData>* instances) {
    boxes->reset(new BoxesData(Shape::kBoxes));
    BoxItem package;
    package.box = box;
    package.class_id = 28;
    package.class_name = "suitcase";
    (*boxes)->items.push_back(package);
    instances->reset(new BoxesData(Shape::kInstances));
    BoxItem segment = package;
    segment.box = cv::Rect2f(box_in_mask);
    segment.mask = cv::Mat::zeros(mask_size, CV_8UC1);
    segment.mask(box_in_mask).setTo(255);
    (*instances)->items.push_back(segment);
}

void TestVolumeSamplesADepthMapOfAnotherSize() {
    std::shared_ptr<BoxesData> boxes;
    std::shared_ptr<BoxesData> instances;
    SuitcaseFixture(cv::Rect2f(100.f, 100.f, 50.f, 50.f), cv::Size(500, 333), cv::Rect(100, 100, 50, 50),
                    &boxes, &instances);
    const cv::Mat frame = cv::Mat::zeros(333, 500, CV_8UC3);

    FakeModelRegistry sync_registry;
    sync_registry.AddModel(CpuFixtureModel("det", Shape::kBoxes), boxes);
    sync_registry.AddModel(CpuFixtureModel("seg", Shape::kInstances), instances);
    sync_registry.AddModel(CpuFixtureModel("depth", Shape::kDenseMap), SplitDepth768());
    const FrameReport expected = RunSync(kVolumeGraph, &sync_registry, frame);

    FakeModelRegistry async_registry;
    async_registry.AddModel(CpuFixtureModel("det", Shape::kBoxes), boxes);
    async_registry.AddModel(CpuFixtureModel("seg", Shape::kInstances), instances);
    async_registry.AddModel(CpuFixtureModel("depth", Shape::kDenseMap), SplitDepth768());
    const FrameReport actual = RunAsync(kVolumeGraph, &async_registry, frame);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    const RecordsData* records = RecordsOf(expected, "volume");
    GRAPH_CHECK(records != NULL && records->items.size() == 1);
    if (records == NULL || records->items.size() != 1) return;
    double area = 0.0;
    double median = 0.0;
    double volume = 0.0;
    // Mask columns 100..149 of 500 sit at depth columns 154..229 of 768: the left half, depth 2.
    GRAPH_CHECK(NamedNumber(records->items[0], "mask_area_px", &area) && area == 2500.0);
    GRAPH_CHECK(NamedNumber(records->items[0], "median_depth", &median) && median == 2.0);
    GRAPH_CHECK(NamedNumber(records->items[0], "volume_proxy", &volume) && volume == 5000.0);
}

/// One cpu input: a payload and the origin its producer's result carries (inv_align a scale for a hand-off).
StageResult CpuInputOf(const StageDataPtr& data, float scale) {
    StageResult input;
    input.data = data;
    input.origin.inv_align = cv::Matx23f(scale, 0.f, 0.f, 0.f, scale, 0.f);
    return input;
}

double VolumeNumber(const CpuReduction& out, const char* name) {
    const RecordsData* records = static_cast<const RecordsData*>(out.result.data.get());
    double value = -1.0;
    if (records == NULL || records->items.size() != 1 || !NamedNumber(records->items[0], name, &value)) return -1.0;
    return value;
}

// A segmenter that ran on a x2 hand-off (1000x666 for a 500x333 frame): its box and mask are in that image's
// pixels. volume pairs it with the detector's box and reports source pixels.
void TestVolumeMapsAHandedOffSegmenterToSourcePixels() {
    std::shared_ptr<BoxesData> boxes;
    std::shared_ptr<BoxesData> instances;
    SuitcaseFixture(cv::Rect2f(100.f, 100.f, 50.f, 50.f), cv::Size(1000, 666), cv::Rect(200, 200, 100, 100),
                    &boxes, &instances);
    std::vector<StageResult> inputs;
    inputs.push_back(CpuInputOf(boxes, 1.f));
    inputs.push_back(CpuInputOf(instances, 0.5f));
    inputs.push_back(CpuInputOf(SplitDepth768(), 1.f));
    const CpuReduction out = ReduceCpu("volume", StageParams(), inputs, cv::Size(500, 333));
    GRAPH_CHECK(out.error.empty());
    GRAPH_CHECK(VolumeNumber(out, "box_x") == 100.0);
    GRAPH_CHECK(VolumeNumber(out, "mask_area_px") == 2500.0);   // 10000 mask pixels of a quarter source pixel
    GRAPH_CHECK(VolumeNumber(out, "median_depth") == 2.0);
    GRAPH_CHECK(VolumeNumber(out, "volume_proxy") == 5000.0);
}

// An input that does not cover the frame (a crop, or a rotated image) is refused, not measured.
void TestVolumeRefusesARasterThatDoesNotCoverTheFrame() {
    std::shared_ptr<BoxesData> boxes;
    std::shared_ptr<BoxesData> instances;
    SuitcaseFixture(cv::Rect2f(100.f, 100.f, 50.f, 50.f), cv::Size(500, 333), cv::Rect(100, 100, 50, 50),
                    &boxes, &instances);
    std::vector<StageResult> inputs;
    inputs.push_back(CpuInputOf(boxes, 1.f));
    inputs.push_back(CpuInputOf(instances, 1.f));
    inputs.push_back(CpuInputOf(SplitDepth768(), 1.f));
    inputs[2].origin.inv_align = cv::Matx23f(0.f, -1.f, 333.f, 1.f, 0.f, 0.f);   // rotated 90 degrees
    const CpuReduction rotated = ReduceCpu("volume", StageParams(), inputs, cv::Size(500, 333));
    GRAPH_CHECK(rotated.error == "volume reads full-frame results, and its densemap input is not one");
    inputs[2].origin = RoiRef();
    inputs[1].origin.from_roi = true;
    const CpuReduction crop = ReduceCpu("volume", StageParams(), inputs, cv::Size(500, 333));
    GRAPH_CHECK(crop.error == "volume reads full-frame results, and its instances input is not one");
}

/// The five face landmarks of a real face (nose toward the camera, mouth below it), 600 mm in front of a 640x480
/// pinhole camera of focal 640, after the head turns `down_deg` about the camera's x axis (positive: the chin goes
/// to the chest, the nose tip moves down in the image).
std::shared_ptr<BoxesData> FaceTurnedDown(double down_deg) {
    const double kFace[5][3] = {{-30, -30, 0}, {30, -30, 0}, {0, 0, -30}, {-25, 30, -10}, {25, 30, -10}};
    const double a = down_deg * 3.14159265358979323846 / 180.0;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem face;
    face.box = cv::Rect2f(250.f, 170.f, 140.f, 140.f);
    for (int i = 0; i < 5; ++i) {
        const double y = kFace[i][1] * std::cos(a) - kFace[i][2] * std::sin(a);
        const double z = kFace[i][1] * std::sin(a) + kFace[i][2] * std::cos(a) + 600.0;
        face.landmarks.push_back(Keypoint(static_cast<float>(640.0 * kFace[i][0] / z + 320.0),
                                          static_cast<float>(640.0 * y / z + 240.0), 1.f));
    }
    boxes->items.push_back(face);
    return boxes;
}

// driver_attention.json raises head_down on pitch < -20: headpose's pitch is negative for a head turned down.
void TestHeadPosePitchIsNegativeForAHeadTurnedDown() {
    double pitch[3] = {0.0, 0.0, 0.0};
    const double turns[3] = {0.0, 25.0, -25.0};
    for (int i = 0; i < 3; ++i) {
        std::vector<StageResult> inputs;
        inputs.push_back(CpuInputOf(FaceTurnedDown(turns[i]), 1.f));
        const CpuReduction out = ReduceCpu("headpose", StageParams(), inputs, cv::Size(640, 480));
        const RecordsData* records = static_cast<const RecordsData*>(out.result.data.get());
        GRAPH_CHECK(out.error.empty() && records != NULL && records->items.size() == 1);
        if (records == NULL || records->items.size() != 1) return;
        GRAPH_CHECK(NamedNumber(records->items[0], "pitch", &pitch[i]));
    }
    GRAPH_CHECK(std::fabs(pitch[0]) < 5.0);   // looking at the camera
    GRAPH_CHECK(pitch[1] < -20.0);            // head down: driver_attention's head_down fires
    GRAPH_CHECK(pitch[2] > 20.0);             // head up
}

// driver_attention's thumbnail showed only a small face box: headpose now draws the head's axes from the nose
// (x red, y green, z blue - where the face points) and labels its yaw, pitch and roll under the face box.
void TestHeadPoseDrawsItsAxesAndItsAngles() {
    const double turns[2] = {0.0, 25.0};
    for (int t = 0; t < 2; ++t) {
        std::shared_ptr<BoxesData> face = FaceTurnedDown(turns[t]);
        std::vector<StageResult> inputs;
        inputs.push_back(CpuInputOf(face, 1.f));
        const CpuReduction out = ReduceCpu("headpose", StageParams(), inputs, cv::Size(640, 480));
        GRAPH_CHECK(out.error.empty() && out.overlay.axes.size() == 1);
        if (out.overlay.axes.size() != 1) return;
        const OverlayAxes& axes = out.overlay.axes[0];
        const Keypoint& nose = face->items[0].landmarks[2];
        GRAPH_CHECK(std::fabs(axes.origin.x - nose.x) < 2.f && std::fabs(axes.origin.y - nose.y) < 2.f);
        GRAPH_CHECK(axes.x.x > axes.origin.x + 20.f);   // x: toward the face's left eye, image right
        GRAPH_CHECK(axes.y.y > axes.origin.y + 20.f);   // y: toward the chin
        if (turns[t] == 0.0) {
            // Facing the camera: the z axis points at the viewer and is foreshortened to (almost) a dot.
            GRAPH_CHECK(cv::norm(axes.z - axes.origin) < 10.0);
        } else {
            GRAPH_CHECK(axes.z.y > axes.origin.y + 10.f);  // the face points down
        }
        GRAPH_CHECK(out.overlay.texts.size() == 1);
        if (out.overlay.texts.size() != 1) return;
        const OverlayText& label = out.overlay.texts[0];
        GRAPH_CHECK(label.text.find("yaw ") == 0 && label.text.find("pitch ") != std::string::npos &&
                    label.text.find("roll ") != std::string::npos);
        const cv::Rect2f& box = face->items[0].box;
        GRAPH_CHECK(label.at.y >= box.y + box.height && label.away.y > 0.f);  // under the face box
        GRAPH_CHECK(!out.overlay.alert);

        FrameReport report;
        report.overlays["headpose"] = out.overlay;
        const cv::Mat canvas = RenderReport(cv::Mat::zeros(480, 640, CV_8UC3), report);
        const cv::Point2f x_mid = (axes.origin + axes.x) * 0.5f;
        const cv::Point2f y_mid = (axes.origin + axes.y) * 0.5f;
        const cv::Vec3b red = canvas.at<cv::Vec3b>(cvRound(x_mid.y), cvRound(x_mid.x));
        GRAPH_CHECK(red[2] > 150 && red[0] < 100 && red[1] < 100);
        if (turns[t] == 0.0) {
            const cv::Vec3b green = canvas.at<cv::Vec3b>(cvRound(y_mid.y), cvRound(y_mid.x));
            GRAPH_CHECK(green[1] > 150 && green[0] < 100 && green[2] < 100);
        } else {
            // Turned down, the z axis runs down over the y axis, drawn on top of it.
            const cv::Point2f z_mid = (axes.origin + axes.z) * 0.5f;
            const cv::Vec3b blue = canvas.at<cv::Vec3b>(cvRound(z_mid.y), cvRound(z_mid.x));
            GRAPH_CHECK(blue[0] > 150 && blue[2] < 100);
        }
    }
    // No face, no overlay.
    std::vector<StageResult> none;
    none.push_back(CpuInputOf(StageDataPtr(new BoxesData(Shape::kBoxes)), 1.f));
    GRAPH_CHECK(IsEmpty(ReduceCpu("headpose", StageParams(), none, cv::Size(640, 480)).overlay));
}

// Both executors put the cpu node's overlay in the report, under the node's id, as a logic node's is.
void TestHeadPoseOverlayReachesTheReportInBothExecutors() {
    FakeModelRegistry registry;
    registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), FaceTurnedDown(25.0));
    const std::string graph =
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"}]}";
    const FrameReport sync = RunSync(graph, &registry, cv::Mat::zeros(480, 640, CV_8UC3));
    const FrameReport async = RunAsync(graph, &registry, cv::Mat::zeros(480, 640, CV_8UC3));
    GRAPH_CHECK(sync.error.empty() && async.error.empty());
    GRAPH_CHECK(sync.overlays.count("pose") == 1 && sync.overlays.at("pose").axes.size() == 1);
    GRAPH_CHECK(sync.logic_nodes.count("pose") == 0);
    GRAPH_CHECK(sync == async);
    FrameReport moved = async;
    moved.overlays["pose"].axes[0].z.x += 1.f;
    GRAPH_CHECK(!(moved == sync));
}

/// FaceTurnedDown(down_deg), moved `dx` pixels right: its box and its landmarks.
BoxItem FaceTurnedDownAt(double down_deg, float dx) {
    BoxItem face = FaceTurnedDown(down_deg)->items[0];
    face.box.x += dx;
    for (std::size_t i = 0; i < face.landmarks.size(); ++i) face.landmarks[i].x += dx;
    return face;
}

// dms_headpose on sample_face.jpg drew one head of five: headpose read the largest face only. It writes one
// record per face with five landmarks, in the faces' order, each naming its face (face_index, and track_id when
// the face is tracked), and draws every head's axes with its angles under that face's box.
void TestHeadPoseWritesOneRecordPerFace() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    const double turns[5] = {0.0, 25.0, -25.0, 0.0, 25.0};
    for (int i = 0; i < 5; ++i) {
        boxes->items.push_back(FaceTurnedDownAt(turns[i], -200.f + 100.f * static_cast<float>(i)));
    }
    boxes->items[3].track_id = 7;
    BoxItem bare;  // a face without landmarks has no pose: no record, and the others keep their index
    bare.box = cv::Rect2f(10.f, 10.f, 30.f, 30.f);
    boxes->items.insert(boxes->items.begin() + 2, bare);
    std::vector<StageResult> inputs;
    inputs.push_back(CpuInputOf(boxes, 1.f));
    const CpuReduction out = ReduceCpu("headpose", StageParams(), inputs, cv::Size(640, 480));
    const RecordsData* records = static_cast<const RecordsData*>(out.result.data.get());
    GRAPH_CHECK(out.error.empty() && records != NULL && records->items.size() == 5);
    GRAPH_CHECK(out.overlay.axes.size() == 5 && out.overlay.texts.size() == 5);
    if (records == NULL || records->items.size() != 5 || out.overlay.axes.size() != 5 ||
        out.overlay.texts.size() != 5) {
        return;
    }
    const int faces[5] = {0, 1, 3, 4, 5};
    for (int i = 0; i < 5; ++i) {
        const BoxItem& face = boxes->items[faces[i]];
        double index = -1.0;
        double pitch = 0.0;
        double track = -1.0;
        GRAPH_CHECK(NamedNumber(records->items[i], "face_index", &index) && index == faces[i]);
        GRAPH_CHECK(NamedNumber(records->items[i], "pitch", &pitch));
        GRAPH_CHECK(NamedNumber(records->items[i], "track_id", &track) == (faces[i] == 4));
        if (faces[i] == 4) GRAPH_CHECK(track == 7.0);
        const double turn = turns[i];
        if (turn > 0.0) GRAPH_CHECK(pitch < -15.0);
        if (turn < 0.0) GRAPH_CHECK(pitch > 15.0);
        const Keypoint& nose = face.landmarks[2];
        GRAPH_CHECK(std::fabs(out.overlay.axes[i].origin.x - nose.x) < 2.f &&
                    std::fabs(out.overlay.axes[i].origin.y - nose.y) < 2.f);
        // Under its own face box, centred: a label beside a box runs over the next face in a row of faces.
        const OverlayText& label = out.overlay.texts[i];
        GRAPH_CHECK(label.text.find("yaw ") == 0);
        GRAPH_CHECK(std::fabs(label.at.x - (face.box.x + face.box.width / 2.f)) < 0.5f &&
                    std::fabs(label.at.y - (face.box.y + face.box.height)) < 0.5f);
        GRAPH_CHECK(label.away.x == 0.f && label.away.y > 0.f);
    }
}

// driver_attention reads several heads as condition reads several records: "yaw > 30" takes their max (any head
// turned right), "yaw < -30" and "pitch < -20" their min ("reduce": "min": any head turned left, any head down).
void TestDriverAttentionFiresWhenAnyHeadIsDown() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(FaceTurnedDownAt(0.0, -150.f));
    boxes->items.push_back(FaceTurnedDownAt(25.0, 150.f));
    FakeModelRegistry registry;
    registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const std::string graph =
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"},"
        "{\"id\":\"down\",\"type\":\"logic\",\"op\":\"condition\",\"params\":{\"value\":\"pitch\","
        "\"comparator\":\"<\",\"threshold\":-20,\"reduce\":\"min\",\"event\":\"head_down\"}}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"},"
        "{\"from\":\"pose\",\"to\":\"down\",\"carry\":\"result\"}]}";
    const FrameReport sync = RunSync(graph, &registry, cv::Mat::zeros(480, 640, CV_8UC3));
    const FrameReport async = RunAsync(graph, &registry, cv::Mat::zeros(480, 640, CV_8UC3));
    GRAPH_CHECK(sync.error.empty() && sync == async);
    const RecordsData* poses = RecordsOf(sync, "pose");
    GRAPH_CHECK(poses != NULL && poses->items.size() == 2);
    GRAPH_CHECK(sync.events.size() == 1 && sync.events[0].event == "head_down");
}

// Graph Composer P1: every non-source node's time in this frame, sync.
void TestSyncExecutorTimesEveryNode() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetLatency("yolov8n", 30);
    registry.SetLatency("casvit_t", 10);
    const GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const FrameReport report =
        SyncExecutor().RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_ms.count("cam") == 0);
    GRAPH_CHECK(report.node_ms.count("od") == 1 && report.node_ms.at("od") >= 29.0);
    // Two person crops run one after the other: both run() calls count.
    GRAPH_CHECK(report.node_ms.count("reid") == 1 && report.node_ms.at("reid") >= 19.0);
    GRAPH_CHECK(report.node_ms.at("od") < 5000.0);
    // Timing is not part of a report's identity: parity compares results.
    FrameReport other = report;
    other.node_ms["od"] += 100.0;
    GRAPH_CHECK(other == report);

    // A cpu node is timed too.
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    BoxItem face;
    face.box = cv::Rect2f(200.f, 120.f, 80.f, 100.f);
    face.class_name = "face";
    face.landmarks.push_back(Keypoint(220.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(260.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(240.f, 180.f, 1.f));
    face.landmarks.push_back(Keypoint(225.f, 200.f, 1.f));
    face.landmarks.push_back(Keypoint(255.f, 200.f, 1.f));
    boxes->items.push_back(face);
    FakeModelRegistry cpu_registry;
    cpu_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport cpu = RunSync(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"}]}",
        &cpu_registry, cv::Mat::zeros(480, 640, CV_8UC3));
    GRAPH_CHECK(cpu.error.empty());
    GRAPH_CHECK(cpu.node_ms.count("pose") == 1);
    GRAPH_CHECK(cpu.node_ms.count("face") == 1);
}

// Graph Composer P1: async node_ms runs from dispatch to the last completion.
void TestAsyncExecutorTimesEveryNode() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetLatency("yolov8n", 30);
    registry.SetLatency("casvit_t", 10);
    const GraphSpec spec = ParseGraphText(CascadeJson(), "t.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    AsyncExecutor executor(StageJobs(4));
    const FrameReport report =
        executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(report.node_ms.count("cam") == 0);
    GRAPH_CHECK(report.node_ms.count("od") == 1 && report.node_ms.at("od") >= 29.0);
    GRAPH_CHECK(report.node_ms.count("reid") == 1 && report.node_ms.at("reid") >= 9.0);
    GRAPH_CHECK(report.node_ms.at("od") < 5000.0);

    // A ROI node that got no crops took no time.
    FakeModelRegistry empty = BuildCascadeRegistry();
    const GraphSpec none = ParseGraphText(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\"},{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{\"classes\":[\"car\"]}}]}",
        "none.json");
    StageGraph none_graph;
    none_graph.Build(none, empty, "/models", false);
    AsyncExecutor none_executor(StageJobs(4));
    const FrameReport no_crops =
        none_executor.RunFrame(none_graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    GRAPH_CHECK(no_crops.node_ms.count("reid") == 1 && no_crops.node_ms.at("reid") == 0.0);
}

// --report-timing: "node_ms" is written only when asked, and changes nothing else.
void TestReportToJsonWritesNodeMsOnlyWhenAsked() {
    FrameReport report;
    report.frame_index = 3;
    report.node_ms["od"] = 12.34567;
    report.node_ms["reid"] = 0.0;
    const nlohmann::json plain = consumer::ReportToJson(report, false);
    GRAPH_CHECK(!plain.contains("node_ms"));
    const nlohmann::json timed = consumer::ReportToJson(report, false, true);
    GRAPH_CHECK(timed.contains("node_ms"));
    GRAPH_CHECK(std::fabs(timed["node_ms"]["od"].get<double>() - 12.346) < 1e-9);
    GRAPH_CHECK(timed["node_ms"]["reid"].get<double>() == 0.0);
    nlohmann::json without = timed;
    without.erase("node_ms");
    GRAPH_CHECK(without.dump() == plain.dump());
}

namespace {

std::string EngineNodeJson(const std::string& engine) {
    return "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"engine\":" + engine + "}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}]}";
}

std::string SharedDetectorsJson(const std::string& od2_engine, const std::string& od2_params) {
    return "{\"version\":1,\"name\":\"shared\",\"nodes\":["
           "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
           "{\"id\":\"od\",\"model\":\"yolov8n\",\"engine\":{\"share\":true}},"
           "{\"id\":\"od2\",\"model\":\"yolov8n\",\"engine\":" + od2_engine + od2_params + "},"
           "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
           "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},{\"from\":\"cam\",\"to\":\"od2\"},"
           "{\"from\":\"od2\",\"to\":\"reid\",\"roi\":{\"classes\":[\"person\"]}}]}";
}

}  // namespace

void TestEngineKeyParsesAndValidates() {
    const GraphSpec plain = ParseGraphText(CascadeJson(), "t.json");
    GRAPH_CHECK(!plain.nodes[1].engine.present);
    GRAPH_CHECK(plain.nodes[1].engine.buffers == 0 && !plain.nodes[1].engine.share);

    const GraphSpec set = ParseGraphText(EngineNodeJson("{\"buffers\":2,\"share\":true}"), "e.json");
    GRAPH_CHECK(set.nodes[1].engine.present);
    GRAPH_CHECK(set.nodes[1].engine.buffers == 2);
    GRAPH_CHECK(set.nodes[1].engine.share);

    std::string message;
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("{\"buffers\":0}"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"buffers\" must be a whole number of at least 1") != std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("{\"buffers\":101}"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"buffers\" must be at most 100") != std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("{\"buffers\":2.5}"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("{\"share\":\"yes\"}"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"share\" must be true or false") != std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("{\"bufers\":2}"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"buffers\"") != std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(EngineNodeJson("[2]"), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\","
        "\"engine\":{\"buffers\":2}}],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"engine\" belongs on a model node") != std::string::npos);

    // Sharers must want the same engine.
    FakeModelRegistry registry = BuildCascadeRegistry();
    GRAPH_CHECK(CpuValidateThrows(SharedDetectorsJson("{\"share\":true}", ",\"params\":{\"score_threshold\":0.5}"),
                                  registry, GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("node \"od2\" \"engine\"") != std::string::npos);
    GRAPH_CHECK(message.find("shares the engine of node \"od\", but its \"params\" differ") != std::string::npos);
    GRAPH_CHECK(CpuValidateThrows(SharedDetectorsJson("{\"share\":true,\"buffers\":2}", ""),
                                  registry, GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("but its \"buffers\" differ (2 vs the default)") != std::string::npos);
    const GraphSpec ok = ParseGraphText(SharedDetectorsJson("{\"share\":true}", ""), "s.json");
    ValidateGraph(ok, registry);

    // Who owns each shared engine: the first sharer of the variant.
    const std::map<std::string, std::string> owners = SharedEngineOwners(ok, registry);
    GRAPH_CHECK(owners.size() == 2);
    GRAPH_CHECK(owners.count("od") == 1 && owners.at("od") == "od");
    GRAPH_CHECK(owners.count("od2") == 1 && owners.at("od2") == "od");
    GRAPH_CHECK(owners.count("reid") == 0);
}

// Graph Composer P1: a node's "buffers" reaches the registry; a registry that
// cannot honour it says so instead of ignoring it.
void TestEngineBuffersReachTheRegistry() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    const GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"od\",\"model\":\"yolov8n\",\"engine\":{\"buffers\":2}},"
        "{\"id\":\"reid\",\"model\":\"casvit_t\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},"
        "{\"from\":\"od\",\"to\":\"reid\",\"roi\":{}}]}",
        "b.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    GRAPH_CHECK(registry.last_engine("yolov8n") != NULL &&
                registry.last_engine("yolov8n")->buffers == 2);
    GRAPH_CHECK(registry.last_engine("casvit_t") != NULL &&
                registry.last_engine("casvit_t")->buffers == 0);

    EngineOptions two;
    two.buffers = 2;
    bool refused = false;
    try {
        registry.IModelRegistry::createStageWith("yolov8n", "/models/yolov8n.dxnn",
                                                 StageParams(), two);
    } catch (const std::runtime_error& error) {
        refused = std::string(error.what()).find("buffer count") != std::string::npos;
    }
    GRAPH_CHECK(refused);
    const std::unique_ptr<IStage> plain = registry.IModelRegistry::createStageWith(
        "yolov8n", "/models/yolov8n.dxnn", StageParams(), EngineOptions());
    GRAPH_CHECK(plain.get() != NULL);
}

// The zoo side opens every engine through OpenEngine, which reads the count
// StaticModelRegistry::createStageWith scoped onto this thread.
void TestEngineBufferScopeSetsAndRestores() {
    GRAPH_CHECK(detail::EngineBuffersSlot() == 0);
    GRAPH_CHECK(detail::EngineOptionForThisThread().bufferCount == 0);
    {
        detail::EngineBufferScope outer(2);
        GRAPH_CHECK(detail::EngineOptionForThisThread().bufferCount == 2);
        {
            detail::EngineBufferScope inner(5);
            GRAPH_CHECK(detail::EngineBuffersSlot() == 5);
        }
        GRAPH_CHECK(detail::EngineBuffersSlot() == 2);
        int seen_elsewhere = -1;
        std::thread other([&seen_elsewhere] { seen_elsewhere = detail::EngineBuffersSlot(); });
        other.join();
        GRAPH_CHECK(seen_elsewhere == 0);
    }
    GRAPH_CHECK(detail::EngineBuffersSlot() == 0);
    try {
        detail::EngineBufferScope failing(3);
        throw std::runtime_error("load failed");
    } catch (const std::runtime_error&) {
    }
    GRAPH_CHECK(detail::EngineBuffersSlot() == 0);
}

// Graph Composer P1: nodes that share get ONE stage; each still gets its own
// results, in both executors and with several frames in flight.
void TestSharedNodesUseOneStage() {
    const GraphSpec spec = ParseGraphText(SharedDetectorsJson("{\"share\":true}", ""), "s.json");
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    ValidateGraph(spec, sync_registry);
    StageGraph sync_graph;
    sync_graph.Build(spec, sync_registry, "/models", false);
    GRAPH_CHECK(sync_registry.stages_made("yolov8n") == 1);
    const FrameReport expected = SyncExecutor().RunFrame(sync_graph, frame, 0);

    FakeModelRegistry async_registry = BuildCascadeRegistry();
    StageGraph async_graph;
    async_graph.Build(spec, async_registry, "/models", false);
    GRAPH_CHECK(async_registry.stages_made("yolov8n") == 1);
    AsyncExecutor executor(StageJobs(4));
    const FrameReport actual = executor.RunFrame(async_graph, frame, 0);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    GRAPH_CHECK(expected.node_results.count("od") == 1 && expected.node_results.count("od2") == 1);
    GRAPH_CHECK(expected.roi_results.count("reid") == 1 && expected.roi_results.at("reid").size() == 2);

    // Without "share", one stage per node, as before.
    FakeModelRegistry plain = BuildCascadeRegistry();
    StageGraph plain_graph;
    plain_graph.Build(ParseGraphText(TwoDetectorsJson(), "two.json"), plain, "/models", false);
    GRAPH_CHECK(plain.stages_made("yolov8n") == 2);
}

void TestASharedStageKeepsEachNodesJobsApartUnderPipelining() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    // Frame i's detector output has its first person at x = 10 + 20 * i,
    // keyed by the frame's pixel (0,0), so a completion handed to another
    // frame's job (or swapped between the two sharing nodes across frames)
    // shows up as the wrong x.
    for (int i = 0; i < 6; ++i) {
        std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
        boxes->items.push_back(MakeItem(10.0f + 20.0f * static_cast<float>(i), 10, 40, 60,
                                        0.8f, "person"));
        boxes->items.push_back(MakeItem(300, 200, 40, 60, 0.9f, "person"));
        registry.SetScriptByFirstPixel("yolov8n", i, boxes);
    }
    registry.SetShuffleSeed("yolov8n", 7);
    const GraphSpec spec = ParseGraphText(SharedDetectorsJson("{\"share\":true}", ""), "s.json");
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    GRAPH_CHECK(registry.stages_made("yolov8n") == 1);
    AsyncOptions options;
    options.max_frames_in_flight = 4;
    AsyncExecutor executor(options);
    for (std::size_t i = 0; i < 6; ++i) {
        executor.Submit(graph, FrameWithKey(static_cast<int>(i)), i);
    }
    executor.Finish(graph);
    FrameReport report;
    std::size_t seen = 0;
    while (executor.TryNext(&report)) {
        GRAPH_CHECK(report.frame_index == seen);
        GRAPH_CHECK(report.error.empty());
        GRAPH_CHECK(report.node_results.count("od") == 1 && report.node_results.count("od2") == 1);
        const float expected_x = 10.0f + 20.0f * static_cast<float>(seen);
        const char* sharers[] = {"od", "od2"};
        for (std::size_t n = 0; n < 2; ++n) {
            std::shared_ptr<const BoxesData> boxes =
                std::dynamic_pointer_cast<const BoxesData>(report.node_results[sharers[n]].data);
            GRAPH_CHECK(boxes && boxes->items.size() == 2);
            if (!boxes || boxes->items.size() != 2) continue;
            bool found = false;
            for (std::size_t b = 0; b < boxes->items.size(); ++b) {
                if (boxes->items[b].box.x == expected_x) found = true;
            }
            GRAPH_CHECK(found);
        }
        GRAPH_CHECK(report.roi_results.count("reid") == 1 && report.roi_results["reid"].size() == 2);
        ++seen;
    }
    GRAPH_CHECK(seen == 6);
}

// Build() on a spec ValidateGraph never saw: a sharer whose params or
// buffers differ from the owner's must not silently run the owner's
// configuration.
void TestBuildRejectsAnUnvalidatedSharerWithOtherSettings() {
    const std::string variants[] = {
        SharedDetectorsJson("{\"share\":true}", ",\"params\":{\"score_threshold\":0.5}"),
        SharedDetectorsJson("{\"share\":true,\"buffers\":2}", "")};
    for (std::size_t v = 0; v < 2; ++v) {
        FakeModelRegistry registry = BuildCascadeRegistry();
        const GraphSpec spec = ParseGraphText(variants[v], "s.json");
        StageGraph graph;
        std::string message;
        try {
            graph.Build(spec, registry, "/models", false);  // no ValidateGraph
        } catch (const std::logic_error& error) {
            message = error.what();
        }
        GRAPH_CHECK(message.find("node \"od2\" shares the engine of node \"od\" but its "
                                 "params or buffers differ") != std::string::npos);
        GRAPH_CHECK(graph.nodes().empty());
    }
}

namespace {

/// sample_people.jpg moved right by 8 px per frame index.
cv::Mat SharedTestFrame(const cv::Mat& image, std::size_t index) {
    cv::Mat moved;
    const cv::Mat shift = (cv::Mat_<float>(2, 3) << 1, 0, 8.0f * static_cast<float>(index), 0, 1, 0);
    cv::warpAffine(image, moved, shift, image.size());
    return moved;
}

}  // namespace

// Real hardware: two nodes on one yolov8-n engine with 2 buffers, 8 frames
// pipelined through DX-RT's own completion threads, against sync.
void TestRealSharedEngineWithBuffers() {
    const char* kTest = "TestRealSharedEngineWithBuffers";
    StaticModelRegistry registry;
    const std::string missing =
        MissingArtifacts(registry, std::vector<std::string>(1, "yolov8-n_640x640"));
    if (!missing.empty()) { Skip(kTest, missing); return; }
    const cv::Mat image = cv::imread(ProjectRoot() + "/sample/img/sample_people.jpg");
    GRAPH_CHECK(!image.empty());
    if (image.empty()) return;
    const GraphSpec spec = ParseGraphText(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"x\"},"
        "{\"id\":\"od\",\"model\":\"yolov8-n_640x640\",\"engine\":{\"share\":true,\"buffers\":2}},"
        "{\"id\":\"od2\",\"model\":\"yolov8-n_640x640\",\"engine\":{\"share\":true,\"buffers\":2}}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"od\"},{\"from\":\"cam\",\"to\":\"od2\"}]}",
        "real_shared.json");
    ValidateGraph(spec, registry);

    StageGraph sync_graph;
    sync_graph.Build(spec, registry, ModelDir());
    SyncExecutor sync_executor;
    std::vector<FrameReport> expected;
    for (std::size_t i = 0; i < 8; ++i) {
        expected.push_back(sync_executor.RunFrame(sync_graph, SharedTestFrame(image, i), i));
    }

    StageGraph async_graph;
    async_graph.Build(spec, registry, ModelDir());
    AsyncOptions options;
    options.max_frames_in_flight = 8;
    AsyncExecutor async_executor(options);
    for (std::size_t i = 0; i < 8; ++i) {
        async_executor.Submit(async_graph, SharedTestFrame(image, i), i);
    }
    async_executor.Finish(async_graph);
    FrameReport report;
    std::size_t k = 0;
    while (async_executor.TryNext(&report)) {
        GRAPH_CHECK(k < expected.size() && report == expected[k]);
        ++k;
    }
    GRAPH_CHECK(k == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        GRAPH_CHECK(expected[i].error.empty());
        const nlohmann::json out = consumer::ReportToJson(expected[i], false);
        GRAPH_CHECK(out["nodes"]["od"]["payload"] == out["nodes"]["od2"]["payload"]);
        GRAPH_CHECK(!out["nodes"]["od"]["payload"]["items"].empty());
    }
}

// Graph Composer P1, the shared-engine restrictions (plan decision 5):
// sharing never crosses StageGraphs, and max_jobs_per_stage counts per node,
// so one shared engine may hold (sharers x N) jobs at once.
void TestSharedEngineRestrictions() {
    const GraphSpec spec = ParseGraphText(SharedDetectorsJson("{\"share\":true}", ""), "s.json");

    FakeModelRegistry registry = BuildCascadeRegistry();
    ValidateGraph(spec, registry);
    StageGraph first;
    first.Build(spec, registry, "/models", false);
    StageGraph second;
    second.Build(spec, registry, "/models", false);
    GRAPH_CHECK(registry.stages_made("yolov8n") == 2);  // one per graph, not one for both

    FakeModelRegistry jobs = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(spec, jobs, "/models", false);
    GRAPH_CHECK(jobs.stages_made("yolov8n") == 1);
    AsyncExecutor executor(StageJobs(1));
    const FrameReport report = executor.RunFrame(graph, cv::Mat::zeros(480, 640, CV_8UC3), 0);
    GRAPH_CHECK(report.error.empty());
    GRAPH_CHECK(jobs.last_stage("yolov8n") != NULL);
    if (jobs.last_stage("yolov8n") == NULL) return;
    GRAPH_CHECK(jobs.last_stage("yolov8n")->max_pending() == 2);  // od's 1 + od2's 1
}

// Graph Composer P1: MODEL_LOAD leaves no engine open, even while the
// StageGraph that failed is still alive (dx_graph keeps it; Studio retries).
void TestAFailedLoadClosesEveryEngineItOpened() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    registry.SetCreateFailure("yolov8n", "Failed to register memory cache", 1);
    const GraphSpec spec = ParseGraphText(TwoDetectorsJson(), "two.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    std::string message;
    bool model_load = false;
    try {
        graph.Build(spec, registry, "/models", false);
    } catch (const GraphError& error) {
        model_load = error.code() == GraphErrorCode::kModelLoad;
        message = error.what();
    }
    GRAPH_CHECK(model_load);
    GRAPH_CHECK(registry.stages_made("yolov8n") == 1);  // od opened, od2 failed
    GRAPH_CHECK(registry.live_stages() == 0);           // and od is closed again
    GRAPH_CHECK(graph.nodes().empty());
    GRAPH_CHECK(message.find("\"engine\": {\"share\": true}") != std::string::npos);
    GRAPH_CHECK(message.find("\"buffers\"") != std::string::npos);
}

// --check --json reports where, what and how apart.
void TestGraphErrorKeepsItsThreeParts() {
    const GraphError error(GraphErrorCode::kGraphEdge, "edge \"a\"->\"b\"", "shapes differ",
                           "use a detector");
    GRAPH_CHECK(error.where() == "edge \"a\"->\"b\"");
    GRAPH_CHECK(error.what_happened() == "shapes differ");
    GRAPH_CHECK(error.how_to_fix() == "use a detector");
    GRAPH_CHECK(std::string(error.what()) ==
                "ERROR [GRAPH_EDGE] edge \"a\"->\"b\": shapes differ\n  -> use a detector");
    std::string where;
    try {
        ParseGraphText("{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a\"},"
                       "{\"id\":\"od\",\"modle\":\"x\"}],\"edges\":[]}",
                       "g.json");
    } catch (const GraphError& parse_error) {
        where = parse_error.where();
    }
    GRAPH_CHECK(where == "g.json node[1] \"od\"");
}

// ============================================================== Graph Composer P2
// Logic nodes: one catalog drives validation and --list-ops.
void TestLogicOpCatalogDescribesEveryOp() {
    const char* const expected[] = {"filter", "zone", "line", "count",
                                    "dwell", "condition", "gallery", "user"};
    const std::vector<LogicOpInfo>& ops = LogicOps();
    GRAPH_CHECK(ops.size() == 8);
    for (std::size_t i = 0; i < 8 && i < ops.size(); ++i) GRAPH_CHECK(ops[i].op == expected[i]);
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const LogicOpInfo& op = ops[i];
        GRAPH_CHECK(!op.summary.empty());
        GRAPH_CHECK(!op.accepts.empty());
        GRAPH_CHECK(op.min_inputs >= 1 && op.max_inputs >= op.min_inputs);
        std::set<std::string> names;
        for (std::size_t p = 0; p < op.params.size(); ++p) {
            const LogicParamInfo& param = op.params[p];
            GRAPH_CHECK(!param.help.empty());
            GRAPH_CHECK(names.insert(param.name).second);
            if (param.kind == LogicParamInfo::kChoice) {
                GRAPH_CHECK(!param.choices.empty());
                if (!param.required) {
                    GRAPH_CHECK(std::find(param.choices.begin(), param.choices.end(),
                                          param.text_default) != param.choices.end());
                }
            }
        }
    }
    GRAPH_CHECK(FindLogicOp("zone") != NULL && FindLogicOp("zone")->events.size() == 2);
    GRAPH_CHECK(FindLogicOp("headpose") == NULL);
    GRAPH_CHECK(FindLogicOp("line")->needs_tracking && FindLogicOp("dwell")->needs_tracking);
    GRAPH_CHECK(!FindLogicOp("zone")->needs_tracking && !FindLogicOp("count")->needs_tracking);
    GRAPH_CHECK(FindLogicOp("filter")->same_shape_out && FindLogicOp("user")->same_shape_out);
    GRAPH_CHECK(!FindLogicOp("line")->same_shape_out);
    GRAPH_CHECK(FindLogicOp("count")->max_inputs == 8);
    GRAPH_CHECK(CpuOpInfos().size() == 2);
    GRAPH_CHECK(FindCpuOpInfo("volume") != NULL && FindCpuOpInfo("volume")->min_inputs == 3);
    GRAPH_CHECK(FindCpuOpInfo("headpose") != NULL && FindCpuOpInfo("headpose")->numbers.size() == 5);
    GRAPH_CHECK(std::string(ToString(LogicParamInfo::kPolygon)) == "polygon");
    GRAPH_CHECK(std::string(ToString(LogicParamInfo::kTextList)) == "text_list");
}

namespace {

/// cam -> od (model "det", tracked unless `tracked` is false), then `nodes`
/// and `edges`, each a JSON list fragment that starts with a comma.
std::string LogicJson(const std::string& nodes, const std::string& edges, bool tracked = true) {
    return std::string("{\"version\":1,\"name\":\"logic\",\"nodes\":["
                       "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
                       "{\"id\":\"od\",\"model\":\"det\"") +
           (tracked ? ",\"track\":{\"algo\":\"iou\",\"iou\":0.3,\"max_age\":30}" : "") + "}" +
           nodes + "],\"edges\":[{\"from\":\"cam\",\"to\":\"od\"}" + edges + "]}";
}

const char kZoneNode[] =
    ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\","
    "\"params\":{\"polygon\":[0.5,0,1,0,1,1,0.5,1]}}";
const char kOdToZ[] = ",{\"from\":\"od\",\"to\":\"z\",\"carry\":\"result\"}";

}  // namespace

void TestParseReadsALogicNode() {
    const GraphSpec spec = ParseGraphText(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":"
        "{\"polygon\":[0,0,1,0,1,1],\"mode\":\"inside\",\"classes\":[]}}",
        kOdToZ), "l.json");
    GRAPH_CHECK(spec.nodes.size() == 3);
    if (spec.nodes.size() != 3) return;
    const NodeSpec& z = spec.nodes[2];
    GRAPH_CHECK(z.is_logic && !z.is_cpu && !z.is_source && z.op == "zone");
    GRAPH_CHECK(!IsModelNode(z) && IsModelNode(spec.nodes[1]) && !IsModelNode(spec.nodes[0]));
    GRAPH_CHECK(z.arrays.count("polygon") == 1 && z.arrays.at("polygon").size() == 6);
    if (z.arrays.count("polygon") == 1) GRAPH_CHECK(z.arrays.at("polygon")[2] == 1.0);
    GRAPH_CHECK(z.params.text.count("mode") == 1 && z.params.text.at("mode") == "inside");
    GRAPH_CHECK(z.params.lists.count("classes") == 1 && z.params.lists.at("classes").empty());
    GRAPH_CHECK(spec.edges[1].carry_result);
}

void TestParseRejectsBadLogicNodes() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zon\"}", ""),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("unknown logic op \"zon\"") != std::string::npos);
    GRAPH_CHECK(message.find("did you mean \"zone\"") != std::string::npos);
    GRAPH_CHECK(message.find("filter, zone, line, count, dwell, condition, gallery, user") !=
                std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"model\":\"det\"}", ""),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("a logic node has no \"model\" and no \"uri\"") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"track\":{\"algo\":\"iou\"}}", ""),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("a logic node cannot track") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"engine\":{\"buffers\":2}}", ""),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"engine\" belongs on a model node") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":[0,\"a\"]}}",
        ""), GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"polygon\" must be a number, a string, or an array of strings "
                             "or of numbers, got [0,\"a\"]") != std::string::npos);

    // A model node keeps its message: number arrays are for logic nodes.
    GRAPH_CHECK(ThrowsWithCode(ParamsGraph("yolov8n", "{\"b\":[1,2]}"),
                               GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"b\" must be a number, a string or an array of strings, "
                             "got [1,2]") != std::string::npos);

    GRAPH_CHECK(ThrowsWithCode(
        "{\"version\":1,\"nodes\":[{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"z\",\"type\":\"logik\",\"op\":\"zone\"}],\"edges\":[]}",
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("did you mean \"logic\"") != std::string::npos);
    GRAPH_CHECK(message.find("accepted node types are \"source\", \"cpu\" and \"logic\"") !=
                std::string::npos);
}

// Until Task 4 checks a logic node's params and inputs, a valid logic graph
// must at least validate without touching a model entry it does not have.
void TestValidateTakesALogicNodeAsANonModelNode() {
    FakeModelRegistry registry;
    registry.AddModel(CpuFixtureModel("det", Shape::kBoxes),
                      StageDataPtr(new BoxesData(Shape::kBoxes)));
    GRAPH_CHECK(ValidationError(LogicJson(kZoneNode, kOdToZ), registry).empty());
    std::string message;
    GRAPH_CHECK(CpuValidateThrows(LogicJson(kZoneNode, ",{\"from\":\"od\",\"to\":\"z\"}"),
                                  registry, GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("a logic node accepts only \"carry\": \"result\" edges") !=
                std::string::npos);
    GRAPH_CHECK(CpuValidateThrows(LogicJson(
        ",{\"id\":\"m\",\"model\":\"det\"}",
        ",{\"from\":\"od\",\"to\":\"m\",\"carry\":\"result\"}"),
        registry, GraphErrorCode::kGraphEdge, &message));
    GRAPH_CHECK(message.find("accepted only by a cpu node or a logic node") != std::string::npos);
}

namespace {

/// det (boxes), attr (scores, crops), emb (vector, crops), lmk (boxes with landmarks).
FakeModelRegistry LogicCheckRegistry() {
    FakeModelRegistry registry;
    registry.AddModel(CpuFixtureModel("det", Shape::kBoxes),
                      StageDataPtr(new BoxesData(Shape::kBoxes)));
    ModelInfo attr = CpuFixtureModel("attr", Shape::kScores);
    attr.input_contract = InputContract::kEither;
    registry.AddModel(attr, StageDataPtr(new ScoresData()));
    ModelInfo emb = CpuFixtureModel("emb", Shape::kVector);
    emb.input_contract = InputContract::kEither;
    registry.AddModel(emb, StageDataPtr(new VectorData()));
    return registry;
}

/// A DXGAL1 file (gallery_retrieval_postprocessor.hpp's format) under the
/// scratch directory: `rows` vectors of one width, built by `model`.
std::string WriteGallery(const std::string& name, const std::string& model,
                         const std::vector<std::vector<float> >& rows,
                         const std::vector<std::string>& labels) {
    const std::string path = ScratchPath(name);
    std::ofstream out(path.c_str(), std::ios::binary);
    out.write("DXGAL1\0\0", 8);
    const auto u32 = [&out](std::size_t value) {
        const unsigned char bytes[4] = {
            static_cast<unsigned char>(value & 0xFF), static_cast<unsigned char>((value >> 8) & 0xFF),
            static_cast<unsigned char>((value >> 16) & 0xFF),
            static_cast<unsigned char>((value >> 24) & 0xFF)};
        out.write(reinterpret_cast<const char*>(bytes), 4);
    };
    const auto blob = [&out, &u32](const std::string& text) {
        u32(text.size());
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
    };
    u32(rows.size());
    u32(rows.empty() ? 0 : rows[0].size());
    for (std::size_t r = 0; r < rows.size(); ++r) {
        out.write(reinterpret_cast<const char*>(rows[r].data()),
                  static_cast<std::streamsize>(rows[r].size() * sizeof(float)));
    }
    blob(model);
    std::string paths;
    std::string joined;
    for (std::size_t r = 0; r < rows.size(); ++r) {
        paths += (r == 0 ? "" : "\n") + std::string("gallery/") + labels[r] + ".jpg";
        joined += (r == 0 ? "" : "\n") + labels[r];
    }
    blob(paths);
    blob(joined);
    return path;
}

std::vector<std::vector<float> > AliceAndBob() {
    std::vector<std::vector<float> > rows(2, std::vector<float>(4, 0.f));
    rows[0][0] = 1.f;  // alice
    rows[1][1] = 1.f;  // bob
    return rows;
}

std::vector<std::string> AliceAndBobLabels() {
    std::vector<std::string> labels;
    labels.push_back("alice");
    labels.push_back("bob");
    return labels;
}

/// The error ValidateGraph throws for `text` with LogicCheckRegistry(), or "".
std::string LogicError(const std::string& text) {
    return ValidationError(text, LogicCheckRegistry());
}

}  // namespace

void TestValidateChecksLogicParams() {
    GRAPH_CHECK(LogicError(LogicJson(kZoneNode, kOdToZ)).empty());
    std::string m = LogicError(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":[0,0,1,0]}}",
        kOdToZ));
    GRAPH_CHECK(m.find("ERROR [GRAPH_SCHEMA] node \"z\" \"params\": \"polygon\" must hold at "
                       "least 3 points") != std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0,0,640,0,640,480]}}", kOdToZ));
    GRAPH_CHECK(m.find("\"polygon\" values are fractions of the frame from 0 to 1, got 640 "
                       "at [2]") != std::string::npos);
    m = LogicError(LogicJson(",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\"}", kOdToZ));
    GRAPH_CHECK(m.find("missing param \"polygon\"") != std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polgon\":[0,0,1,0,1,1]}}",
        kOdToZ));
    GRAPH_CHECK(m.find("unknown param \"polgon\" for op \"zone\"") != std::string::npos);
    GRAPH_CHECK(m.find("did you mean \"polygon\"?") != std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":"
        "{\"polygon\":[0,0,1,0,1,1],\"mode\":\"inner\"}}", kOdToZ));
    GRAPH_CHECK(m.find("\"mode\" must be one of inside, outside, got \"inner\"") !=
                std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\",\"params\":{\"min_score\":35}}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"min_score\" must be a number from 0 to 1, got 35") != std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"l\",\"type\":\"logic\",\"op\":\"line\",\"params\":{\"line\":[0.5,0,0.5,0]}}",
        ",{\"from\":\"od\",\"to\":\"l\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"line\" needs two different points") != std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"d\",\"type\":\"logic\",\"op\":\"dwell\",\"params\":{\"threshold_s\":0}}",
        ",{\"from\":\"od\",\"to\":\"d\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"threshold_s\" must be a number greater than 0, got 0") !=
                std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"condition\",\"params\":{\"event\":\"a b\"}}",
        ",{\"from\":\"od\",\"to\":\"c\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"event\" may hold letters, digits, _ . : - (1 to 64), got \"a b\"") !=
                std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"2nd\",\"input\":\"boxes\",\"output\":\"boxes\"}}",
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"name\" must be a C identifier that is not a C++ keyword, got \"2nd\"") != std::string::npos);
}

void TestValidateChecksLogicInputs() {
    // Wrong kind: a zone fed per-crop scores.
    std::string m = LogicError(LogicJson(
        ",{\"id\":\"a\",\"model\":\"attr\"}" + std::string(kZoneNode),
        ",{\"from\":\"od\",\"to\":\"a\",\"roi\":{}},{\"from\":\"a\",\"to\":\"z\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("ERROR [GRAPH_EDGE] edge \"a\"->\"z\": zone accepts boxes, obboxes or "
                       "instances, got scores per crop from \"a\"") != std::string::npos);
    // Too few inputs.
    m = LogicError(LogicJson(kZoneNode, ""));
    GRAPH_CHECK(m.find("zone takes 1 input, got nothing") != std::string::npos);
    // A line needs track ids, and says where to add them, through a filter.
    m = LogicError(LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\"}"
        ",{\"id\":\"l\",\"type\":\"logic\",\"op\":\"line\",\"params\":{\"line\":[0.5,1,0.5,0]}}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"},"
        "{\"from\":\"f\",\"to\":\"l\",\"carry\":\"result\"}", false));
    GRAPH_CHECK(m.find("line needs track ids, and the boxes from \"f\" have none") !=
                std::string::npos);
    GRAPH_CHECK(m.find("add \"track\": {\"algo\": \"iou\"} to node \"od\"") != std::string::npos);
    // The same graph tracked validates; a zone's boxes feed an ROI edge.
    GRAPH_CHECK(LogicError(LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\"}"
        ",{\"id\":\"l\",\"type\":\"logic\",\"op\":\"line\",\"params\":{\"line\":[0.5,1,0.5,0]}}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"},"
        "{\"from\":\"f\",\"to\":\"l\",\"carry\":\"result\"}")).empty());
    GRAPH_CHECK(LogicError(LogicJson(
        std::string(kZoneNode) + ",{\"id\":\"a\",\"model\":\"attr\"}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"a\",\"roi\":{\"pad\":0.05}}")).empty());
    // face5 through a zone looks at the model the boxes come from.
    m = LogicError(LogicJson(
        std::string(kZoneNode) + ",{\"id\":\"a\",\"model\":\"attr\"}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"a\",\"roi\":{\"align\":\"face5\"}}"));
    GRAPH_CHECK(m.find("model \"det\" produces no landmarks") != std::string::npos);
    // A count node's records: a condition reads one of its numbers.
    const std::string counted =
        ",{\"id\":\"n\",\"type\":\"logic\",\"op\":\"count\",\"params\":{\"classes\":[\"person\"]}}"
        ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"condition\",\"params\":{\"value\":\"%s\"}}";
    const std::string count_edges =
        ",{\"from\":\"od\",\"to\":\"n\",\"carry\":\"result\"},"
        "{\"from\":\"n\",\"to\":\"c\",\"carry\":\"result\"}";
    std::string good = counted;
    good.replace(good.find("%s"), 2, "person");
    GRAPH_CHECK(LogicError(LogicJson(good, count_edges)).empty());
    std::string bad = counted;
    bad.replace(bad.find("%s"), 2, "people");
    m = LogicError(LogicJson(bad, count_edges));
    GRAPH_CHECK(m.find("\"value\" is \"people\", a number \"n\" does not write") !=
                std::string::npos);
    GRAPH_CHECK(m.find("\"n\" writes: total, person") != std::string::npos);
    // Crops with scores: with:/without: a label.
    const std::string crops =
        ",{\"id\":\"a\",\"model\":\"attr\"}"
        ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"condition\",\"params\":{\"value\":\"%s\"}}";
    const std::string crop_edges =
        ",{\"from\":\"od\",\"to\":\"a\",\"roi\":{}},{\"from\":\"a\",\"to\":\"c\",\"carry\":\"result\"}";
    good = crops;
    good.replace(good.find("%s"), 2, "without:Hat");
    GRAPH_CHECK(LogicError(LogicJson(good, crop_edges)).empty());
    bad = crops;
    bad.replace(bad.find("%s"), 2, "hat");
    m = LogicError(LogicJson(bad, crop_edges));
    GRAPH_CHECK(m.find("give \"count\", \"with:<label>\" or \"without:<label>\"") !=
                std::string::npos);
    // A user node passes its input through.
    m = LogicError(LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"mine\",\"input\":\"boxes\",\"output\":\"records\"}}",
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"output\" is \"records\", and a user node passes its input through") !=
                std::string::npos);
    m = LogicError(LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"mine\",\"input\":\"scores\",\"output\":\"scores\"}}",
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"input\" is \"scores\", but \"od\" gives boxes") != std::string::npos);
}

void TestValidateChecksTheGalleryFile() {
    const std::string mine = WriteGallery("emb.bin", "emb", AliceAndBob(), AliceAndBobLabels());
    const std::string theirs = WriteGallery("other.bin", "other", AliceAndBob(), AliceAndBobLabels());
    const std::string edges =
        ",{\"from\":\"od\",\"to\":\"e\",\"roi\":{}},{\"from\":\"e\",\"to\":\"g\",\"carry\":\"result\"}";
    const auto graph = [](const std::string& path) {
        return ",{\"id\":\"e\",\"model\":\"emb\"},{\"id\":\"g\",\"type\":\"logic\",\"op\":"
               "\"gallery\",\"params\":{\"gallery\":\"" + path + "\"}}";
    };
    GRAPH_CHECK(LogicError(LogicJson(graph(mine), edges)).empty());
    std::string m = LogicError(LogicJson(graph(theirs), edges));
    GRAPH_CHECK(m.find("the gallery was built by \"other\", but \"e\" runs \"emb\"") !=
                std::string::npos);
    m = LogicError(LogicJson(graph(ScratchPath("absent.bin")), edges));
    GRAPH_CHECK(m.find("gallery not found") != std::string::npos);
    GRAPH_CHECK(m.find("python3 scripts/build_gallery_database.py") != std::string::npos);
    std::remove(mine.c_str());
    std::remove(theirs.c_str());
}

/// A gallery whose header promises more rows than the file holds fails as a
/// load error, before anything the size of the header's claim is allocated: a
/// truncated or crafted .bin must not end the run with std::bad_alloc.
void TestAGalleryLargerThanItsFileIsRefusedBeforeAllocating() {
    const std::string path = ScratchPath("crafted.bin");
    const auto write = [&path](std::uint32_t n, std::uint32_t dim, std::size_t floats,
                               std::uint32_t blob) {
        std::ofstream out(path.c_str(), std::ios::binary | std::ios::trunc);
        out.write("DXGAL1\0\0", 8);
        const std::uint32_t words[2] = {n, dim};
        out.write(reinterpret_cast<const char*>(words), sizeof(words));   // little-endian hosts
        const std::vector<float> rows(floats, 0.5f);
        out.write(reinterpret_cast<const char*>(rows.data()),
                  static_cast<std::streamsize>(rows.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(&blob), sizeof(blob));
    };
    write(0xFFFFFFFFu, 0xFFFFFFFFu, 4, 0);
    RetrievalGallery g;
    bool threw = false;
    try {
        g = loadRetrievalGallery(path);
    } catch (...) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
    GRAPH_CHECK(!g.loaded);
    GRAPH_CHECK(g.error.find("truncated embeddings") != std::string::npos);

    // the rows fit, but a metadata blob claims 256 MB the file does not have
    write(1, 4, 4, 0x10000000u);
    threw = false;
    try {
        g = loadRetrievalGallery(path);
    } catch (...) {
        threw = true;
    }
    GRAPH_CHECK(!threw);
    GRAPH_CHECK(!g.loaded);
    GRAPH_CHECK(g.error.find("truncated metadata") != std::string::npos);
    std::remove(path.c_str());
}

void TestNodeOutputShapeFollowsFilterZoneAndUser() {
    const GraphSpec spec = ParseGraphText(LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\"}" + std::string(kZoneNode) +
            ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"count\"}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"},"
        "{\"from\":\"f\",\"to\":\"z\",\"carry\":\"result\"},"
        "{\"from\":\"z\",\"to\":\"c\",\"carry\":\"result\"}"), "s.json");
    const FakeModelRegistry registry = LogicCheckRegistry();
    GRAPH_CHECK(NodeOutputShape(spec, registry, "z") == Shape::kBoxes);
    GRAPH_CHECK(NodeOutputShape(spec, registry, "c") == Shape::kRecords);
    GRAPH_CHECK(NodeOutputShape(spec, registry, "cam") == Shape::kFrame);
    GRAPH_CHECK(BoxSourceNode(spec, "z") != NULL && BoxSourceNode(spec, "z")->id == "od");
    GRAPH_CHECK(BoxSourceNode(spec, "c") == NULL);
}

void TestFrameTimeCountsFramesOrWallSeconds() {
    FakeModelRegistry registry = BuildCascadeRegistry();
    StageGraph graph;
    graph.Build(ParseGraphText(CascadeJson(), "t.json"), registry, "/models", false);
    GRAPH_CHECK(graph.FrameTime(0, 15) == 15.0 / kDefaultStreamFps);
    graph.SetStreamClock(0, 10.0, false);
    GRAPH_CHECK(graph.FrameTime(0, 15) == 15.0 / 10.0);
    graph.SetStreamClock(0, 0.0, false);  // an image: no rate, the default
    GRAPH_CHECK(graph.FrameTime(0, 3) == 3.0 / kDefaultStreamFps);
    graph.SetStreamClock(0, 25.0, true);  // a live source: seconds since its first frame
    const double first = graph.FrameTime(0, 100);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    const double later = graph.FrameTime(0, 101);
    GRAPH_CHECK(first >= 0.0 && first < 0.02);
    GRAPH_CHECK(later >= 0.025 && later < 5.0);
    bool threw = false;
    try {
        graph.SetStreamClock(5, 10.0, false);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    GRAPH_CHECK(threw);
}

void TestReportsCarryTheFrameTime() {
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
    FakeModelRegistry sync_registry = BuildCascadeRegistry();
    StageGraph sync_graph;
    sync_graph.Build(ParseGraphText(CascadeJson(), "t.json"), sync_registry, "/models", false);
    sync_graph.SetStreamClock(0, 10.0, false);
    const FrameReport expected = SyncExecutor().RunFrame(sync_graph, frame, 12);
    FakeModelRegistry async_registry = BuildCascadeRegistry();
    StageGraph async_graph;
    async_graph.Build(ParseGraphText(CascadeJson(), "t.json"), async_registry, "/models", false);
    async_graph.SetStreamClock(0, 10.0, false);
    const FrameReport actual = AsyncExecutor(StageJobs(4)).RunFrame(async_graph, frame, 12);
    GRAPH_CHECK(expected.time_s == 12.0 / 10.0);
    GRAPH_CHECK(expected == actual);
    GRAPH_CHECK(expected.events.empty() && expected.overlays.empty());
}

void TestOperatorEqualityComparesTimeEventsAndOverlays() {
    FrameReport a;
    a.time_s = 0.5;
    GraphEvent event;
    event.node = "z";
    event.event = "enter";
    event.track_id = 3;
    event.time_s = 0.5;
    event.data.numbers.push_back(std::make_pair(std::string("count"), 1.0));
    a.events.push_back(event);
    LogicOverlay overlay;
    overlay.polygons.push_back(std::vector<cv::Point2f>(3, cv::Point2f(1.f, 2.f)));
    a.overlays["z"] = overlay;
    FrameReport b = a;
    GRAPH_CHECK(a == b);
    b.time_s = 0.6;
    GRAPH_CHECK(!(a == b));
    b = a;
    b.events[0].track_id = 4;
    GRAPH_CHECK(!(a == b));
    b = a;
    b.events[0].data.numbers[0].second = 2.0;
    GRAPH_CHECK(!(a == b));
    b = a;
    b.overlays["z"].alert = true;
    GRAPH_CHECK(!(a == b));
    b = a;
    b.overlays["y"] = overlay;
    GRAPH_CHECK(!(a == b));
}

void TestReportToJsonWritesEventsOnlyWhenThereAreSome() {
    FrameReport report;
    report.frame_index = 7;
    report.stream = "cam";
    report.time_s = 1.23449;
    GRAPH_CHECK(!consumer::ReportToJson(report, false).contains("events"));
    GraphEvent event;
    event.node = "z";
    event.event = "enter";
    event.track_id = 3;
    event.time_s = 1.23449;
    event.data.numbers.push_back(std::make_pair(std::string("count"), 1.0));
    event.data.text.push_back(std::make_pair(std::string("name"), std::string("bob")));
    report.events.push_back(event);
    const nlohmann::json json = consumer::ReportToJson(report, false);
    GRAPH_CHECK(json.contains("events") && json["events"].size() == 1);
    GRAPH_CHECK(json["events"][0].dump() ==
                "{\"data\":{\"count\":1,\"name\":\"bob\"},\"event\":\"enter\",\"node\":\"z\","
                "\"time\":1.234,\"track_id\":3}");
    GRAPH_CHECK(consumer::EventLineJson(event, report).dump() ==
                "{\"frame\":7,\"time\":1.234,\"stream\":\"cam\",\"node\":\"z\",\"event\":\"enter\","
                "\"track_id\":3,\"data\":{\"count\":1,\"name\":\"bob\"}}");
    event.track_id = -1;
    event.data.numbers[0].second = std::numeric_limits<double>::quiet_NaN();
    GRAPH_CHECK(consumer::EventLineJson(event, report).dump().find(
                    "\"track_id\":-1,\"data\":{\"count\":\"NaN\"") != std::string::npos);
}

namespace {

const BoxesData* BoxesOf(const FrameReport& report, const std::string& id) {
    std::map<std::string, StageResult>::const_iterator it = report.node_results.find(id);
    if (it == report.node_results.end() || !it->second.data) return NULL;
    return dynamic_cast<const BoxesData*>(it->second.data.get());
}

/// One 40x60 "person" at (x, 200), or nobody for x < 0. Positions 10 px
/// apart overlap with IoU 0.6, so the IoU tracker keeps one id.
std::shared_ptr<BoxesData> LogicPersonAt(float x) {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    if (x >= 0.f) boxes->items.push_back(MakeItem(x, 200.f, 40.f, 60.f, 0.9f, "person"));
    return boxes;
}

/// Six frames (keys 0..5) of a 640x480 scene: one person walks right across
/// x = 320, the frame's middle, and back. Bottom-centre anchors x: 305,
/// 315, 325, 335, 325, 315 (y 260).
const float kWalk[] = {285.f, 295.f, 305.f, 315.f, 305.f, 295.f};

/// LogicCheckRegistry() plus the walk scripted on "det" and a "Jacket" on
/// every "attr" crop.
FakeModelRegistry LogicRegistry() {
    FakeModelRegistry registry = LogicCheckRegistry();
    for (int k = 0; k < 6; ++k) registry.SetScriptByFirstPixel("det", k, LogicPersonAt(kWalk[k]));
    std::shared_ptr<ScoresData> jacket(new ScoresData());
    ClassificationResult item;
    item.class_id = 7;
    item.class_name = "Jacket";
    item.confidence = 0.9f;
    jacket->items.push_back(item);
    registry.SetScriptByFirstPixel("attr", 0, jacket);
    return registry;
}

std::vector<int> Keys(int count) {
    std::vector<int> keys;
    for (int i = 0; i < count; ++i) keys.push_back(i);
    return keys;
}

/// `text` over frames FrameWithKey(keys[i]), stream clock 10 fps (frame k
/// is at k / 10 s), through one executor.
std::vector<FrameReport> RunLogicFrames(const std::string& text, FakeModelRegistry registry,
                                        const std::vector<int>& keys, bool async) {
    const GraphSpec spec = ParseGraphText(text, "logic.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    graph.SetStreamClock(0, 10.0, false);
    std::vector<FrameReport> reports;
    if (!async) {
        SyncExecutor executor;
        for (std::size_t i = 0; i < keys.size(); ++i) {
            reports.push_back(executor.RunFrame(graph, FrameWithKey(keys[i]), i));
        }
        return reports;
    }
    AsyncExecutor executor(Frames(4));
    for (std::size_t i = 0; i < keys.size(); ++i) executor.Submit(graph, FrameWithKey(keys[i]), i);
    executor.Finish(graph);
    FrameReport report;
    while (executor.TryNext(&report)) reports.push_back(report);
    return reports;
}

}  // namespace

void TestFilterKeepsClassScoreAndArea() {
    NodeSpec node;
    node.id = "f";
    node.is_logic = true;
    node.op = "filter";
    node.params.lists["classes"].push_back("person");
    node.params.numeric["min_score"] = 0.5;
    node.params.numeric["min_area"] = 1000.0;
    const LogicConfig config = MakeLogicConfig(node);
    GRAPH_CHECK(config.op == "filter" && config.numbers.at("min_area") == 1000.0);
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(0, 0, 40, 60, 0.9f, "person"));  // kept
    boxes->items.push_back(MakeItem(0, 0, 40, 60, 0.9f, "car"));     // class
    boxes->items.push_back(MakeItem(0, 0, 40, 60, 0.3f, "person"));  // score
    boxes->items.push_back(MakeItem(0, 0, 10, 10, 0.9f, "person"));  // area 100
    LogicInput input;
    input.from = "od";
    input.data = boxes;
    LogicFrame at;
    at.frame_size = cv::Size(640, 480);
    LogicState state;
    const LogicOutput out = RunLogic(config, std::vector<LogicInput>(1, input), at, &state);
    GRAPH_CHECK(out.error.empty());
    const BoxesData* kept = dynamic_cast<const BoxesData*>(out.result.data.get());
    GRAPH_CHECK(kept != NULL && kept->items.size() == 1 && kept->box_shape == Shape::kBoxes);
    GRAPH_CHECK(out.events.empty() && out.overlay.polygons.empty());
    // Boxes from a handed-off x2 image come back to source pixels first:
    // 40x60 becomes 20x30, under min_area.
    LogicInput scaled = input;
    scaled.origin.inv_align = cv::Matx23f(0.5f, 0.f, 0.f, 0.f, 0.5f, 0.f);
    const LogicOutput half = RunLogic(config, std::vector<LogicInput>(1, scaled), at, &state);
    const BoxesData* restored = dynamic_cast<const BoxesData*>(half.result.data.get());
    GRAPH_CHECK(restored != NULL && restored->items.empty());
    const LogicOutput none = RunLogic(config, std::vector<LogicInput>(), at, &state);
    GRAPH_CHECK(none.error == "missing boxes input");
    GRAPH_CHECK(dynamic_cast<const BoxesData*>(none.result.data.get()) != NULL);
    // The catalog's defaults fill what the graph leaves out.
    NodeSpec bare = node;
    bare.params = StageParams();
    const LogicConfig defaults = MakeLogicConfig(bare);
    GRAPH_CHECK(defaults.numbers.at("min_score") == 0.0 && defaults.lists.count("classes") == 0);
}

void TestZoneKeepsTheBoxesInsideAndRaisesEnterAndExit() {
    const std::string text = LogicJson(
        std::string(kZoneNode) + ",{\"id\":\"a\",\"model\":\"attr\"}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"a\",\"roi\":{}}");
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    const std::size_t inside[] = {0, 0, 1, 1, 1, 0};
    for (std::size_t i = 0; i < 6; ++i) {
        GRAPH_CHECK(reports[i].error.empty());
        const BoxesData* z = BoxesOf(reports[i], "z");
        GRAPH_CHECK(z != NULL && z->items.size() == inside[i]);
        GRAPH_CHECK(reports[i].roi_results.count("a") == 1 &&
                    reports[i].roi_results.at("a").size() == inside[i]);
        GRAPH_CHECK(reports[i].overlays.count("z") == 1 &&
                    reports[i].overlays.at("z").polygons.size() == 1);
        GRAPH_CHECK(reports[i].time_s == static_cast<double>(i) / 10.0);
        GRAPH_CHECK(reports[i].node_ms.count("z") == 1);
    }
    GRAPH_CHECK(reports[2].events.size() == 1);
    if (reports[2].events.size() == 1) {
        const GraphEvent& enter = reports[2].events[0];
        GRAPH_CHECK(enter.node == "z" && enter.event == "enter" && enter.track_id == 0);
        GRAPH_CHECK(enter.time_s == 2.0 / 10.0);
    }
    GRAPH_CHECK(reports[3].events.empty() && reports[4].events.empty());
    GRAPH_CHECK(reports[5].events.size() == 1 && reports[5].events[0].event == "exit");
    // The polygon in source pixels; the crop comes from the zone, with its track.
    GRAPH_CHECK(reports[0].overlays.at("z").polygons[0][0] == cv::Point2f(320.f, 0.f));
    GRAPH_CHECK(reports[2].roi_results.at("a")[0].origin.parent_node == "z");
    GRAPH_CHECK(reports[2].roi_results.at("a")[0].origin.track_id == 0);
    GRAPH_CHECK(reports[2].overlays.at("z").alert && !reports[0].overlays.at("z").alert);
}

void TestBuildGivesEachLogicNodeAStatePerStream() {
    const std::string text =
        "{\"version\":1,\"nodes\":[{\"id\":\"cam1\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"det\",\"track\":{\"algo\":\"iou\"}}" + std::string(kZoneNode) +
        "],\"edges\":[{\"from\":\"cam1\",\"to\":\"od\"},{\"from\":\"cam2\",\"to\":\"od\"}" +
        std::string(kOdToZ) + "]}";
    FakeModelRegistry registry = LogicCheckRegistry();
    const GraphSpec spec = ParseGraphText(text, "two.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    const NodeRuntime& z = graph.nodes()[3];
    GRAPH_CHECK(z.is_logic && z.logic && z.logic->op == "zone" && !z.stage);
    GRAPH_CHECK(z.logic_states.size() == 2 && z.logic_states[0] && z.logic_states[1]);
    GRAPH_CHECK(z.logic_states.size() == 2 && z.logic_states[0] != z.logic_states[1]);
    GRAPH_CHECK(z.output_shape == Shape::kBoxes);
}

// A logic node runs on the driver thread, frame-ordered per stream; its
// results, events and overlays equal SyncExecutor's whatever order the
// detector's completions come back in.
void TestAsyncLogicMatchesSyncOverFrames() {
    const std::string text = LogicJson(
        std::string(kZoneNode) + ",{\"id\":\"a\",\"model\":\"attr\"}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"a\",\"roi\":{}}");
    const std::vector<FrameReport> expected = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    FakeModelRegistry reversed = LogicRegistry();
    reversed.SetDeliveryOrder("det", FakeModelRegistry::kReverse);
    reversed.SetShuffleSeed("attr", 5);
    const std::vector<FrameReport> actual = RunLogicFrames(text, reversed, Keys(6), true);
    GRAPH_CHECK(actual.size() == expected.size());
    for (std::size_t i = 0; i < expected.size() && i < actual.size(); ++i) {
        GRAPH_CHECK(actual[i].frame_index == i);
        GRAPH_CHECK(expected[i] == actual[i]);
        GRAPH_CHECK(actual[i].node_ms.count("z") == 1);
    }
}

// Two zones raise events in one frame: both executors list them in the
// stream's topological order.
void TestAsyncListsEventsInNodeOrder() {
    const std::string text = LogicJson(
        ",{\"id\":\"right\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0.5,0,1,0,1,1,0.5,1]}}"
        ",{\"id\":\"left\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0,0,0.5,0,0.5,1,0,1]}}",
        ",{\"from\":\"od\",\"to\":\"right\",\"carry\":\"result\"},"
        "{\"from\":\"od\",\"to\":\"left\",\"carry\":\"result\"}");
    const std::vector<FrameReport> expected = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    const std::vector<FrameReport> actual = RunLogicFrames(text, LogicRegistry(), Keys(6), true);
    GRAPH_CHECK(expected.size() == 6 && actual.size() == 6);
    if (expected.size() != 6 || actual.size() != 6) return;
    // Frame 2: the person crosses from left to right.
    GRAPH_CHECK(expected[2].events.size() == 2);
    if (expected[2].events.size() == 2) {
        GRAPH_CHECK(expected[2].events[0].node == "right" && expected[2].events[0].event == "enter");
        GRAPH_CHECK(expected[2].events[1].node == "left" && expected[2].events[1].event == "exit");
    }
    for (std::size_t i = 0; i < 6; ++i) GRAPH_CHECK(expected[i] == actual[i]);
}

// A logic node's boxes reach a cpu node on a "carry": "result" edge, in both
// executors, as a model's would: headpose behind a filter sees only the
// filter's boxes, so the face is face 0. Unfiltered, the landmark-less "car"
// comes first and writes no row: the face is face 1.
void TestCpuNodeBehindAFilterGetsOnlyTheFilteredBoxes() {
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(0.f, 0.f, 300.f, 300.f, 0.9f, "car"));
    BoxItem face = MakeItem(200.f, 120.f, 80.f, 100.f, 0.9f, "face");
    face.landmarks.push_back(Keypoint(220.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(260.f, 150.f, 1.f));
    face.landmarks.push_back(Keypoint(240.f, 180.f, 1.f));
    face.landmarks.push_back(Keypoint(225.f, 200.f, 1.f));
    face.landmarks.push_back(Keypoint(255.f, 200.f, 1.f));
    boxes->items.push_back(face);
    const std::string head =
        "{\"version\":1,\"name\":\"filtered_pose\",\"nodes\":["
        "{\"id\":\"cam\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"face\",\"model\":\"face\"},";
    const std::string filtered = head +
        "{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\",\"params\":{\"classes\":[\"face\"]}},"
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"f\",\"carry\":\"result\"},"
        "{\"from\":\"f\",\"to\":\"pose\",\"carry\":\"result\"}]}";
    const std::string direct = head +
        "{\"id\":\"pose\",\"type\":\"cpu\",\"op\":\"headpose\"}],"
        "\"edges\":[{\"from\":\"cam\",\"to\":\"face\"},"
        "{\"from\":\"face\",\"to\":\"pose\",\"carry\":\"result\"}]}";
    const cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);

    FakeModelRegistry sync_registry;
    sync_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport expected = RunSync(filtered, &sync_registry, frame);
    FakeModelRegistry async_registry;
    async_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport actual = RunAsync(filtered, &async_registry, frame);

    GRAPH_CHECK(expected.error.empty());
    GRAPH_CHECK(expected == actual);
    const BoxesData* kept = BoxesOf(expected, "f");
    GRAPH_CHECK(kept != NULL && kept->items.size() == 1);
    const RecordsData* records = RecordsOf(expected, "pose");
    GRAPH_CHECK(records != NULL && records->items.size() == 1);
    double index = -1.0;
    if (records != NULL && records->items.size() == 1) {
        double yaw = 0.0;
        GRAPH_CHECK(NamedNumber(records->items[0], "yaw", &yaw) && std::isfinite(yaw));
        GRAPH_CHECK(NamedNumber(records->items[0], "face_index", &index) && index == 0.0);
    }
    FakeModelRegistry direct_registry;
    direct_registry.AddModel(CpuFixtureModel("face", Shape::kBoxes), boxes);
    const FrameReport unfiltered = RunSync(direct, &direct_registry, frame);
    const RecordsData* direct_records = RecordsOf(unfiltered, "pose");
    GRAPH_CHECK(direct_records != NULL && direct_records->items.size() == 1);
    if (direct_records != NULL && direct_records->items.size() == 1) {
        GRAPH_CHECK(NamedNumber(direct_records->items[0], "face_index", &index) && index == 1.0);
    }
}

// An event's "data" is one JSON object: numbers then text. A key used twice
// would let the later value replace the earlier one silently, so RunLogic
// keeps the first and reports the repeat.
void TestEventDataKeepsEachKeyOnce() {
    RecordItem data;
    data.numbers.push_back(std::make_pair(std::string("count"), 1.0));
    data.text.push_back(std::make_pair(std::string("name"), std::string("bob")));
    GRAPH_CHECK(DropRepeatedKeys(&data).empty());
    GRAPH_CHECK(data.numbers.size() == 1 && data.text.size() == 1);
    data.text.push_back(std::make_pair(std::string("count"), std::string("two")));
    data.numbers.push_back(std::make_pair(std::string("count"), 3.0));
    data.numbers.push_back(std::make_pair(std::string("score"), 0.5));
    GRAPH_CHECK(DropRepeatedKeys(&data) == "count");
    GRAPH_CHECK(data.numbers.size() == 2 && data.numbers[0].second == 1.0 &&
                data.numbers[1].first == "score");
    GRAPH_CHECK(data.text.size() == 1 && data.text[0].first == "name");
    GRAPH_CHECK(DropRepeatedKeys(&data).empty());
}

namespace {

/// A vertical line through the frame's middle, from A (bottom) to B (top):
/// its right is the image's right, so walking right is "in".
const char kLineNode[] =
    ",{\"id\":\"l\",\"type\":\"logic\",\"op\":\"line\",\"params\":{\"line\":[0.5,1,0.5,0]}}";
const char kOdToL[] = ",{\"from\":\"od\",\"to\":\"l\",\"carry\":\"result\"}";

double NumberOf(const RecordItem& item, const std::string& name) {
    double value = -1.0;
    NamedNumber(item, name, &value);
    return value;
}

FakeModelRegistry ReversedLogicRegistry() {
    FakeModelRegistry registry = LogicRegistry();
    registry.SetDeliveryOrder("det", FakeModelRegistry::kReverse);
    return registry;
}

}  // namespace

void TestLineCountsCrossingsBothWays() {
    const std::string text = LogicJson(kLineNode, kOdToL);
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    const double in[] = {0, 0, 1, 1, 1, 1};
    const double out[] = {0, 0, 0, 0, 0, 1};
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(reports[i], "l");
        GRAPH_CHECK(records != NULL && records->items.size() == 1);
        if (records == NULL || records->items.size() != 1) continue;
        GRAPH_CHECK(NumberOf(records->items[0], "in") == in[i]);
        GRAPH_CHECK(NumberOf(records->items[0], "out") == out[i]);
        GRAPH_CHECK(NumberOf(records->items[0], "total") == in[i] + out[i]);
        GRAPH_CHECK(reports[i].overlays.at("l").lines.size() == 1);
        if (i != 2 && i != 5) GRAPH_CHECK(reports[i].events.empty());
    }
    GRAPH_CHECK(reports[2].events.size() == 1 && reports[5].events.size() == 1);
    if (reports[2].events.size() == 1 && reports[5].events.size() == 1) {
        GRAPH_CHECK(reports[2].events[0].event == "cross" && reports[2].events[0].track_id == 0);
        GRAPH_CHECK(reports[2].events[0].data.text[0].second == "in");
        GRAPH_CHECK(reports[5].events[0].data.text[0].second == "out");
        GRAPH_CHECK(NumberOf(reports[5].events[0].data, "out") == 1.0);
    }
    GRAPH_CHECK(reports[0].overlays.at("l").lines[0].first == cv::Point2f(320.f, 480.f));
    // "direction": "in" raises only the inward event; the counts keep both.
    const std::vector<FrameReport> in_only = RunLogicFrames(
        LogicJson(",{\"id\":\"l\",\"type\":\"logic\",\"op\":\"line\",\"params\":"
                  "{\"line\":[0.5,1,0.5,0],\"direction\":\"in\"}}", kOdToL),
        LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(in_only.size() == 6 && in_only[2].events.size() == 1 && in_only[5].events.empty());
    const std::vector<FrameReport> async = RunLogicFrames(text, ReversedLogicRegistry(), Keys(6), true);
    GRAPH_CHECK(async.size() == 6);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
}

void TestCountCountsEachInputAndClass() {
    const std::string text = LogicJson(
        ",{\"id\":\"left\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0,0,0.5,0,0.5,1,0,1]}}"
        ",{\"id\":\"right\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0.5,0,1,0,1,1,0.5,1]}}"
        ",{\"id\":\"n\",\"type\":\"logic\",\"op\":\"count\",\"params\":{\"classes\":[\"person\",\"car\"]}}",
        ",{\"from\":\"od\",\"to\":\"left\",\"carry\":\"result\"},"
        "{\"from\":\"od\",\"to\":\"right\",\"carry\":\"result\"},"
        "{\"from\":\"right\",\"to\":\"n\",\"carry\":\"result\"},"
        "{\"from\":\"left\",\"to\":\"n\",\"carry\":\"result\"}");
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    const double right[] = {0, 0, 1, 1, 1, 0};
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(reports[i], "n");
        GRAPH_CHECK(records != NULL && records->items.size() == 2);
        if (records == NULL || records->items.size() != 2) continue;
        // Inputs in edge order: right -> n is declared before left -> n.
        GRAPH_CHECK(records->items[0].text.size() == 1 &&
                    records->items[0].text[0].second == "right");
        GRAPH_CHECK(records->items[1].text[0].second == "left");
        GRAPH_CHECK(NumberOf(records->items[0], "total") == right[i]);
        GRAPH_CHECK(NumberOf(records->items[0], "person") == right[i]);
        GRAPH_CHECK(NumberOf(records->items[0], "car") == 0.0);
        GRAPH_CHECK(NumberOf(records->items[1], "total") == 1.0 - right[i]);
        GRAPH_CHECK(reports[i].overlays.at("n").texts.size() == 2);
    }
    const std::vector<FrameReport> async = RunLogicFrames(text, ReversedLogicRegistry(), Keys(6), true);
    GRAPH_CHECK(async.size() == 6);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
}

namespace {

std::string ConditionNode(const std::string& params) {
    return ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"condition\",\"params\":" + params + "}";
}

std::vector<std::size_t> FramesWithEvents(const std::vector<FrameReport>& reports) {
    std::vector<std::size_t> frames;
    for (std::size_t i = 0; i < reports.size(); ++i) {
        if (!reports[i].events.empty()) frames.push_back(i);
    }
    return frames;
}

}  // namespace

void TestDwellTimesEachStayAndRaisesOneEvent() {
    const std::string text = LogicJson(
        std::string(kZoneNode) +
            ",{\"id\":\"d\",\"type\":\"logic\",\"op\":\"dwell\",\"params\":"
            "{\"threshold_s\":0.15,\"grace_s\":1}}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"d\",\"carry\":\"result\"}");
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    const std::size_t rows[] = {0, 0, 1, 1, 1, 0};
    const double seconds[] = {0, 0, 0.0, 0.1, 0.2, 0};
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(reports[i], "d");
        GRAPH_CHECK(records != NULL && records->items.size() == rows[i]);
        if (records == NULL || records->items.size() != 1) continue;
        GRAPH_CHECK(NumberOf(records->items[0], "track_id") == 0.0);
        GRAPH_CHECK(std::fabs(NumberOf(records->items[0], "seconds") - seconds[i]) < 1e-9);
    }
    // Frames 2 (zone enter) and 4 (0.2 s >= 0.15 s): one dwell event, once.
    std::size_t dwell_events = 0;
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t e = 0; e < reports[i].events.size(); ++e) {
            if (reports[i].events[e].event != "dwell") continue;
            ++dwell_events;
            GRAPH_CHECK(i == 4 && reports[i].events[e].node == "d");
            GRAPH_CHECK(std::fabs(NumberOf(reports[i].events[e].data, "seconds") - 0.2) < 1e-9);
        }
    }
    GRAPH_CHECK(dwell_events == 1);
    GRAPH_CHECK(reports[4].overlays.at("d").highlights.size() == 1);
    GRAPH_CHECK(reports[3].overlays.at("d").highlights.empty());
    const std::vector<FrameReport> async = RunLogicFrames(text, ReversedLogicRegistry(), Keys(6), true);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
}

void TestConditionRaisesOnTheRisingEdgeOrEveryCooldown() {
    const std::string edges =
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"c\",\"carry\":\"result\"}";
    const std::string rising = LogicJson(
        std::string(kZoneNode) + ConditionNode("{\"event\":\"intrusion\"}"), edges);
    const std::vector<FrameReport> reports = RunLogicFrames(rising, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    // Zone events (enter at 2, exit at 5) come first: z precedes c.
    GRAPH_CHECK(reports[2].events.size() == 2);
    if (reports[2].events.size() == 2) {
        const GraphEvent& alarm = reports[2].events[1];
        GRAPH_CHECK(alarm.node == "c" && alarm.event == "intrusion" && alarm.track_id == 0);
        GRAPH_CHECK(NumberOf(alarm.data, "value") == 1.0 && NumberOf(alarm.data, "threshold") == 0.0);
        GRAPH_CHECK(alarm.data.text.size() == 1 && alarm.data.text[0].first == "track_ids" &&
                    alarm.data.text[0].second == "0");
    }
    const double active[] = {0, 0, 1, 1, 1, 0};
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(reports[i], "c");
        GRAPH_CHECK(records != NULL && records->items.size() == 1);
        if (records != NULL && records->items.size() == 1) {
            GRAPH_CHECK(NumberOf(records->items[0], "active") == active[i]);
        }
        GRAPH_CHECK(reports[i].overlays.at("c").texts.size() == 1);
    }
    // cooldown 0.15 s: frame 2, then frame 4 (0.2 s later), not frame 3.
    const std::vector<FrameReport> repeated = RunLogicFrames(LogicJson(
        std::string(kZoneNode) + ConditionNode("{\"event\":\"intrusion\",\"cooldown_s\":0.15}"),
        edges), LogicRegistry(), Keys(6), false);
    std::vector<std::size_t> alarms;
    for (std::size_t i = 0; i < repeated.size(); ++i) {
        for (std::size_t e = 0; e < repeated[i].events.size(); ++e) {
            if (repeated[i].events[e].node == "c") alarms.push_back(i);
        }
    }
    GRAPH_CHECK(alarms.size() == 2 && alarms[0] == 2 && alarms[1] == 4);
    const std::vector<FrameReport> async = RunLogicFrames(rising, ReversedLogicRegistry(), Keys(6), true);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
}

// An untracked detector whose completions come back in reverse: only the
// per-stream gate keeps a condition's rising edge on frame 1. Without it the
// condition would see frame 2 first and raise its event there.
void TestAsyncLogicGateOrdersReversedCompletions() {
    FakeModelRegistry registry = LogicCheckRegistry();
    registry.SetScriptByFirstPixel("det", 0, LogicPersonAt(-1.f));
    registry.SetScriptByFirstPixel("det", 1, LogicPersonAt(100.f));
    registry.SetScriptByFirstPixel("det", 2, LogicPersonAt(110.f));
    const std::string text = LogicJson(ConditionNode("{\"event\":\"seen\"}"),
                                       ",{\"from\":\"od\",\"to\":\"c\",\"carry\":\"result\"}", false);
    const std::vector<FrameReport> expected = RunLogicFrames(text, registry, Keys(3), false);
    const std::vector<std::size_t> fired = FramesWithEvents(expected);
    GRAPH_CHECK(fired.size() == 1 && fired[0] == 1);
    FakeModelRegistry reversed = registry;
    reversed.SetDeliveryOrder("det", FakeModelRegistry::kReverse);
    const std::vector<FrameReport> actual = RunLogicFrames(text, reversed, Keys(3), true);
    GRAPH_CHECK(actual.size() == 3);
    for (std::size_t i = 0; i < 3 && i < actual.size(); ++i) GRAPH_CHECK(expected[i] == actual[i]);
}

void TestPerCropScoresReachACondition() {
    const std::string edges =
        ",{\"from\":\"od\",\"to\":\"a\",\"roi\":{}},{\"from\":\"a\",\"to\":\"c\",\"carry\":\"result\"}";
    const auto graph = [&edges](const std::string& value) {
        return LogicJson(",{\"id\":\"a\",\"model\":\"attr\"}" +
                             ConditionNode("{\"value\":\"" + value + "\",\"event\":\"no_hat\"}"),
                         edges);
    };
    // Every crop wears a Jacket and no Hat.
    const std::vector<FrameReport> without = RunLogicFrames(graph("without:Hat"), LogicRegistry(),
                                                            Keys(6), false);
    GRAPH_CHECK(without.size() == 6);
    if (without.size() != 6) return;
    const std::vector<std::size_t> fired = FramesWithEvents(without);
    GRAPH_CHECK(fired.size() == 1 && fired[0] == 0);
    if (!without[0].events.empty()) {
        GRAPH_CHECK(without[0].events[0].event == "no_hat" && without[0].events[0].track_id == 0);
    }
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(without[i], "c");
        GRAPH_CHECK(records != NULL && records->items.size() == 1 &&
                    NumberOf(records->items[0], "value") == 1.0);
    }
    const std::vector<FrameReport> with_hat = RunLogicFrames(graph("with:Hat"), LogicRegistry(),
                                                             Keys(6), false);
    GRAPH_CHECK(FramesWithEvents(with_hat).empty());
    const std::vector<FrameReport> with_jacket = RunLogicFrames(graph("with:Jacket"),
                                                                LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(FramesWithEvents(with_jacket).size() == 1);
    FakeModelRegistry shuffled = ReversedLogicRegistry();
    shuffled.SetShuffleSeed("attr", 9);
    const std::vector<FrameReport> async = RunLogicFrames(graph("without:Hat"), shuffled, Keys(6), true);
    GRAPH_CHECK(async.size() == 6);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(without[i] == async[i]);
}

namespace {

std::shared_ptr<VectorData> Embedding(float a, float b, float c, float d) {
    std::shared_ptr<VectorData> vector(new VectorData());
    vector->values.push_back(a);
    vector->values.push_back(b);
    vector->values.push_back(c);
    vector->values.push_back(d);
    return vector;
}

std::string GalleryGraph(const std::string& path, const std::string& extra) {
    return LogicJson(
        ",{\"id\":\"e\",\"model\":\"emb\"},{\"id\":\"g\",\"type\":\"logic\",\"op\":\"gallery\","
        "\"params\":{\"gallery\":\"" + path + "\",\"threshold\":0.5" + extra + "}}",
        ",{\"from\":\"od\",\"to\":\"e\",\"roi\":{}},{\"from\":\"e\",\"to\":\"g\",\"carry\":\"result\"}");
}

std::string TextOf(const RecordItem& item, const std::string& name) {
    for (std::size_t i = 0; i < item.text.size(); ++i) {
        if (item.text[i].first == name) return item.text[i].second;
    }
    return std::string();
}

}  // namespace

void TestGalleryNamesEachCropAndAnnouncesNameChanges() {
    const std::string path =
        WriteGallery("logic_gallery.bin", "emb", AliceAndBob(), AliceAndBobLabels());
    FakeModelRegistry registry = LogicRegistry();
    registry.SetScriptByFirstPixel("emb", 0, Embedding(1.f, 0.f, 0.f, 0.f));
    const std::vector<FrameReport> reports =
        RunLogicFrames(GalleryGraph(path, ""), registry, Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(reports[i], "g");
        GRAPH_CHECK(records != NULL && records->items.size() == 1);
        if (records == NULL || records->items.size() != 1) continue;
        GRAPH_CHECK(TextOf(records->items[0], "name") == "alice");
        GRAPH_CHECK(NumberOf(records->items[0], "score") == 1.0);
        GRAPH_CHECK(NumberOf(records->items[0], "track_id") == 0.0);
        GRAPH_CHECK(reports[i].overlays.at("g").texts.size() == 1 &&
                    reports[i].overlays.at("g").texts[0].text == "alice 1");
    }
    // Announced once, when track 0 first gets its name.
    const std::vector<std::size_t> fired = FramesWithEvents(reports);
    GRAPH_CHECK(fired.size() == 1 && fired[0] == 0);
    if (!reports[0].events.empty()) {
        GRAPH_CHECK(reports[0].events[0].event == "match" && reports[0].events[0].track_id == 0);
        GRAPH_CHECK(TextOf(reports[0].events[0].data, "name") == "alice");
    }
    FakeModelRegistry reversed = registry;
    reversed.SetDeliveryOrder("det", FakeModelRegistry::kReverse);
    reversed.SetShuffleSeed("emb", 4);
    const std::vector<FrameReport> async =
        RunLogicFrames(GalleryGraph(path, ""), reversed, Keys(6), true);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);

    // A stranger is "unknown"; "events": "all" announces that too.
    FakeModelRegistry stranger = LogicRegistry();
    stranger.SetScriptByFirstPixel("emb", 0, Embedding(0.f, 0.f, 1.f, 0.f));
    const std::vector<FrameReport> unknown =
        RunLogicFrames(GalleryGraph(path, ",\"events\":\"all\""), stranger, Keys(2), false);
    GRAPH_CHECK(unknown.size() == 2 && unknown[0].events.size() == 1 && unknown[1].events.empty());
    if (unknown.size() == 2 && unknown[0].events.size() == 1) {
        GRAPH_CHECK(unknown[0].events[0].event == "unknown");
        GRAPH_CHECK(TextOf(RecordsOf(unknown[0], "g")->items[0], "name") == "unknown");
    }
    const std::vector<FrameReport> quiet =
        RunLogicFrames(GalleryGraph(path, ""), stranger, Keys(2), false);
    GRAPH_CHECK(FramesWithEvents(quiet).empty());
    std::remove(path.c_str());
}

void TestUserNodePassesItsInputThrough() {
    const std::string text = LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"my_rule\",\"input\":\"boxes\",\"output\":\"boxes\"}}" + std::string(kZoneNode),
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"},"
        "{\"from\":\"u\",\"to\":\"z\",\"carry\":\"result\"}");
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    for (std::size_t i = 0; i < 6; ++i) {
        const BoxesData* od = BoxesOf(reports[i], "od");
        const BoxesData* user = BoxesOf(reports[i], "u");
        GRAPH_CHECK(od != NULL && user != NULL && user->items.size() == od->items.size());
        if (od == NULL || user == NULL || user->items.size() != 1 || od->items.size() != 1) continue;
        GRAPH_CHECK(user->items[0].box == od->items[0].box);
        GRAPH_CHECK(user->items[0].track_id == od->items[0].track_id);
        GRAPH_CHECK(reports[i].overlays.count("u") == 0 && reports[i].logic_nodes.count("u") == 1);
    }
    GRAPH_CHECK(reports[2].events.size() == 1 && reports[2].events[0].node == "z");
    const std::vector<FrameReport> async = RunLogicFrames(text, ReversedLogicRegistry(), Keys(6), true);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
}

void TestRenderReportDrawsLogicOverlaysButNotTheirBoxes() {
    const cv::Mat black = cv::Mat::zeros(100, 200, CV_8UC3);
    const cv::Vec3b none(0, 0, 0);
    FrameReport report;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(120, 20, 40, 40, 0.9f, "person"));
    StageResult zone;
    zone.data = boxes;
    report.node_results["z"] = zone;
    LogicOverlay overlay;
    std::vector<cv::Point2f> square;
    square.push_back(cv::Point2f(10.f, 10.f));
    square.push_back(cv::Point2f(90.f, 10.f));
    square.push_back(cv::Point2f(90.f, 90.f));
    square.push_back(cv::Point2f(10.f, 90.f));
    overlay.polygons.push_back(square);
    report.overlays["z"] = overlay;
    const cv::Mat drawn = RenderReport(black, report);
    GRAPH_CHECK(drawn.at<cv::Vec3b>(10, 50) != none);   // the polygon's edge
    GRAPH_CHECK(drawn.at<cv::Vec3b>(50, 50) != none);   // its tinted inside
    GRAPH_CHECK(drawn.at<cv::Vec3b>(20, 140) == none);  // the zone's boxes are not redrawn
    report.overlays["z"].highlights.push_back(cv::Rect2f(120.f, 20.f, 40.f, 40.f));
    GRAPH_CHECK(RenderReport(black, report).at<cv::Vec3b>(20, 140) != none);

    // A line from A (bottom) to B (top): its "in" arrow points right.
    FrameReport crossing;
    LogicOverlay line;
    line.lines.push_back(std::make_pair(cv::Point2f(50.f, 90.f), cv::Point2f(50.f, 10.f)));
    crossing.overlays["l"] = line;
    const cv::Mat arrow = RenderReport(black, crossing);
    GRAPH_CHECK(arrow.at<cv::Vec3b>(50, 50) != none);
    GRAPH_CHECK(arrow.at<cv::Vec3b>(50, 65) != none);
    GRAPH_CHECK(arrow.at<cv::Vec3b>(50, 35) == none);

    // Panel labels (at.x < 0) stack from the top left.
    FrameReport counted;
    LogicOverlay panel;
    OverlayText label;
    label.text = "n cam: 3";
    panel.texts.push_back(label);
    counted.overlays["n"] = panel;
    const cv::Mat labelled = RenderReport(black, counted);
    GRAPH_CHECK(cv::countNonZero(labelled(cv::Rect(0, 0, 200, 40)).reshape(1)) > 0);
    GRAPH_CHECK(cv::countNonZero(labelled(cv::Rect(0, 60, 200, 40)).reshape(1)) == 0);

    // The recent-events panel: bottom left, nothing elsewhere.
    cv::Mat log = black.clone();
    DrawEventLog(&log, std::vector<std::string>(1, "0.2s z enter #0"));
    GRAPH_CHECK(cv::countNonZero(log(cv::Rect(0, 70, 200, 30)).reshape(1)) > 0);
    GRAPH_CHECK(cv::countNonZero(log(cv::Rect(0, 0, 200, 40)).reshape(1)) == 0);
}

void TestEventFileWriterWritesOneLinePerEvent() {
    const std::string path = ScratchPath("events.jsonl");
    {
        cli::EventFileWriter writer;
        std::string problem;
        GRAPH_CHECK(writer.Open(path, &problem));
        GRAPH_CHECK(writer.Append("{\"a\":1}", &problem));
        GRAPH_CHECK(writer.Append("{\"b\":2}", &problem));
    }
    std::ifstream in(path.c_str());
    std::string first;
    std::string second;
    std::string third;
    std::getline(in, first);
    std::getline(in, second);
    GRAPH_CHECK(first == "{\"a\":1}" && second == "{\"b\":2}" && !std::getline(in, third));
    std::remove(path.c_str());
    cli::EventFileWriter bad;
    std::string problem;
    GRAPH_CHECK(!bad.Open("/nonexistent_dir_for_graph_test/e.jsonl", &problem));
    GRAPH_CHECK(problem == "could not write /nonexistent_dir_for_graph_test/e.jsonl");
}

namespace {

std::vector<std::size_t> ReadIndices(consumer::StreamReader* reader) {
    std::vector<std::size_t> indices;
    std::size_t stream = 0;
    std::size_t index = 0;
    cv::Mat frame;
    while (reader->Next(&stream, &frame, &index)) indices.push_back(index);
    return indices;
}

std::vector<consumer::StreamInput> OneInput(const std::string& path) {
    std::vector<consumer::StreamInput> inputs(1);
    inputs[0].source = "cam";
    inputs[0].uri = path;
    inputs[0].input = consumer::OpenInput(path);
    return inputs;
}

}  // namespace

void TestStreamReaderLoopsAVideo() {
    const std::string video = ScratchPath("loop.avi");
    {
        cv::VideoWriter writer(video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 10,
                               cv::Size(64, 48));
        for (int i = 0; i < 4; ++i) writer.write(cv::Mat(48, 64, CV_8UC3, cv::Scalar(i * 40, 0, 0)));
    }
    consumer::StreamReader looped(OneInput(video), 10);
    looped.SetLoop(true);
    const std::vector<std::size_t> indices = ReadIndices(&looped);
    GRAPH_CHECK(indices.size() == 10);
    for (std::size_t i = 0; i < indices.size(); ++i) GRAPH_CHECK(indices[i] == i);
    GRAPH_CHECK(looped.restarts(0) == 2);  // 4 + 4 + 2 frames
    consumer::StreamReader once(OneInput(video), 10);
    GRAPH_CHECK(ReadIndices(&once).size() == 4 && once.restarts(0) == 0);
    // An image does not loop: one frame, then the stream ends.
    const std::string image = ScratchPath("loop.png");
    GRAPH_CHECK(cv::imwrite(image, cv::Mat(48, 64, CV_8UC3, cv::Scalar(9, 9, 9))));
    consumer::StreamReader still(OneInput(image), 0);
    still.SetLoop(true);
    GRAPH_CHECK(ReadIndices(&still).size() == 1 && still.restarts(0) == 0);
    std::remove(video.c_str());
    std::remove(image.c_str());
}

void TestConditionOnRecordsNeedsAValue() {
    // A line writes in, out and total, no "count": the default does not fit.
    std::string m = LogicError(LogicJson(
        std::string(kLineNode) + ConditionNode("{\"event\":\"busy\"}"),
        std::string(kOdToL) + ",{\"from\":\"l\",\"to\":\"c\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("a condition on records needs \"value\": one of in, out, total") !=
                std::string::npos);
    GRAPH_CHECK(m.find("\"value\" is \"count\"") == std::string::npos);
    GRAPH_CHECK(m.find("write \"value\": \"in\"") != std::string::npos);
    // Written but wrong: the old message, naming what it wrote.
    m = LogicError(LogicJson(
        std::string(kLineNode) + ConditionNode("{\"value\":\"inside\"}"),
        std::string(kOdToL) + ",{\"from\":\"l\",\"to\":\"c\",\"carry\":\"result\"}"));
    GRAPH_CHECK(m.find("\"value\" is \"inside\", a number \"l\" does not write") !=
                std::string::npos);
}

void TestABooleanOnALogicNodeGetsItsParamsHint() {
    std::string message;
    GRAPH_CHECK(ThrowsWithCode(
        LogicJson(",{\"id\":\"z\",\"type\":\"logic\",\"op\":\"zone\",\"params\":"
                  "{\"polygon\":[0,0,1,0,1,1],\"mode\":true}}", kOdToZ),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"mode\" must be a number, a string, or an array of strings or of "
                             "numbers, got true") != std::string::npos);
    GRAPH_CHECK(message.find("\"mode\" takes one of inside, outside") != std::string::npos);
    GRAPH_CHECK(message.find("switch") == std::string::npos);
    GRAPH_CHECK(ThrowsWithCode(
        LogicJson(",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\",\"params\":"
                  "{\"min_score\":false,\"classes\":true,\"strict\":true}}",
                  ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"}"),
        GraphErrorCode::kGraphSchema, &message));
    GRAPH_CHECK(message.find("\"min_score\" takes a number") != std::string::npos);
    GRAPH_CHECK(message.find("\"classes\" takes an array of strings") != std::string::npos);
    GRAPH_CHECK(message.find("filter accepts: classes, min_score, min_area") != std::string::npos);
    GRAPH_CHECK(message.find("switch") == std::string::npos);
}

void TestCountRejectsTotalAndRepeatedClasses() {
    const std::string edges = ",{\"from\":\"od\",\"to\":\"n\",\"carry\":\"result\"}";
    const auto count = [](const std::string& classes) {
        return ",{\"id\":\"n\",\"type\":\"logic\",\"op\":\"count\",\"params\":{\"classes\":" +
               classes + "}}";
    };
    GRAPH_CHECK(LogicError(LogicJson(count("[\"person\",\"car\"]"), edges)).empty());
    std::string m = LogicError(LogicJson(count("[\"person\",\"total\"]"), edges));
    GRAPH_CHECK(m.find("ERROR [GRAPH_SCHEMA] node \"n\" \"params\": \"classes\" holds \"total\", "
                       "the name of the count of every listed class") != std::string::npos);
    m = LogicError(LogicJson(count("[\"person\",\"car\",\"person\"]"), edges));
    GRAPH_CHECK(m.find("\"classes\" lists \"person\" twice") != std::string::npos);
    GRAPH_CHECK(m.find("list each class once") != std::string::npos);
    // A filter may keep a class named total: it writes no numbers.
    GRAPH_CHECK(LogicError(LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\",\"params\":{\"classes\":[\"total\"]}}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"}")).empty());
}

// Only a logic node that has something to draw gets an overlay entry; every
// logic node that ran is in logic_nodes, so the shape passes skip its payload.
void TestOnlyLogicNodesThatDrawHaveAnOverlay() {
    const std::string text = LogicJson(
        ",{\"id\":\"f\",\"type\":\"logic\",\"op\":\"filter\"}" + std::string(kZoneNode) +
            ",{\"id\":\"d\",\"type\":\"logic\",\"op\":\"dwell\",\"params\":{\"threshold_s\":1}}",
        ",{\"from\":\"od\",\"to\":\"f\",\"carry\":\"result\"},"
        "{\"from\":\"f\",\"to\":\"z\",\"carry\":\"result\"},"
        "{\"from\":\"z\",\"to\":\"d\",\"carry\":\"result\"}");
    const std::vector<FrameReport> reports = RunLogicFrames(text, LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    if (reports.size() != 6) return;
    for (std::size_t i = 0; i < 6; ++i) {
        GRAPH_CHECK(reports[i].overlays.count("f") == 0);
        GRAPH_CHECK(reports[i].overlays.count("z") == 1);
        // dwell draws a timer only for a track inside the zone (frames 2-4).
        GRAPH_CHECK(reports[i].overlays.count("d") == (i >= 2 && i <= 4 ? 1u : 0u));
        GRAPH_CHECK(reports[i].logic_nodes.size() == 3 && reports[i].logic_nodes.count("f") == 1);
        GRAPH_CHECK(reports[i].logic_nodes.count("od") == 0);
    }
    const std::vector<FrameReport> async = RunLogicFrames(text, ReversedLogicRegistry(), Keys(6), true);
    GRAPH_CHECK(async.size() == 6);
    for (std::size_t i = 0; i < 6 && i < async.size(); ++i) GRAPH_CHECK(reports[i] == async[i]);
    FrameReport other = reports[0];
    other.logic_nodes.erase("f");
    GRAPH_CHECK(!(other == reports[0]));
    // A logic node with nothing to draw still keeps its boxes out of the shape passes.
    const cv::Mat black = cv::Mat::zeros(100, 200, CV_8UC3);
    FrameReport filtered;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(120, 20, 40, 40, 0.9f, "person"));
    StageResult kept;
    kept.data = boxes;
    filtered.node_results["f"] = kept;
    GRAPH_CHECK(cv::countNonZero(RenderReport(black, filtered).reshape(1)) > 0);
    filtered.logic_nodes.insert("f");
    GRAPH_CHECK(cv::countNonZero(RenderReport(black, filtered).reshape(1)) == 0);
}

namespace {

/// A logic node's typed config, from its op and number params.
LogicConfig ConfigOf(const std::string& op, const std::map<std::string, double>& numbers) {
    NodeSpec node;
    node.id = op.substr(0, 1);
    node.is_logic = true;
    node.op = op;
    node.params.numeric = numbers;
    if (op == "zone") {
        const double half[] = {0.5, 0, 1, 0, 1, 1, 0.5, 1};
        node.arrays["polygon"].assign(half, half + 8);
    }
    if (op == "line") {
        const double middle[] = {0.5, 1, 0.5, 0};
        node.arrays["line"].assign(middle, middle + 4);
    }
    return MakeLogicConfig(node);
}

/// One full-frame boxes input: a person at x (none for x < 0) with `track`.
std::vector<LogicInput> PersonInput(float x, int track) {
    std::shared_ptr<BoxesData> boxes = LogicPersonAt(x);
    if (!boxes->items.empty()) boxes->items[0].track_id = track;
    LogicInput input;
    input.from = "od";
    input.data = boxes;
    return std::vector<LogicInput>(1, input);
}

/// Frame k of a 640x480 stream at `fps`: the time StageGraph::FrameTime gives.
LogicFrame FrameAt(std::size_t k, double fps) {
    LogicFrame at;
    at.frame_size = cv::Size(640, 480);
    at.time_s = static_cast<double>(k) / fps;
    return at;
}

}  // namespace

// Stream time is k / fps: a threshold that is a whole number of frames is
// reached on that frame, never one later because of rounding.
void TestTimeThresholdsHoldOnTheFrameTheyAreDue() {
    const double kFps = 30.0;
    const double cooldowns[] = {0.1, 1.0};
    for (std::size_t c = 0; c < 2; ++c) {
        std::map<std::string, double> numbers;
        numbers["cooldown_s"] = cooldowns[c];
        const LogicConfig condition = ConfigOf("condition", numbers);
        const std::size_t period = static_cast<std::size_t>(std::lround(cooldowns[c] * kFps));
        LogicState state;
        std::size_t late = 0;
        for (std::size_t k = 0; k < 3000; ++k) {
            const LogicOutput out = RunLogic(condition, PersonInput(400.f, -1), FrameAt(k, kFps),
                                             &state);
            if (!out.events.empty() != (k % period == 0)) ++late;
        }
        GRAPH_CHECK(late == 0);
    }
    const double thresholds[] = {0.1, 1.0};
    for (std::size_t t = 0; t < 2; ++t) {
        std::map<std::string, double> numbers;
        numbers["threshold_s"] = thresholds[t];
        const LogicConfig dwell = ConfigOf("dwell", numbers);
        const std::size_t due = static_cast<std::size_t>(std::lround(thresholds[t] * kFps));
        std::size_t wrong = 0;
        for (std::size_t start = 0; start < 300; ++start) {
            LogicState state;
            for (std::size_t k = start; k <= start + due; ++k) {
                const LogicOutput out = RunLogic(dwell, PersonInput(400.f, 1), FrameAt(k, kFps),
                                                 &state);
                if (!out.events.empty() != (k == start + due)) ++wrong;
            }
        }
        GRAPH_CHECK(wrong == 0);
    }
    // A gap of exactly grace_s keeps the stay.
    std::map<std::string, double> numbers;
    numbers["threshold_s"] = 100.0;
    numbers["grace_s"] = 0.1;
    const LogicConfig dwell = ConfigOf("dwell", numbers);
    std::size_t restarted = 0;
    for (std::size_t start = 0; start < 300; ++start) {
        LogicState state;
        RunLogic(dwell, PersonInput(400.f, 1), FrameAt(start, kFps), &state);
        const LogicOutput back = RunLogic(dwell, PersonInput(400.f, 1), FrameAt(start + 3, kFps),
                                          &state);
        const RecordsData* rows = dynamic_cast<const RecordsData*>(back.result.data.get());
        if (rows == NULL || rows->items.size() != 1 ||
            NumberOf(rows->items[0], "seconds") <= 0.0) {
            ++restarted;
        }
    }
    GRAPH_CHECK(restarted == 0);
}

// Decision 11: a track unseen for more than 5 s is forgotten. A zone says
// "exit" for one that was inside; a line and a dwell drop its memory.
void TestLogicForgetsATrackAfterFiveSeconds() {
    const double kFps = 30.0;
    const LogicConfig zone = ConfigOf("zone", std::map<std::string, double>());
    LogicState zone_state;
    std::vector<std::size_t> exits;
    for (std::size_t k = 0; k < 200; ++k) {
        const LogicOutput out = RunLogic(zone, PersonInput(k < 10 ? 400.f : -1.f, 3),
                                         FrameAt(k, kFps), &zone_state);
        for (std::size_t e = 0; e < out.events.size(); ++e) {
            if (out.events[e].event == "exit") exits.push_back(k);
        }
        if (k == 159) GRAPH_CHECK(zone_state.tracks.size() == 1);  // 150 frames: 5 s exactly
    }
    // Last seen at frame 9; frame 160 is the first more than 5 s later.
    GRAPH_CHECK(exits.size() == 1 && exits[0] == 160);
    GRAPH_CHECK(zone_state.tracks.empty());

    // A line: left of the middle at frame 0, then gone. Back on the right
    // within 5 s it crosses; after 5 s it is a new track and does not.
    const LogicConfig line = ConfigOf("line", std::map<std::string, double>());
    const std::size_t returns[] = {100, 200};
    for (std::size_t r = 0; r < 2; ++r) {
        LogicState state;
        std::size_t crossings = 0;
        for (std::size_t k = 0; k <= returns[r]; ++k) {
            const float x = k == 0 ? 200.f : (k == returns[r] ? 400.f : -1.f);
            crossings += RunLogic(line, PersonInput(x, 3), FrameAt(k, kFps), &state).events.size();
            if (k + 1 == returns[r]) GRAPH_CHECK(state.tracks.size() == (r == 0 ? 1u : 0u));
        }
        GRAPH_CHECK(crossings == (r == 0 ? 1u : 0u));
    }

    // A dwell with grace 1 s forgets after max(5, 1) s.
    std::map<std::string, double> numbers;
    numbers["grace_s"] = 1.0;
    const LogicConfig dwell = ConfigOf("dwell", numbers);
    LogicState dwell_state;
    for (std::size_t k = 0; k < 170; ++k) {
        RunLogic(dwell, PersonInput(k == 0 ? 400.f : -1.f, 3), FrameAt(k, kFps), &dwell_state);
        if (k == 150) GRAPH_CHECK(dwell_state.tracks.size() == 1);
    }
    GRAPH_CHECK(dwell_state.tracks.empty());
}

// Two streams through one zone, interleaved frame by frame: each keeps its
// own tracks, in both executors.
void TestTwoStreamsKeepTheirOwnLogicState() {
    const std::string text =
        "{\"version\":1,\"nodes\":[{\"id\":\"cam1\",\"type\":\"source\",\"uri\":\"a.jpg\"},"
        "{\"id\":\"cam2\",\"type\":\"source\",\"uri\":\"b.jpg\"},"
        "{\"id\":\"od\",\"model\":\"det\",\"track\":{\"algo\":\"iou\"}}" + std::string(kZoneNode) +
        "],\"edges\":[{\"from\":\"cam1\",\"to\":\"od\"},{\"from\":\"cam2\",\"to\":\"od\"}" +
        std::string(kOdToZ) + "]}";
    FakeModelRegistry registry = LogicRegistry();          // stream 0 walks keys 0..5
    registry.SetScriptByFirstPixel("det", 10, LogicPersonAt(500.f));  // stream 1 stays inside
    const GraphSpec spec = ParseGraphText(text, "two.json");
    ValidateGraph(spec, registry);
    std::vector<FrameReport> runs[2];
    for (int async = 0; async < 2; ++async) {
        FakeModelRegistry used = registry;
        if (async) used.SetDeliveryOrder("det", FakeModelRegistry::kReverse);
        StageGraph graph;
        graph.Build(spec, used, "/models", false);
        graph.SetStreamClock(0, 10.0, false);
        graph.SetStreamClock(1, 10.0, false);
        if (!async) {
            SyncExecutor executor;
            for (std::size_t i = 0; i < 6; ++i) {
                runs[0].push_back(executor.RunFrame(graph, 0, FrameWithKey(static_cast<int>(i)), i));
                runs[0].push_back(executor.RunFrame(graph, 1, FrameWithKey(10), i));
            }
        } else {
            AsyncExecutor executor(Frames(4));
            for (std::size_t i = 0; i < 6; ++i) {
                executor.Submit(graph, 0, FrameWithKey(static_cast<int>(i)), i);
                executor.Submit(graph, 1, FrameWithKey(10), i);
            }
            executor.Finish(graph);
            FrameReport report;
            while (executor.TryNext(&report)) runs[1].push_back(report);
        }
    }
    GRAPH_CHECK(runs[0].size() == 12 && runs[1].size() == 12);
    if (runs[0].size() != 12 || runs[1].size() != 12) return;
    std::vector<std::string> first;   // stream cam1: "frame event"
    std::vector<std::string> second;  // stream cam2
    for (std::size_t i = 0; i < 12; ++i) {
        GRAPH_CHECK(runs[0][i] == runs[1][i]);
        const FrameReport& report = runs[0][i];
        for (std::size_t e = 0; e < report.events.size(); ++e) {
            const std::string row = std::to_string(report.frame_index) + " " + report.events[e].event;
            (report.stream == "cam1" ? first : second).push_back(row);
        }
    }
    GRAPH_CHECK(first.size() == 2 && first[0] == "2 enter" && first[1] == "5 exit");
    GRAPH_CHECK(second.size() == 1 && second[0] == "0 enter");
}

// A condition on records: a line's "in", a count's "total" reduced over its
// rows, a dwell's longest "seconds".
void TestConditionReadsRecordsNumbers() {
    const std::vector<FrameReport> lined = RunLogicFrames(LogicJson(
        std::string(kLineNode) +
            ConditionNode("{\"value\":\"in\",\"comparator\":\">\",\"threshold\":0}"),
        std::string(kOdToL) + ",{\"from\":\"l\",\"to\":\"c\",\"carry\":\"result\"}"),
        LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(lined.size() == 6);
    if (lined.size() != 6) return;
    for (std::size_t i = 0; i < 6; ++i) {
        const RecordsData* records = RecordsOf(lined[i], "c");
        GRAPH_CHECK(records != NULL && records->items.size() == 1 &&
                    NumberOf(records->items[0], "value") == (i >= 2 ? 1.0 : 0.0));
    }
    std::vector<std::size_t> fired;
    for (std::size_t i = 0; i < 6; ++i) {
        for (std::size_t e = 0; e < lined[i].events.size(); ++e) {
            if (lined[i].events[e].node == "c") fired.push_back(i);
        }
    }
    GRAPH_CHECK(fired.size() == 1 && fired[0] == 2);

    // count over the two halves: one person in all, so "sum" is 1 and "min" 0.
    const std::string halves =
        ",{\"id\":\"left\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0,0,0.5,0,0.5,1,0,1]}}"
        ",{\"id\":\"right\",\"type\":\"logic\",\"op\":\"zone\",\"params\":{\"polygon\":"
        "[0.5,0,1,0,1,1,0.5,1]}}"
        ",{\"id\":\"n\",\"type\":\"logic\",\"op\":\"count\"}";
    const std::string half_edges =
        ",{\"from\":\"od\",\"to\":\"left\",\"carry\":\"result\"},"
        "{\"from\":\"od\",\"to\":\"right\",\"carry\":\"result\"},"
        "{\"from\":\"left\",\"to\":\"n\",\"carry\":\"result\"},"
        "{\"from\":\"right\",\"to\":\"n\",\"carry\":\"result\"},"
        "{\"from\":\"n\",\"to\":\"c\",\"carry\":\"result\"}";
    const char* reduces[] = {"sum", "min", "max"};
    const double values[] = {1.0, 0.0, 1.0};
    for (std::size_t r = 0; r < 3; ++r) {
        const std::vector<FrameReport> counted = RunLogicFrames(LogicJson(
            halves + ConditionNode(std::string("{\"value\":\"total\",\"reduce\":\"") + reduces[r] +
                                   "\",\"comparator\":\"==\",\"threshold\":1}"),
            half_edges), LogicRegistry(), Keys(6), false);
        GRAPH_CHECK(counted.size() == 6);
        for (std::size_t i = 0; i < counted.size(); ++i) {
            const RecordsData* records = RecordsOf(counted[i], "c");
            GRAPH_CHECK(records != NULL && records->items.size() == 1 &&
                        NumberOf(records->items[0], "value") == values[r] &&
                        NumberOf(records->items[0], "active") == (values[r] == 1.0 ? 1.0 : 0.0));
        }
    }

    // dwell's longest stay in the zone reaches 0.15 s on frame 4.
    const std::vector<FrameReport> dwelled = RunLogicFrames(LogicJson(
        std::string(kZoneNode) +
            ",{\"id\":\"d\",\"type\":\"logic\",\"op\":\"dwell\",\"params\":{\"threshold_s\":10}}" +
            ConditionNode("{\"value\":\"seconds\",\"reduce\":\"max\",\"comparator\":\">=\","
                          "\"threshold\":0.15,\"event\":\"loiter\"}"),
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"d\",\"carry\":\"result\"},"
                              "{\"from\":\"d\",\"to\":\"c\",\"carry\":\"result\"}"),
        LogicRegistry(), Keys(6), false);
    std::vector<std::size_t> loiter;
    for (std::size_t i = 0; i < dwelled.size(); ++i) {
        for (std::size_t e = 0; e < dwelled[i].events.size(); ++e) {
            if (dwelled[i].events[e].event == "loiter") loiter.push_back(i);
        }
    }
    GRAPH_CHECK(loiter.size() == 1 && loiter[0] == 4);
}

namespace {

LogicConfig GalleryConfig(const std::string& path, double threshold) {
    NodeSpec node;
    node.id = "g";
    node.is_logic = true;
    node.op = "gallery";
    node.params.text["gallery"] = path;
    node.params.numeric["threshold"] = threshold;
    return MakeLogicConfig(node);
}

/// One per-crop input: a crop of track `track` with `vector`.
std::vector<LogicInput> CropOf(const std::shared_ptr<VectorData>& vector, int track) {
    StageResult crop;
    crop.data = vector;
    crop.origin.from_roi = true;
    crop.origin.track_id = track;
    crop.origin.src_box = cv::Rect2f(10.f, 10.f, 40.f, 60.f);
    LogicInput input;
    input.from = "e";
    input.per_crop = true;
    input.crops.push_back(crop);
    return std::vector<LogicInput>(1, input);
}

}  // namespace

void TestGalleryMatchesOnlyAVectorWithADirection() {
    const std::string path =
        WriteGallery("logic_gallery2.bin", "emb", AliceAndBob(), AliceAndBobLabels());
    // A zero vector points nowhere: "unknown", even when any score would do.
    const LogicConfig any = GalleryConfig(path, 0.0);
    LogicState state;
    const LogicOutput zero =
        RunLogic(any, CropOf(Embedding(0.f, 0.f, 0.f, 0.f), 1), FrameAt(0, 10.0), &state);
    const RecordsData* rows = dynamic_cast<const RecordsData*>(zero.result.data.get());
    GRAPH_CHECK(zero.error.empty() && zero.events.empty());
    GRAPH_CHECK(rows != NULL && rows->items.size() == 1 && TextOf(rows->items[0], "name") == "unknown");
    // A track that turns from alice into bob is announced again, as bob.
    const LogicConfig gallery = GalleryConfig(path, 0.5);
    LogicState named;
    std::vector<std::string> said;
    for (std::size_t k = 0; k < 4; ++k) {
        const std::shared_ptr<VectorData> who =
            k < 2 ? Embedding(1.f, 0.f, 0.f, 0.f) : Embedding(0.f, 1.f, 0.f, 0.f);
        const LogicOutput out = RunLogic(gallery, CropOf(who, 7), FrameAt(k, 10.0), &named);
        for (std::size_t e = 0; e < out.events.size(); ++e) {
            said.push_back(std::to_string(k) + " " + out.events[e].event + " " +
                           TextOf(out.events[e].data, "name"));
        }
    }
    GRAPH_CHECK(said.size() == 2 && said[0] == "0 match alice" && said[1] == "2 match bob");
    // A vector of another width is an error that names both widths.
    std::shared_ptr<VectorData> short_vector(new VectorData());
    short_vector->values.assign(3, 1.f);
    const LogicOutput wrong = RunLogic(gallery, CropOf(short_vector, 7), FrameAt(5, 10.0), &named);
    GRAPH_CHECK(wrong.error == "a vector of 3 values, but the gallery holds 4");
    std::remove(path.c_str());
}


namespace {

/// The pixels of `drawn` that differ from `plain`, as a mask.
cv::Mat LabelDiffMask(const cv::Mat& plain, const cv::Mat& drawn) {
    cv::Mat diff;
    cv::absdiff(plain, drawn, diff);
    cv::Mat mask;
    cv::cvtColor(diff, mask, cv::COLOR_BGR2GRAY);
    return mask > 0;
}

/// `overlay` without its labels.
LogicOverlay WithoutTexts(LogicOverlay overlay) {
    overlay.texts.clear();
    return overlay;
}

}  // namespace

// A zone's label sits inside the zone at its top-left corner and leaves a
// box label alone; a line's label sits beside the line, on the side away
// from its "in" arrow.
void TestLogicLabelsCoverNeitherBoxLabelsNorTheInArrow() {
    const cv::Mat grey(640, 640, CV_8UC3, cv::Scalar(90, 90, 90));
    NodeSpec node;
    node.id = "zone";
    node.is_logic = true;
    node.op = "zone";
    const double right[] = {0.45, 0.35, 1.0, 0.35, 1.0, 1.0, 0.45, 1.0};
    node.arrays["polygon"].assign(right, right + 8);
    const LogicConfig zone = MakeLogicConfig(node);
    // The person of sample_people.jpg in the zone: its label row is just
    // above the zone's top edge, where the zone's corner is.
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(316, 233, 130, 195, 0.87f, "person"));
    LogicInput input;
    input.from = "od";
    input.data = boxes;
    LogicFrame at;
    at.frame_size = cv::Size(640, 640);
    LogicState state;
    const LogicOutput out = RunLogic(zone, std::vector<LogicInput>(1, input), at, &state);
    GRAPH_CHECK(out.overlay.texts.size() == 1);
    if (out.overlay.texts.size() != 1) return;
    GRAPH_CHECK(out.overlay.texts[0].at == cv::Point2f(288.f, 224.f));
    FrameReport report;
    StageResult detected;
    detected.data = boxes;
    report.node_results["od"] = detected;
    report.overlays["zone"] = WithoutTexts(out.overlay);
    const cv::Mat plain = RenderReport(grey, report);
    report.overlays["zone"] = out.overlay;
    const cv::Mat changed = LabelDiffMask(plain, RenderReport(grey, report));
    int baseline = 0;
    const cv::Size box_text =
        cv::getTextSize("person: 87%", cv::FONT_HERSHEY_SIMPLEX, 0.5, 2, &baseline);
    const cv::Rect box_label(316, 223 - box_text.height - 5, box_text.width,
                             box_text.height + 5 + baseline + 1);
    GRAPH_CHECK(cv::countNonZero(changed) > 0);
    GRAPH_CHECK(cv::countNonZero(changed(box_label)) == 0);  // the box's label is intact
    const cv::Rect label = cv::boundingRect(changed);
    GRAPH_CHECK(label.x >= 288 && label.y >= 224);           // inside the zone, at its corner
    GRAPH_CHECK(label.x < 320 && label.y < 280);
    // A crop's scores listed at the zone's corner (pass 4) stay readable too.
    std::shared_ptr<ScoresData> scores(new ScoresData());
    ClassificationResult hat;
    hat.class_name = "Hat";
    hat.confidence = 0.9f;
    scores->items.push_back(hat);
    StageResult crop;
    crop.data = scores;
    crop.origin.from_roi = true;
    crop.origin.src_box = cv::Rect2f(290.f, 226.f, 130.f, 195.f);
    report.roi_results["attr"].push_back(crop);
    report.overlays["zone"] = WithoutTexts(out.overlay);
    const cv::Mat scored = RenderReport(grey, report);
    report.overlays["zone"] = out.overlay;
    const cv::Mat score_changed = LabelDiffMask(scored, RenderReport(grey, report));
    GRAPH_CHECK(cv::countNonZero(score_changed(cv::Rect(290, 226, 60, 22))) == 0);
    GRAPH_CHECK(cv::countNonZero(score_changed) > 0);

    // A line from A (bottom) to B (top): the arrow points right, the label
    // is left of the line, beside its midpoint.
    std::map<std::string, double> none;
    const LogicConfig line = ConfigOf("line", none);  // x = 0.5 of a 640x480 frame
    LogicState crossings;
    LogicFrame frame;
    frame.frame_size = cv::Size(640, 480);
    const LogicOutput lined = RunLogic(line, PersonInput(100.f, 1), frame, &crossings);
    GRAPH_CHECK(lined.overlay.texts.size() == 1);
    if (lined.overlay.texts.size() != 1) return;
    const cv::Mat black = cv::Mat::zeros(480, 640, CV_8UC3);
    FrameReport crossing;
    crossing.overlays["l"] = WithoutTexts(lined.overlay);
    const cv::Mat bare = RenderReport(black, crossing);
    crossing.overlays["l"] = lined.overlay;
    const cv::Mat line_changed = LabelDiffMask(bare, RenderReport(black, crossing));
    GRAPH_CHECK(cv::countNonZero(line_changed) > 0);
    const cv::Rect line_label = cv::boundingRect(line_changed);
    GRAPH_CHECK(line_label.x + line_label.width <= 318);      // left of the line
    GRAPH_CHECK(line_label.x + line_label.width >= 300);      // right beside it
    GRAPH_CHECK(line_label.y < 240 && line_label.y + line_label.height > 220);  // at its middle

    // The top-left panel moves down past a box label in its way.
    FrameReport paneled;
    std::shared_ptr<BoxesData> top_left(new BoxesData(Shape::kBoxes));
    top_left->items.push_back(MakeItem(5, 40, 200, 200, 0.9f, "person"));
    StageResult corner;
    corner.data = top_left;
    paneled.node_results["od"] = corner;
    LogicOverlay panel;
    OverlayText row;
    row.text = "alarm: 1 > 0";
    panel.texts.push_back(row);
    paneled.overlays["alarm"] = panel;
    FrameReport unpaneled = paneled;
    unpaneled.overlays.clear();
    const cv::Mat panel_changed =
        LabelDiffMask(RenderReport(black, unpaneled), RenderReport(black, paneled));
    const int person_label_bottom = 30 + baseline;  // drawDetection: baseline at y1 - 10
    GRAPH_CHECK(cv::countNonZero(panel_changed) > 0);
    GRAPH_CHECK(cv::countNonZero(panel_changed(cv::Rect(0, 0, 640, person_label_bottom + 1))) == 0);
}

// Values a condition reads from records are compared with the same 1e-6
// slack as stream times: a dwell of 0.15 s that is 0.15 minus a rounding
// error is ">= 0.15".
void TestConditionComparesRecordValuesWithSlack() {
    struct Case {
        double value;
        const char* comparator;
        double threshold;
        bool active;
    };
    const Case cases[] = {
        {0.15 - 1e-12, ">=", 0.15, true},  {0.15 - 1e-12, "<", 0.15, false},
        {0.1 * 3, "<=", 0.3, true},        {0.1 * 3, ">", 0.3, false},
        {0.1 * 3, "==", 0.3, true},        {0.1 * 3, "!=", 0.3, false},
        {0.15 - 1e-3, ">=", 0.15, false},  {0.15 + 1e-3, ">", 0.15, true},
        {0.15 - 1e-3, "<", 0.15, true},    {0.15 + 1e-3, "!=", 0.15, true},
    };
    for (std::size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        NodeSpec node;
        node.id = "c";
        node.is_logic = true;
        node.op = "condition";
        node.params.text["value"] = "seconds";
        node.params.text["comparator"] = cases[i].comparator;
        node.params.numeric["threshold"] = cases[i].threshold;
        const LogicConfig condition = MakeLogicConfig(node);
        std::shared_ptr<RecordsData> records(new RecordsData());
        RecordItem row;
        row.numbers.push_back(std::make_pair(std::string("seconds"), cases[i].value));
        records->items.push_back(row);
        LogicInput input;
        input.from = "d";
        input.data = records;
        LogicState state;
        const LogicOutput out =
            RunLogic(condition, std::vector<LogicInput>(1, input), FrameAt(0, 30.0), &state);
        const RecordsData* rows = dynamic_cast<const RecordsData*>(out.result.data.get());
        const bool active = rows != NULL && rows->items.size() == 1 &&
                            NumberOf(rows->items[0], "active") == 1.0;
        if (active != cases[i].active) {
            std::printf("  case %zu: %.17g %s %.17g\n", i, cases[i].value, cases[i].comparator,
                        cases[i].threshold);
        }
        GRAPH_CHECK(active == cases[i].active);
        GRAPH_CHECK(out.events.size() == (cases[i].active ? 1u : 0u));
    }
}


namespace {

/// One event as text without its time: what a fresh pass must repeat.
std::string EventKey(const GraphEvent& event) {
    std::string text = event.node + " " + event.event + " #" + std::to_string(event.track_id);
    for (std::size_t i = 0; i < event.data.numbers.size(); ++i) {
        char number[32];
        std::snprintf(number, sizeof(number), "%.6g", event.data.numbers[i].second);
        text += " " + event.data.numbers[i].first + "=" + number;
    }
    return text;
}

/// `text` over the walk twice (keys 0..5, then 0..5 again), the stream
/// restarted (--loop) before the second pass when `restart`.
std::vector<FrameReport> RunWalkTwice(const std::string& text, bool restart, bool async) {
    FakeModelRegistry registry = LogicRegistry();
    const GraphSpec spec = ParseGraphText(text, "loop.json");
    ValidateGraph(spec, registry);
    StageGraph graph;
    graph.Build(spec, registry, "/models", false);
    graph.SetStreamClock(0, 10.0, false);
    std::vector<FrameReport> reports;
    SyncExecutor sync;
    AsyncExecutor pipelined(Frames(4));
    for (std::size_t i = 0; i < 12; ++i) {
        if (i == 6 && restart) graph.RestartStream(0);
        const cv::Mat frame = FrameWithKey(static_cast<int>(i % 6));
        if (async) {
            pipelined.Submit(graph, frame, i);
        } else {
            reports.push_back(sync.RunFrame(graph, frame, i));
        }
    }
    if (async) {
        pipelined.Finish(graph);
        FrameReport report;
        while (pipelined.TryNext(&report)) reports.push_back(report);
    }
    return reports;
}

}  // namespace

// --loop: a stream that starts its video again starts its trackers and its
// logic nodes' state again too, so the second pass raises what the first did.
void TestARestartedStreamStartsItsTracksAndLogicAgain() {
    const std::string text = LogicJson(
        std::string(kZoneNode) +
            ",{\"id\":\"d\",\"type\":\"logic\",\"op\":\"dwell\",\"params\":{\"threshold_s\":0.2}}",
        std::string(kOdToZ) + ",{\"from\":\"z\",\"to\":\"d\",\"carry\":\"result\"}");
    for (int async = 0; async < 2; ++async) {
        const std::vector<FrameReport> looped = RunWalkTwice(text, true, async != 0);
        GRAPH_CHECK(looped.size() == 12);
        if (looped.size() != 12) continue;
        std::vector<std::string> first;
        std::vector<std::string> second;
        for (std::size_t i = 0; i < 12; ++i) {
            for (std::size_t e = 0; e < looped[i].events.size(); ++e) {
                (i < 6 ? first : second)
                    .push_back(std::to_string(i % 6) + " " + EventKey(looped[i].events[e]));
            }
        }
        GRAPH_CHECK(!first.empty() && first == second);
        for (std::size_t i = 0; i < 6; ++i) {
            const BoxesData* a = dynamic_cast<const BoxesData*>(
                looped[i].node_results.at("od").data.get());
            const BoxesData* b = dynamic_cast<const BoxesData*>(
                looped[i + 6].node_results.at("od").data.get());
            GRAPH_CHECK(a != NULL && b != NULL && a->items.size() == b->items.size());
            for (std::size_t k = 0; a != NULL && b != NULL && k < a->items.size(); ++k) {
                GRAPH_CHECK(a->items[k].track_id == b->items[k].track_id);
            }
        }
        // Without the restart the stay goes on: no second dwell event.
        const std::vector<FrameReport> straight = RunWalkTwice(text, false, async != 0);
        std::size_t dwells = 0;
        for (std::size_t i = 0; i < straight.size(); ++i) {
            for (std::size_t e = 0; e < straight[i].events.size(); ++e) {
                dwells += straight[i].events[e].event == "dwell" ? 1 : 0;
            }
        }
        GRAPH_CHECK(dwells == 1);
    }
}


// Seconds in event data and records ("seconds", "*_s") are written to
// 0.001 s like "time": 0.30000000000000004 is 0.3. Other numbers keep
// every digit.
void TestSecondsAreWrittenRoundedLikeTime() {
    FrameReport report;
    report.frame_index = 3;
    report.stream = "cam";
    report.time_s = 0.3;
    GraphEvent event;
    event.node = "d";
    event.event = "dwell";
    event.track_id = 1;
    event.time_s = 0.3;
    event.data.numbers.push_back(std::make_pair(std::string("seconds"), 0.1 * 3));
    event.data.numbers.push_back(std::make_pair(std::string("wait_s"), 1.00049));
    event.data.numbers.push_back(std::make_pair(std::string("score"), 0.123456789));
    report.events.push_back(event);
    std::shared_ptr<RecordsData> records(new RecordsData());
    RecordItem row;
    row.numbers.push_back(std::make_pair(std::string("track_id"), 1.0));
    row.numbers.push_back(std::make_pair(std::string("seconds"), 0.1 * 3));
    records->items.push_back(row);
    StageResult dwell;
    dwell.data = records;
    report.node_results["d"] = dwell;
    const std::string line = consumer::EventLineJson(event, report).dump();
    GRAPH_CHECK(line.find("\"data\":{\"seconds\":0.3,\"wait_s\":1,\"score\":0.123456789}") !=
                std::string::npos);
    const nlohmann::json json = consumer::ReportToJson(report, false);
    GRAPH_CHECK(json["events"][0]["data"]["seconds"].dump() == "0.3");
    GRAPH_CHECK(json["nodes"]["d"]["payload"]["items"][0]["numbers"]["seconds"].dump() == "0.3");
    // dwell measures to the nanosecond: 34/30 - 1/30 is 1.1, not 1.0999999999999999,
    // in its records and its event (a condition reading it sees 1.1 too).
    std::map<std::string, double> numbers;
    numbers["threshold_s"] = 1.1;
    numbers["grace_s"] = 2.0;  // frame 1 to 34 is one stay
    const LogicConfig config = ConfigOf("dwell", numbers);
    LogicState state;
    RunLogic(config, PersonInput(400.f, 1), FrameAt(1, 30.0), &state);
    const LogicOutput later = RunLogic(config, PersonInput(400.f, 1), FrameAt(34, 30.0), &state);
    const RecordsData* rows = dynamic_cast<const RecordsData*>(later.result.data.get());
    GRAPH_CHECK(rows != NULL && rows->items.size() == 1 &&
                NumberOf(rows->items[0], "seconds") == 1.1);
    GRAPH_CHECK(later.events.size() == 1 && NumberOf(later.events[0].data, "seconds") == 1.1);
}

namespace {

/// A per-crop ReID result as gallery_known_person.json's "reid" reports
/// it: a vector, and its model's own top matches as a scores port.
StageResult ReidCrop(const cv::Rect2f& crop) {
    StageResult result;
    std::shared_ptr<VectorData> vector(new VectorData());
    vector->values.assign(512, 0.1f);
    result.data = vector;
    std::shared_ptr<ScoresData> matches(new ScoresData());
    for (int i = 0; i < 5; ++i) {
        ClassificationResult match;
        match.class_name = "sample/reid/gallery/p" + std::to_string(i) + ".jpg";
        match.confidence = 0.9f - 0.1f * static_cast<float>(i);
        matches->items.push_back(match);
    }
    result.ports["matches"] = matches;
    result.origin.from_roi = true;
    result.origin.src_box = crop;
    return result;
}

/// cam -> od -> {reid, attr}, and a gallery node "known" reading `gallery_reads`.
GraphSpec KnownPersonSpec(const std::string& gallery_reads) {
    GraphSpec spec;
    const char* ids[] = {"cam", "od", "reid", "attr", "known"};
    for (std::size_t i = 0; i < 5; ++i) {
        NodeSpec node;
        node.id = ids[i];
        spec.nodes.push_back(node);
    }
    spec.nodes[0].is_source = true;
    spec.nodes[4].is_logic = true;
    spec.nodes[4].op = "gallery";
    const char* edges[][2] = {{"cam", "od"}, {"od", "reid"}, {"od", "attr"}};
    for (std::size_t i = 0; i < 3; ++i) {
        EdgeSpec edge;
        edge.from = edges[i][0];
        edge.to = edges[i][1];
        spec.edges.push_back(edge);
    }
    EdgeSpec carried;
    carried.from = gallery_reads;
    carried.to = "known";
    carried.carry_result = true;
    spec.edges.push_back(carried);
    return spec;
}

}  // namespace

// scrfd -> 3ddfa-v2 wrote "vec[3]" over each face's "face: 87%": 3DDFA's "pose" port (yaw, pitch, roll) went
// through pass 4 like an embedding. A port rides on a result that is drawn already (the 68 landmarks), and
// "vec[N]" says only its length, so a vector port draws nothing and takes no label area. A node whose own
// result is a vector (a ReID embedding) still says "vec[N]".
void TestAVectorPortDrawsNoVecLabel() {
    const cv::Mat black = cv::Mat::zeros(400, 400, CV_8UC3);
    std::shared_ptr<KeypointsData> marks(new KeypointsData());
    PoseResult face;
    for (int i = 0; i < 68; ++i) {
        face.keypoints.push_back(Keypoint(20.f + static_cast<float>(i % 10) * 8.f,
                                          30.f + static_cast<float>(i / 10) * 12.f, 1.f));
    }
    marks->items.push_back(face);
    StageResult lmk;
    lmk.data = marks;
    lmk.origin.from_roi = true;
    lmk.origin.src_box = cv::Rect2f(100.f, 100.f, 120.f, 150.f);
    lmk.origin.inv_align = cv::Matx23f(1.f, 0.f, 100.f, 0.f, 1.f, 100.f);
    FrameReport plain;
    plain.roi_results["lmk"].push_back(lmk);
    FrameReport ported = plain;
    std::shared_ptr<VectorData> pose(new VectorData());
    pose->values.push_back(10.f);
    pose->values.push_back(-5.f);
    pose->values.push_back(2.f);
    ported.roi_results["lmk"][0].ports["pose"] = pose;
    std::vector<cv::Rect> plain_labels;
    std::vector<cv::Rect> ported_labels;
    const cv::Mat a = RenderReport(black, plain, RenderOptions(), &plain_labels);
    const cv::Mat b = RenderReport(black, ported, RenderOptions(), &ported_labels);
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    GRAPH_CHECK(cv::countNonZero(diff.reshape(1)) == 0);
    GRAPH_CHECK(plain_labels == ported_labels);

    FrameReport reid;
    reid.roi_results["reid"].push_back(ReidCrop(cv::Rect2f(40.f, 60.f, 120.f, 300.f)));
    reid.roi_results["reid"][0].ports.clear();
    const cv::Mat drawn = RenderReport(black, reid);
    GRAPH_CHECK(cv::countNonZero(drawn(cv::Rect(40, 60, 80, 24)).reshape(1)) > 0);  // "vec[512]"
}

// gallery_known_person: the gallery node names each person under its box,
// so the ReID node it reads draws no "vec[512]" and no top-5 list over the
// person; a node no gallery reads keeps its text.
void TestAGalleryNodeQuietsTheTextOfTheNodeItReads() {
    const RenderOptions quiet = RenderOptionsFor(KnownPersonSpec("reid"));
    GRAPH_CHECK(quiet.quiet_text == std::set<std::string>{"reid"});
    GRAPH_CHECK(RenderOptionsFor(KnownPersonSpec("attr")).quiet_text ==
                std::set<std::string>{"attr"});
    GraphSpec no_gallery = KnownPersonSpec("reid");
    no_gallery.nodes[4].op = "user";
    GRAPH_CHECK(RenderOptionsFor(no_gallery).quiet_text.empty());

    const cv::Mat black = cv::Mat::zeros(400, 400, CV_8UC3);
    FrameReport report;
    report.roi_results["reid"].push_back(ReidCrop(cv::Rect2f(40.f, 60.f, 120.f, 300.f)));
    std::shared_ptr<ScoresData> hat(new ScoresData());
    ClassificationResult row;
    row.class_name = "Hat";
    row.confidence = 0.9f;
    hat->items.push_back(row);
    StageResult attr;
    attr.data = hat;
    attr.origin.from_roi = true;
    attr.origin.src_box = cv::Rect2f(240.f, 60.f, 120.f, 300.f);
    report.roi_results["attr"].push_back(attr);
    LogicOverlay named;
    OverlayText name;
    name.at = cv::Point2f(40.f, 360.f);
    name.text = "sample_people_p0 0.98";
    named.texts.push_back(name);
    report.overlays["known"] = named;
    report.logic_nodes.insert("known");
    report.node_results["known"].data.reset(new RecordsData());

    const cv::Rect reid_text(40, 60, 200, 16 * 6 + 8);  // the vector line and 5 matches
    const cv::Rect attr_text(240, 60, 160, 24);
    const cv::Rect name_text(40, 330, 200, 40);
    const cv::Mat loud = RenderReport(black, report);
    GRAPH_CHECK(cv::countNonZero(loud(reid_text).reshape(1)) > 0);
    const cv::Mat drawn = RenderReport(black, report, quiet);
    GRAPH_CHECK(cv::countNonZero(drawn(reid_text).reshape(1)) == 0);
    GRAPH_CHECK(cv::countNonZero(drawn(attr_text).reshape(1)) > 0);  // read by no gallery
    GRAPH_CHECK(cv::countNonZero(drawn(name_text).reshape(1)) > 0);  // the gallery's name
    // The label areas a caller gets back leave the quiet lines out too.
    std::vector<cv::Rect> labels;
    RenderReport(black, report, quiet, &labels);
    bool reid_listed = false;
    bool name_listed = false;
    for (std::size_t i = 0; i < labels.size(); ++i) {
        reid_listed = reid_listed || (labels[i] & reid_text).area() > 0;
        name_listed = name_listed || (labels[i] & name_text).area() > 0;
    }
    GRAPH_CHECK(!reid_listed);
    GRAPH_CHECK(name_listed);
}

// v_loitering_dwell: a dwell timer under a box at the bottom edge is not
// covered by the recent-events panel, which moves to a free corner.
void TestEventLogKeepsOffTheLabelsRenderReportDrew() {
    const cv::Mat black = cv::Mat::zeros(360, 640, CV_8UC3);
    const std::vector<std::string> lines = {"0.5s zone enter #0", "5.5s dwell dwell #0"};
    FrameReport report;
    LogicOverlay dwell;
    OverlayText timer;
    timer.at = cv::Point2f(60.f, 358.f);  // the box's bottom left, on the frame's edge
    timer.text = "#0 5.8s";
    timer.alert = true;
    dwell.texts.push_back(timer);
    report.overlays["dwell"] = dwell;
    std::vector<cv::Rect> labels;
    const cv::Mat rendered = RenderReport(black, report, RenderOptions(), &labels);
    GRAPH_CHECK(labels.size() == 1);
    if (labels.size() != 1) return;
    const cv::Rect timer_box = labels[0];
    GRAPH_CHECK(cv::countNonZero(rendered(timer_box).reshape(1)) > 0);
    GRAPH_CHECK(timer_box.x < 200 && timer_box.y + timer_box.height > 330);

    // Without labels to avoid: bottom left, as before.
    cv::Mat plain = rendered.clone();
    DrawEventLog(&plain, lines);
    const cv::Rect bottom_left = cv::boundingRect(LabelDiffMask(rendered, plain));
    GRAPH_CHECK(bottom_left.x < 20 && bottom_left.y > 300);
    GRAPH_CHECK((bottom_left & timer_box).area() > 0);  // the defect: over the timer

    // Told where the labels are: off the timer, to the bottom right.
    cv::Mat moved = rendered.clone();
    DrawEventLog(&moved, lines, labels);
    const cv::Mat changed = LabelDiffMask(rendered, moved);
    GRAPH_CHECK(cv::countNonZero(changed) > 0);
    GRAPH_CHECK(cv::countNonZero(changed(timer_box)) == 0);
    const cv::Rect panel = cv::boundingRect(changed);
    GRAPH_CHECK(panel.x > 320 && panel.y > 300 && panel.x + panel.width > 600);

    // Bottom right taken too: the top right.
    std::vector<cv::Rect> both = labels;
    both.push_back(cv::Rect(560, 330, 80, 30));
    cv::Mat top = rendered.clone();
    DrawEventLog(&top, lines, both);
    const cv::Mat top_changed = LabelDiffMask(rendered, top);
    const cv::Rect top_panel = cv::boundingRect(top_changed);
    GRAPH_CHECK(top_panel.x > 320 && top_panel.y + top_panel.height < 60);
    for (std::size_t i = 0; i < both.size(); ++i) {
        GRAPH_CHECK(cv::countNonZero(top_changed(both[i] & cv::Rect(0, 0, 640, 360))) == 0);
    }
}

// ---------------------------------------------------------------------
// Results for front ends (spec 16): --draw, --output-layers, the grid.
// ---------------------------------------------------------------------

namespace {

/// A frame that is not one colour, so a blend shows against it.
cv::Mat GradientFrame(int rows, int cols) {
    cv::Mat frame(rows, cols, CV_8UC3);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            frame.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>((x * 255) / cols),
                                                  static_cast<uchar>((y * 255) / rows),
                                                  static_cast<uchar>(((x + y) * 7) % 256));
        }
    }
    return frame;
}

/// seg (a full-frame labelmap), od (two boxes), cls (scores on od's first
/// crop) and zone (a logic node: a polygon and a label) - every pass.
FrameReport FourNodeReport() {
    FrameReport report;
    std::shared_ptr<LabelMapData> seg(new LabelMapData());
    seg->labels = cv::Mat(30, 40, CV_8UC1, cv::Scalar(0));
    seg->labels(cv::Rect(0, 15, 40, 15)).setTo(cv::Scalar(2));
    report.node_results["seg"].data = seg;
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(200, 60, 80, 120, 0.9f, "person"));
    boxes->items.push_back(MakeItem(30, 140, 60, 60, 0.8f, "car"));
    report.node_results["od"].data = boxes;
    std::shared_ptr<ScoresData> scores(new ScoresData());
    ClassificationResult hat;
    hat.class_name = "Hat";
    hat.confidence = 0.7f;
    scores->items.push_back(hat);
    StageResult cls;
    cls.data = scores;
    cls.origin.from_roi = true;
    cls.origin.src_box = cv::Rect2f(200.f, 60.f, 80.f, 120.f);
    report.roi_results["cls"].push_back(cls);
    LogicOverlay zone;
    std::vector<cv::Point2f> square;
    square.push_back(cv::Point2f(20.f, 20.f));
    square.push_back(cv::Point2f(150.f, 20.f));
    square.push_back(cv::Point2f(150.f, 120.f));
    square.push_back(cv::Point2f(20.f, 120.f));
    zone.polygons.push_back(square);
    OverlayText count;
    count.text = "zone: 1";
    zone.texts.push_back(count);
    report.overlays["zone"] = zone;
    report.logic_nodes.insert("zone");
    report.node_results["zone"].data = boxes;  // a zone's payload is its detector's boxes
    return report;
}

/// `report` with only the nodes in `keep`.
FrameReport KeepNodes(FrameReport report, const std::set<std::string>& keep) {
    for (std::map<std::string, StageResult>::iterator it = report.node_results.begin();
         it != report.node_results.end();) {
        if (keep.count(it->first) == 0) it = report.node_results.erase(it); else ++it;
    }
    for (std::map<std::string, std::vector<StageResult> >::iterator it =
             report.roi_results.begin(); it != report.roi_results.end();) {
        if (keep.count(it->first) == 0) it = report.roi_results.erase(it); else ++it;
    }
    for (std::map<std::string, LogicOverlay>::iterator it = report.overlays.begin();
         it != report.overlays.end();) {
        if (keep.count(it->first) == 0) it = report.overlays.erase(it); else ++it;
    }
    return report;
}

int MaxDifference(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    double most = 0.0;
    cv::minMaxLoc(diff.reshape(1), NULL, &most);
    return static_cast<int>(most);
}

std::vector<std::string> LayerNodes(const std::vector<RenderLayer>& layers) {
    std::vector<std::string> nodes;
    for (std::size_t i = 0; i < layers.size(); ++i) nodes.push_back(layers[i].node);
    return nodes;
}

}  // namespace

// --draw: the named nodes alone, as if the others had not reported; a
// logic node's id draws its overlay; the label areas are the drawn ones'.
void TestDrawFilterDrawsOnlyTheNamedNodes() {
    GRAPH_CHECK(RenderOptions().Draws("anything"));
    RenderOptions od_only;
    od_only.draw.insert("od");
    GRAPH_CHECK(od_only.Draws("od") && !od_only.Draws("seg"));

    const cv::Mat source = GradientFrame(240, 320);
    const FrameReport report = FourNodeReport();
    const cv::Mat everything = RenderReport(source, report);
    GRAPH_CHECK(MaxDifference(everything, source) > 0);
    std::vector<cv::Rect> labels;
    const cv::Mat od = RenderReport(source, report, od_only, &labels);
    std::vector<cv::Rect> od_labels;
    const cv::Mat alone = RenderReport(source, KeepNodes(report, {"od"}), RenderOptions(),
                                       &od_labels);
    GRAPH_CHECK(MaxDifference(od, alone) == 0);
    GRAPH_CHECK(labels == od_labels);
    GRAPH_CHECK(MaxDifference(od(cv::Rect(0, 230, 320, 10)), source(cv::Rect(0, 230, 320, 10))) ==
                0);  // seg's lower half is not blended

    RenderOptions zone_and_cls;
    zone_and_cls.draw = {"zone", "cls"};
    const cv::Mat logic = RenderReport(source, report, zone_and_cls);
    GRAPH_CHECK(MaxDifference(logic,
                              RenderReport(source, KeepNodes(report, {"zone", "cls"}))) == 0);
    GRAPH_CHECK(MaxDifference(logic(cv::Rect(40, 40, 60, 60)), source(cv::Rect(40, 40, 60, 60))) >
                0);  // the zone's tint
    // A node that is not in the report draws nothing and fails nothing.
    RenderOptions absent;
    absent.draw.insert("nobody");
    GRAPH_CHECK(MaxDifference(RenderReport(source, report, absent), source) == 0);
}

// Two renderings, on black and on white, give a layer's colour and alpha
// exactly for every draw that blends a fixed colour at a fixed alpha.
void TestLayerFromBackdropsRecoversColourAndAlpha() {
    cv::Mat black = cv::Mat::zeros(1, 3, CV_8UC3);
    cv::Mat white(1, 3, CV_8UC3, cv::Scalar(255, 255, 255));
    // pixel 0: opaque (10, 20, 30); pixel 1: half of (200, 100, 0); pixel 2: nothing.
    black.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    white.at<cv::Vec3b>(0, 0) = cv::Vec3b(10, 20, 30);
    black.at<cv::Vec3b>(0, 1) = cv::Vec3b(100, 50, 0);
    white.at<cv::Vec3b>(0, 1) = cv::Vec3b(228, 178, 128);
    const cv::Mat layer = LayerFromBackdrops(black, white);
    GRAPH_CHECK(layer.type() == CV_8UC4);
    GRAPH_CHECK(layer.at<cv::Vec4b>(0, 0) == cv::Vec4b(10, 20, 30, 255));
    const cv::Vec4b half = layer.at<cv::Vec4b>(0, 1);
    GRAPH_CHECK(std::abs(half[3] - 128) <= 1 && std::abs(half[0] - 200) <= 2 &&
                std::abs(half[1] - 100) <= 2 && half[2] <= 1);
    GRAPH_CHECK(layer.at<cv::Vec4b>(0, 2)[3] == 0);
    // CompositeLayers puts it back over any base.
    const cv::Mat grey(1, 3, CV_8UC3, cv::Scalar(60, 60, 60));
    const cv::Mat stacked = CompositeLayers(grey, std::vector<cv::Mat>(1, layer));
    GRAPH_CHECK(stacked.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30));
    GRAPH_CHECK(std::abs(stacked.at<cv::Vec3b>(0, 1)[0] - 130) <= 2);
    GRAPH_CHECK(stacked.at<cv::Vec3b>(0, 2) == cv::Vec3b(60, 60, 60));
}

// --output-layers: one RGBA layer per node that drew, in RenderReport's
// pass order, so base + every layer stacked is the merged image.
void TestLayersStackBackToTheMergedImage() {
    const cv::Mat source = GradientFrame(240, 320);
    const FrameReport report = FourNodeReport();
    const std::vector<RenderLayer> layers = RenderLayers(source.size(), report);
    // seg in pass 1, od in pass 3, cls in pass 4, the zone in pass 5.
    GRAPH_CHECK(LayerNodes(layers) == std::vector<std::string>({"seg", "od", "cls", "zone"}));
    std::vector<cv::Mat> rgba;
    for (std::size_t i = 0; i < layers.size(); ++i) {
        GRAPH_CHECK(layers[i].rgba.type() == CV_8UC4 && layers[i].rgba.size() == source.size());
        rgba.push_back(layers[i].rgba);
    }
    const cv::Mat merged = RenderReport(source, report);
    const cv::Mat stacked = CompositeLayers(source, rgba);
    GRAPH_CHECK(MaxDifference(stacked, merged) <= 3);
    // A layer is transparent where its node drew nothing: od's far from its boxes.
    std::vector<cv::Mat> channels;
    cv::split(layers[1].rgba, channels);
    GRAPH_CHECK(cv::countNonZero(channels[3](cv::Rect(120, 200, 60, 40))) == 0);
    GRAPH_CHECK(cv::countNonZero(channels[3]) > 0);
    // seg's layer is half transparent where it blends.
    cv::split(layers[0].rgba, channels);
    GRAPH_CHECK(std::abs(channels[3].at<uchar>(200, 300) - 128) <= 1);

    // With --draw, only the drawn nodes have layers, stacking to that image.
    RenderOptions some;
    some.draw = {"od", "zone"};
    const std::vector<RenderLayer> drawn = RenderLayers(source.size(), report, some);
    GRAPH_CHECK(LayerNodes(drawn) == std::vector<std::string>({"od", "zone"}));
    std::vector<cv::Mat> drawn_rgba;
    for (std::size_t i = 0; i < drawn.size(); ++i) drawn_rgba.push_back(drawn[i].rgba);
    GRAPH_CHECK(MaxDifference(CompositeLayers(source, drawn_rgba),
                              RenderReport(source, report, some)) <= 3);
    // A node that drew no pixel has no layer; an empty report has none.
    GRAPH_CHECK(RenderLayers(source.size(), FrameReport()).empty());
}

// --display with several sources: one window, a grid of labelled cells.
void TestDisplayGridTilesTheStreams() {
    GRAPH_CHECK(cli::GridShape(1) == cv::Size(1, 1));
    GRAPH_CHECK(cli::GridShape(2) == cv::Size(2, 1));
    GRAPH_CHECK(cli::GridShape(3) == cv::Size(2, 2));
    GRAPH_CHECK(cli::GridShape(4) == cv::Size(2, 2));
    GRAPH_CHECK(cli::GridShape(5) == cv::Size(3, 2));
    GRAPH_CHECK(cli::GridShape(9) == cv::Size(3, 3));
    GRAPH_CHECK(cli::GridShape(10) == cv::Size(4, 3));

    cv::Size canvas;
    std::vector<cv::Rect> cells = cli::GridCells(2, cv::Size(640, 360), cv::Size(1920, 1080),
                                                 &canvas);
    GRAPH_CHECK(canvas == cv::Size(1280, 360));
    GRAPH_CHECK(cells.size() == 2 && cells[0] == cv::Rect(0, 0, 640, 360) &&
                cells[1] == cv::Rect(640, 0, 640, 360));
    cells = cli::GridCells(3, cv::Size(640, 360), cv::Size(1920, 1080), &canvas);
    GRAPH_CHECK(canvas == cv::Size(1280, 720));
    GRAPH_CHECK(cells.size() == 3 && cells[2] == cv::Rect(0, 360, 640, 360));
    // Too big for the bound: every cell shrinks alike, aspect kept.
    cells = cli::GridCells(2, cv::Size(1920, 1080), cv::Size(1920, 1080), &canvas);
    GRAPH_CHECK(canvas == cv::Size(1920, 540));
    GRAPH_CHECK(cells[1] == cv::Rect(960, 0, 960, 540));

    // The tiled image: each frame fills its cell (letterboxed when its
    // aspect differs), and its source id is written at the cell's top left.
    std::vector<cv::Mat> frames;
    frames.push_back(cv::Mat(360, 640, CV_8UC3, cv::Scalar(0, 0, 200)));
    frames.push_back(cv::Mat(720, 640, CV_8UC3, cv::Scalar(0, 200, 0)));  // taller
    frames.push_back(cv::Mat());                                          // no frame yet
    const std::vector<std::string> names = {"cam1", "cam2", "cam3"};
    const cv::Mat grid = cli::TileFrames(frames, names, cv::Size(640, 360));
    GRAPH_CHECK(grid.size() == cv::Size(1280, 720) && grid.type() == CV_8UC3);
    GRAPH_CHECK(grid.at<cv::Vec3b>(300, 600) == cv::Vec3b(0, 0, 200));
    GRAPH_CHECK(grid.at<cv::Vec3b>(300, 960) == cv::Vec3b(0, 200, 0));    // centre of cell 2
    GRAPH_CHECK(grid.at<cv::Vec3b>(300, 650) == cv::Vec3b(0, 0, 0));      // its letterbox
    GRAPH_CHECK(grid.at<cv::Vec3b>(600, 320) != cv::Vec3b(0, 0, 200));   // cell 3: empty
    // The labels: white text on the cell's top left.
    cv::Mat label_area = grid(cv::Rect(0, 0, 120, 30)), white;
    cv::inRange(label_area, cv::Scalar(250, 250, 250), cv::Scalar(255, 255, 255), white);
    GRAPH_CHECK(cv::countNonZero(white) > 0);
    cv::inRange(grid(cv::Rect(640, 0, 120, 30)), cv::Scalar(250, 250, 250),
                cv::Scalar(255, 255, 255), white);
    GRAPH_CHECK(cv::countNonZero(white) > 0);
    cv::inRange(grid(cv::Rect(0, 360, 120, 30)), cv::Scalar(250, 250, 250),
                cv::Scalar(255, 255, 255), white);
    GRAPH_CHECK(cv::countNonZero(white) > 0);
    // Nothing is labelled mid-cell.
    cv::inRange(grid(cv::Rect(200, 150, 200, 100)), cv::Scalar(250, 250, 250),
                cv::Scalar(255, 255, 255), white);
    GRAPH_CHECK(cv::countNonZero(white) == 0);
}

// --output-layers' file names: <dir>/<stem>[_<source>]_<frame6>_<part>.png.
void TestLayerFileNames() {
    GRAPH_CHECK(cli::LayerFileName("result", "cam", false, 3, "od") == "result_000003_od.png");
    GRAPH_CHECK(cli::LayerFileName("result", "cam2", true, 0, "base") ==
                "result_cam2_000000_base.png");
    GRAPH_CHECK(cli::LayersStem("out/run.v1/result.png", "g/fanout.json") == "result");
    GRAPH_CHECK(cli::LayersStem("out/result", "g/fanout.json") == "result");
    GRAPH_CHECK(cli::LayersStem("", "g/fanout_od_seg_depth.json") == "fanout_od_seg_depth");
}

// ---------------------------------------------------------------------
// Phase 3: the runtime root (common/utility/repo_path.hpp)
// ---------------------------------------------------------------------

namespace {

/// Sets $DXAPP_ROOT for one scope (NULL: unset) and puts the old value back.
class RootEnv {
 public:
    explicit RootEnv(const char* value) : had_(false) {
        const char* old = std::getenv("DXAPP_ROOT");
        if (old != NULL) {
            had_ = true;
            old_ = old;
        }
        if (value == NULL) {
            ::unsetenv("DXAPP_ROOT");
        } else {
            ::setenv("DXAPP_ROOT", value, 1);
        }
    }
    ~RootEnv() {
        if (had_) {
            ::setenv("DXAPP_ROOT", old_.c_str(), 1);
        } else {
            ::unsetenv("DXAPP_ROOT");
        }
    }

 private:
    bool had_;
    std::string old_;
};

}  // namespace

void TestRuntimeRootIsDxappRootElseTheBuildsRepository() {
    {
        RootEnv unset(NULL);
        GRAPH_CHECK(::dxapp::RuntimeRoot() == std::string(PROJECT_ROOT_DIR));
        GRAPH_CHECK(consumer::ProjectRoot() == std::string(PROJECT_ROOT_DIR));
    }
    {
        RootEnv empty("");
        GRAPH_CHECK(::dxapp::RuntimeRoot() == std::string(PROJECT_ROOT_DIR));
    }
    RootEnv moved("/opt/exported");
    GRAPH_CHECK(::dxapp::RuntimeRoot() == "/opt/exported");
    GRAPH_CHECK(::dxapp::ResolveRepoRelative("sample/img/a.jpg") == "/opt/exported/sample/img/a.jpg");
    GRAPH_CHECK(::dxapp::ResolveRepoRelative("/abs/a.jpg") == "/abs/a.jpg");
    GRAPH_CHECK(consumer::ProjectRoot() == "/opt/exported");
    GRAPH_CHECK(consumer::DefaultModelDir() == "/opt/exported/assets/models");
    ModelInfo info;
    info.model_name = "yolov8-n_640x640";
    info.variant = "yolov8-n_640x640";
    info.family = "yolov8";
    info.task = "object_detection";
    GRAPH_CHECK(detail::StageConfigPath(info) ==
                "/opt/exported/src/cpp_example/object_detection/yolov8/yolov8-n_640x640/config.json");
}

void TestFindMarkedRootWalksUpToTheMarker() {
    const std::string base = ScratchPath("marked_root");
    const std::string inner = base + "/project/build/bin";
    GRAPH_CHECK(std::system(("mkdir -p '" + inner + "'").c_str()) == 0);
    GRAPH_CHECK(::dxapp::FindMarkedRoot(inner).empty());
    {
        std::ofstream marker((base + "/project/" + ::dxapp::RootMarkerName()).c_str());
        marker << "dx_app graph project\n";
    }
    GRAPH_CHECK(::dxapp::FindMarkedRoot(inner) == base + "/project");
    GRAPH_CHECK(::dxapp::FindMarkedRoot(inner + "/") == base + "/project");
    GRAPH_CHECK(::dxapp::FindMarkedRoot(base + "/project") == base + "/project");
    GRAPH_CHECK(::dxapp::FindMarkedRoot(base).empty());
    GRAPH_CHECK(::dxapp::FindMarkedRoot("").empty());
    GRAPH_CHECK(!::dxapp::ExecutableDir().empty());
    GRAPH_CHECK(!::dxapp::WorkingDir().empty());
    GRAPH_CHECK(std::system(("rm -rf '" + base + "'").c_str()) == 0);
}

// ---------------------------------------------------------------------
// Phase 3: a user node's name becomes a class name in an exported project
// ---------------------------------------------------------------------

void TestUserNodeNamesAreCppNamesThatAreNotKeywords() {
    GRAPH_CHECK(IsUserLogicName("tag_people") && IsUserLogicName("_count2") && IsUserLogicName("Rule"));
    GRAPH_CHECK(!IsUserLogicName("class") && !IsUserLogicName("register") && !IsUserLogicName("and"));
    GRAPH_CHECK(!IsUserLogicName("2fast") && !IsUserLogicName("a-b") && !IsUserLogicName(""));
    GRAPH_CHECK(!IsUserLogicName("a__b") && !IsUserLogicName("_Upper"));
    GRAPH_CHECK(!IsUserLogicName(std::string(65, 'a')) && IsUserLogicName(std::string(64, 'a')));
    GRAPH_CHECK(IsEventName("seen") && IsEventName("zone.enter:2-b"));
    GRAPH_CHECK(!IsEventName("two words") && !IsEventName("") && !IsEventName(std::string(65, 'e')));
    const std::string keyword = LogicError(LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"class\",\"input\":\"boxes\",\"output\":\"boxes\"}}",
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"}"));
    GRAPH_CHECK(keyword.find("\"name\" must be a C identifier that is not a C++ keyword, got \"class\"") !=
                std::string::npos);
}

void TestUserNodesSharingANameTakeOneKind() {
    const std::string two_kinds = LogicJson(
        ",{\"id\":\"u1\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"rule\",\"input\":\"boxes\",\"output\":\"boxes\"}}"
        ",{\"id\":\"c\",\"type\":\"logic\",\"op\":\"count\"}"
        ",{\"id\":\"u2\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"rule\",\"input\":\"records\",\"output\":\"records\"}}",
        ",{\"from\":\"od\",\"to\":\"u1\",\"carry\":\"result\"}"
        ",{\"from\":\"od\",\"to\":\"c\",\"carry\":\"result\"}"
        ",{\"from\":\"c\",\"to\":\"u2\",\"carry\":\"result\"}");
    const std::string error = LogicError(two_kinds);
    GRAPH_CHECK(error.find("node \"u2\"") != std::string::npos);
    GRAPH_CHECK(error.find("\"name\" \"rule\" is also the name of node \"u1\", which takes boxes, "
                           "not records") != std::string::npos);
    const std::string one_kind = LogicJson(
        ",{\"id\":\"u1\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"rule\",\"input\":\"boxes\",\"output\":\"boxes\"}}"
        ",{\"id\":\"u2\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"rule\",\"input\":\"boxes\",\"output\":\"boxes\"}}",
        ",{\"from\":\"od\",\"to\":\"u1\",\"carry\":\"result\"}"
        ",{\"from\":\"od\",\"to\":\"u2\",\"carry\":\"result\"}");
    GRAPH_CHECK(LogicError(one_kind).empty());
}

// ---------------------------------------------------------------------
// Phase 3: the user-logic hook (common/graph/user_logic.hpp)
// ---------------------------------------------------------------------

namespace {

/// A user node's class for the tests: names every box "tagged", counts its
/// frames, raises one "seen" event and writes one panel line per frame.
struct TagBoxes {
    int frames;
    TagBoxes() : frames(0) {}
    BoxesData Run(const BoxesData& input, UserContext& ctx) {
        ++frames;
        BoxesData out = input;
        for (std::size_t i = 0; i < out.items.size(); ++i) out.items[i].class_name = "tagged";
        RecordItem data;
        data.numbers.push_back(std::make_pair(std::string("frames"), static_cast<double>(frames)));
        ctx.Raise("seen", out.items.empty() ? -1 : out.items[0].track_id, data);
        ctx.Label("frames " + std::to_string(frames));
        return out;
    }
};

/// Takes scores: given boxes, TypedUserLogic refuses the input.
struct WantsScores {
    ScoresData Run(const ScoresData& input, UserContext&) { return input; }
};

}  // namespace

void TestUserLogicRegistryMakesOneObjectPerCall() {
    ClearUserLogic();
    GRAPH_CHECK(!MakeUserLogic("tag_boxes"));
    RegisterUserLogic<BoxesData, TagBoxes>("tag_boxes");
    RegisterUserLogic<ScoresData, WantsScores>("wants_scores");
    const std::vector<std::string> names = RegisteredUserLogicNames();
    GRAPH_CHECK(names.size() == 2 && names[0] == "tag_boxes" && names[1] == "wants_scores");

    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10.f, 10.f, 20.f, 20.f, 0.9f, "person"));
    UserContext ctx("u", 1.5, cv::Size(640, 480));
    GRAPH_CHECK(ctx.node() == "u" && ctx.time_s() == 1.5 && ctx.frame_size() == cv::Size(640, 480));
    std::unique_ptr<UserLogic> first = MakeUserLogic("tag_boxes");
    std::unique_ptr<UserLogic> second = MakeUserLogic("tag_boxes");
    GRAPH_CHECK(first && second && first.get() != second.get());
    if (!first || !second) return;
    first->Run(boxes, ctx);
    const StageDataPtr out = first->Run(boxes, ctx);
    const BoxesData* tagged = dynamic_cast<const BoxesData*>(out.get());
    GRAPH_CHECK(tagged != NULL && tagged->items.size() == 1 && tagged->items[0].class_name == "tagged");
    GRAPH_CHECK(boxes->items[0].class_name == "person");  // the input is not changed
    GRAPH_CHECK(ctx.events().size() == 2 && ctx.events()[1].node == "u" &&
                ctx.events()[1].event == "seen" && ctx.events()[1].time_s == 1.5);
    GRAPH_CHECK(NumberOf(ctx.events()[1].data, "frames") == 2.0);  // one object keeps its count
    GRAPH_CHECK(ctx.labels().size() == 2 && ctx.labels()[1].text == "frames 2" &&
                ctx.labels()[1].at.x < 0.f);  // the top-left panel
    UserContext fresh("u", 0.0, cv::Size(1, 1));
    second->Run(boxes, fresh);
    GRAPH_CHECK(fresh.events().size() == 1 && NumberOf(fresh.events()[0].data, "frames") == 1.0);
    fresh.LabelAt("here", cv::Point2f(5.f, 6.f), true);
    GRAPH_CHECK(fresh.labels().back().at == cv::Point2f(5.f, 6.f) && fresh.labels().back().alert);

    bool refused = false;
    try {
        MakeUserLogic("wants_scores")->Run(boxes, ctx);
    } catch (const std::exception& e) {
        refused = std::string(e.what()).find("boxes") != std::string::npos;
    }
    GRAPH_CHECK(refused);
    ClearUserLogic();
    GRAPH_CHECK(RegisteredUserLogicNames().empty() && !MakeUserLogic("tag_boxes"));
}

namespace {

/// Throws on its second frame.
struct ThrowsOnFrameTwo {
    int frames;
    ThrowsOnFrameTwo() : frames(0) {}
    BoxesData Run(const BoxesData& input, UserContext&) {
        if (++frames == 2) throw std::runtime_error("bad frame");
        return input;
    }
};

/// Returns obboxes for boxes: another kind.
struct ReturnsObBoxes {
    BoxesData Run(const BoxesData& input, UserContext&) {
        BoxesData out(Shape::kObBoxes);
        out.items = input.items;
        return out;
    }
};

/// Raises an event whose name has a space.
struct RaisesABadName {
    BoxesData Run(const BoxesData& input, UserContext& ctx) {
        ctx.Raise("two words");
        return input;
    }
};

/// od -> u (user "tag_boxes", boxes) -> z (zone), over LogicRegistry's walk.
std::string UserGraph() {
    return LogicJson(
        ",{\"id\":\"u\",\"type\":\"logic\",\"op\":\"user\",\"params\":"
        "{\"name\":\"tag_boxes\",\"input\":\"boxes\",\"output\":\"boxes\"}}" +
            std::string(kZoneNode),
        ",{\"from\":\"od\",\"to\":\"u\",\"carry\":\"result\"},"
        "{\"from\":\"u\",\"to\":\"z\",\"carry\":\"result\"}");
}

std::vector<GraphEvent> EventsOf(const FrameReport& report, const std::string& node) {
    std::vector<GraphEvent> out;
    for (std::size_t i = 0; i < report.events.size(); ++i) {
        if (report.events[i].node == node) out.push_back(report.events[i]);
    }
    return out;
}

}  // namespace

void TestARegisteredUserClassRunsInPlaceOfThePassThrough() {
    ClearUserLogic();
    RegisterUserLogic<BoxesData, TagBoxes>("tag_boxes");
    const std::vector<FrameReport> reports = RunLogicFrames(UserGraph(), LogicRegistry(), Keys(6), false);
    GRAPH_CHECK(reports.size() == 6);
    for (std::size_t i = 0; i < reports.size(); ++i) {
        const BoxesData* user = BoxesOf(reports[i], "u");
        GRAPH_CHECK(user != NULL && user->items.size() == 1 && user->items[0].class_name == "tagged");
        const std::vector<GraphEvent> mine = EventsOf(reports[i], "u");
        GRAPH_CHECK(mine.size() == 1 && mine[0].event == "seen" &&
                    mine[0].time_s == reports[i].time_s);
        if (!mine.empty()) GRAPH_CHECK(NumberOf(mine[0].data, "frames") == static_cast<double>(i + 1));
        GRAPH_CHECK(reports[i].overlays.count("u") == 1 &&
                    reports[i].overlays.at("u").texts.size() == 1 &&
                    reports[i].overlays.at("u").texts[0].text == "frames " + std::to_string(i + 1));
        GRAPH_CHECK(reports[i].error.empty());
        if (!reports[i].events.empty()) GRAPH_CHECK(reports[i].events[0].node == "u");  // node order
    }
    if (reports.size() == 6) GRAPH_CHECK(EventsOf(reports[2], "z").size() == 1);  // the zone still runs
    const std::vector<FrameReport> async =
        RunLogicFrames(UserGraph(), ReversedLogicRegistry(), Keys(6), true);
    GRAPH_CHECK(async.size() == 6);
    for (std::size_t i = 0; i < 6 && i < async.size() && i < reports.size(); ++i) {
        GRAPH_CHECK(reports[i] == async[i]);
    }
    ClearUserLogic();
}

void TestEachLogicStateHasItsOwnUserObject() {
    ClearUserLogic();
    RegisterUserLogic<BoxesData, TagBoxes>("tag_boxes");
    const GraphSpec spec = ParseGraphText(UserGraph(), "u.json");
    const LogicConfig config = MakeLogicConfig(spec.nodes[2]);  // cam, od, u, z
    LogicInput input;
    input.edge = 1;
    input.from = "od";
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10.f, 10.f, 20.f, 20.f, 0.9f, "person"));
    input.data = boxes;
    LogicFrame frame;
    frame.frame_size = cv::Size(100, 100);
    LogicState stream_a;
    LogicState stream_b;
    RunLogic(config, std::vector<LogicInput>(1, input), frame, &stream_a);
    const LogicOutput again = RunLogic(config, std::vector<LogicInput>(1, input), frame, &stream_a);
    const LogicOutput other = RunLogic(config, std::vector<LogicInput>(1, input), frame, &stream_b);
    GRAPH_CHECK(again.events.size() == 1 && NumberOf(again.events[0].data, "frames") == 2.0);
    GRAPH_CHECK(other.events.size() == 1 && NumberOf(other.events[0].data, "frames") == 1.0);
    ClearUserLogic();
    LogicState unregistered;
    const LogicOutput through = RunLogic(config, std::vector<LogicInput>(1, input), frame, &unregistered);
    const BoxesData* same = dynamic_cast<const BoxesData*>(through.result.data.get());
    GRAPH_CHECK(through.events.empty() && same != NULL && same->items[0].class_name == "person");
}

void TestUserClassFailuresPassTheInputThroughAndSayWhy() {
    ClearUserLogic();
    RegisterUserLogic<BoxesData, ThrowsOnFrameTwo>("tag_boxes");
    std::vector<FrameReport> reports = RunLogicFrames(UserGraph(), LogicRegistry(), Keys(3), false);
    GRAPH_CHECK(reports.size() == 3);
    if (reports.size() == 3) {
        GRAPH_CHECK(reports[0].error.empty() && reports[2].error.empty());
        GRAPH_CHECK(reports[1].error.find("node \"u\": user logic \"tag_boxes\" failed: bad frame") !=
                    std::string::npos);
        const BoxesData* user = BoxesOf(reports[1], "u");
        GRAPH_CHECK(user != NULL && user->items.size() == 1 && user->items[0].class_name != "tagged");
    }
    ClearUserLogic();
    RegisterUserLogic<BoxesData, ReturnsObBoxes>("tag_boxes");
    reports = RunLogicFrames(UserGraph(), LogicRegistry(), Keys(1), false);
    GRAPH_CHECK(reports.size() == 1 &&
                reports[0].error.find("user logic \"tag_boxes\" returned obboxes; it must return "
                                      "boxes") != std::string::npos);
    if (reports.size() == 1) GRAPH_CHECK(BoxesOf(reports[0], "u")->shape() == Shape::kBoxes);
    ClearUserLogic();
    RegisterUserLogic<BoxesData, RaisesABadName>("tag_boxes");
    reports = RunLogicFrames(UserGraph(), LogicRegistry(), Keys(1), false);
    GRAPH_CHECK(reports.size() == 1 &&
                reports[0].error.find("user logic \"tag_boxes\" raised \"two words\", which is not "
                                      "an event name") != std::string::npos);
    if (reports.size() == 1) GRAPH_CHECK(EventsOf(reports[0], "u").empty());
    ClearUserLogic();
}

// ---------------------------------------------------------------------
// Phase 3: an exported project's command line (cli::ProjectArgs)
// ---------------------------------------------------------------------

namespace {

std::vector<std::string> Words(const char* const* words, std::size_t count) {
    return std::vector<std::string>(words, words + count);
}

}  // namespace

void TestProjectArgsFillInTheProjectsGraphAndModels() {
    cli::ProjectDefaults defaults;
    defaults.graph = "graph.json";
    defaults.model_dir = "models";

    const char* bare[] = {"app"};
    cli::ProjectCall call = cli::ProjectArgs(Words(bare, 1), defaults, "/p");
    const char* bare_out[] = {"app", "--model-dir", "/p/models", "--graph", "/p/graph.json"};
    GRAPH_CHECK(call.kind == cli::kAsyncExecutor && call.args == Words(bare_out, 5));

    const char* sync[] = {"app", "--sync", "--output", "r.jpg"};
    call = cli::ProjectArgs(Words(sync, 4), defaults, "/p");
    const char* sync_out[] = {"app", "--output", "r.jpg", "--model-dir", "/p/models", "--graph",
                              "/p/graph.json"};
    GRAPH_CHECK(call.kind == cli::kSyncExecutor && call.args == Words(sync_out, 7));

    const char* check[] = {"app", "--check", "--json"};
    call = cli::ProjectArgs(Words(check, 3), defaults, "/p");
    const char* check_out[] = {"app", "--check", "--json", "--model-dir", "/p/models", "--graph",
                               "/p/graph.json"};
    GRAPH_CHECK(call.args == Words(check_out, 7));

    const char* given[] = {"app", "--check", "other.json", "--model-dir", "/m"};
    call = cli::ProjectArgs(Words(given, 5), defaults, "/p");
    GRAPH_CHECK(call.args == Words(given, 5));

    const char* graph[] = {"app", "--graph", "g2.json"};
    call = cli::ProjectArgs(Words(graph, 3), defaults, "/p");
    const char* graph_out[] = {"app", "--graph", "g2.json", "--model-dir", "/p/models"};
    GRAPH_CHECK(call.args == Words(graph_out, 5));

    const char* ops[] = {"app", "--list-ops", "--json"};
    call = cli::ProjectArgs(Words(ops, 3), defaults, "/p");
    const char* ops_out[] = {"app", "--list-ops", "--json", "--model-dir", "/p/models"};
    GRAPH_CHECK(call.args == Words(ops_out, 5));

    const char* help[] = {"app", "--help"};
    GRAPH_CHECK(cli::ProjectArgs(Words(help, 2), defaults, "/p").args.size() == 4);  // no --graph

    const char* absolute[] = {"app"};
    cli::ProjectDefaults fixed;
    fixed.graph = "/abs/g.json";
    fixed.model_dir = "/abs/models";
    const char* absolute_out[] = {"app", "--model-dir", "/abs/models", "--graph", "/abs/g.json"};
    GRAPH_CHECK(cli::ProjectArgs(Words(absolute, 1), fixed, "/p").args == Words(absolute_out, 5));
}

// ---------------------------------------------------------------------
// Phase 3 review: nothing a user class does ends the run
// ---------------------------------------------------------------------

namespace {

/// Its constructor throws: the class cannot be made.
struct ThrowsInItsConstructor {
    ThrowsInItsConstructor() { throw std::runtime_error("no camera config"); }
    BoxesData Run(const BoxesData& input, UserContext&) { return input; }
};

/// Throws something that is not a std::exception.
struct ThrowsAnInt {
    BoxesData Run(const BoxesData&, UserContext&) { throw 1; }
};

}  // namespace

void TestAUserClassThatCannotBeMadeOrThrowsAnythingPassesThrough() {
    ClearUserLogic();
    RegisterUserLogic<BoxesData, ThrowsInItsConstructor>("tag_boxes");
    std::vector<FrameReport> reports = RunLogicFrames(UserGraph(), LogicRegistry(), Keys(3), false);
    GRAPH_CHECK(reports.size() == 3);
    if (reports.size() == 3) {
        GRAPH_CHECK(reports[0].error.find("node \"u\": user logic \"tag_boxes\" could not be made: "
                                          "no camera config") != std::string::npos);
        GRAPH_CHECK(reports[1].error.empty() && reports[2].error.empty());  // said once, then pass-through
        const BoxesData* user = BoxesOf(reports[2], "u");
        GRAPH_CHECK(user != NULL && user->items.size() == 1 && user->items[0].class_name == "person");
        GRAPH_CHECK(EventsOf(reports[2], "z").size() == 1);  // the zone still runs
    }
    const std::vector<FrameReport> async = RunLogicFrames(UserGraph(), ReversedLogicRegistry(), Keys(3), true);
    GRAPH_CHECK(async.size() == 3);
    for (std::size_t i = 0; i < 3 && i < async.size() && i < reports.size(); ++i) {
        GRAPH_CHECK(reports[i] == async[i]);
    }
    ClearUserLogic();
    RegisterUserLogic<BoxesData, ThrowsAnInt>("tag_boxes");
    reports = RunLogicFrames(UserGraph(), ReversedLogicRegistry(), Keys(2), true);
    GRAPH_CHECK(reports.size() == 2);
    for (std::size_t i = 0; i < reports.size(); ++i) {
        GRAPH_CHECK(reports[i].error.find("user logic \"tag_boxes\" failed: an exception that is not a "
                                          "std::exception") != std::string::npos);
        const BoxesData* user = BoxesOf(reports[i], "u");
        GRAPH_CHECK(user != NULL && user->items.size() == 1);
    }
    ClearUserLogic();
}

void TestAUserNodeWithoutANamePassesThrough() {
    ClearUserLogic();
    RegisterUserLogic<BoxesData, TagBoxes>("tag_boxes");
    LogicConfig config;  // an unvalidated spec: no "name"
    config.id = "u";
    config.op = "user";
    LogicInput input;
    input.edge = 1;
    input.from = "od";
    std::shared_ptr<BoxesData> boxes(new BoxesData(Shape::kBoxes));
    boxes->items.push_back(MakeItem(10.f, 10.f, 20.f, 20.f, 0.9f, "person"));
    input.data = boxes;
    LogicFrame frame;
    frame.frame_size = cv::Size(100, 100);
    LogicState state;
    const LogicOutput out = RunLogic(config, std::vector<LogicInput>(1, input), frame, &state);
    const BoxesData* same = dynamic_cast<const BoxesData*>(out.result.data.get());
    GRAPH_CHECK(out.events.empty() && out.error.empty() && same != NULL && same->items[0].class_name == "person");
    ClearUserLogic();
}

void TestDxappRootLosesItsTrailingSlash() {
    RootEnv slash("/opt/exported//");
    GRAPH_CHECK(::dxapp::RuntimeRoot() == "/opt/exported");
    GRAPH_CHECK(::dxapp::ResolveRepoRelative("sample/a.jpg") == "/opt/exported/sample/a.jpg");
    RootEnv top("/");
    GRAPH_CHECK(::dxapp::RuntimeRoot() == "/");
}

}  // namespace graph
}  // namespace dxapp

int main() {
    using namespace dxapp::graph;
    TestShapeVocabulary();
    TestPendingJobsIssueEachIdOnceAndClaimEachJobOnce();
    TestDetectionConversion();
    TestFaceConversionKeepsLandmarks();
    TestObbConversionKeepsAngle();
    TestEmbeddingConversion();
    TestRoiRefDefaultsToIdentity();
    TestInstanceSegmentationConversion();
    TestPoseConversionIsDirectCopy();
    TestHandLandmarkConversion();
    TestPalmDetectorBoxesAreNamedHand();
    TestHandLandmarksAreDrawnWithTheHandSkeleton();
    TestSuperPointKeypointsAreDrawnAsPointsOnly();
    TestFaceAlignmentLandmarksAreDrawnWithTheFaceOutline();
    TestFaceAlignmentConversion();
    TestSegmentationConversionCatchesTransposition();
    TestSegmentationConversionDefensiveSizeCheck();
    TestDepthConversion();
    TestRestorationConversion();
    TestClassificationConversion();
    TestDetection3DConversion();
    TestFakeRegistryServesScriptedResults();
    TestFakeRegistryReportsNotReady();
    TestFakeStageSubmitDeliversScriptedResultAndHonoursOrder();
    TestFakeStageForwardFlushOrder();
    TestFakeStageFailurePropagatesToCallback();
    TestRegistryRejectsUnknownAndNotReadyStage();
    TestParseMinimalGraph();
    TestParseRejectsMalformedJson();
    TestParseRejectsWrongTypes();
    TestParseRejectsUnsupportedVersion();
    TestParseRejectsReservedKeys();
    TestParseRejectsTheLayerFileNamesAsNodeIds();
    TestSchemaRejectsAFractionalVersion();
    TestSchemaRejectsANonStringName();
    TestSchemaChecksEachRoiNumber();
    TestSchemaReportsEveryRoiViolationAtOnce();
    TestSchemaChecksTrackFields();
    TestSchemaKeepsTodaysSingleTypeErrors();
    TestValidateAcceptsCascade();
    TestValidateRejectsUnknownModel();
    TestValidateRejectsNotReadyModel();
    TestValidateRejectsRoiFromNonProducer();
    TestValidateRejectsRoiIntoFullFrameConsumer();
    TestValidateRejectsFrameIntoRoiOnlyConsumer();
    TestValidateRejectsAlignWithoutLandmarks();
    TestValidateAcceptsAlignFromFaceDetector();
    TestValidateRejectsCycle();
    TestValidateRejectsOrphanNode();
    TestValidateRejectsUnknownEndpoint();
    TestValidateRejectsSourceWithIncomingEdge();
    TestValidateRejectsDuplicateEdge();
    TestValidateRejectsNoSource();
    TestValidateRejectsBoxes3d();
    TestRouteFiltersByClassAndScore();
    TestRouteAppliesMaxCutByScore();
    TestRoutePadExpandsAndClips();
    TestRouteDropsFullyOutOfBoundsBox();
    TestRouteDropsBoxBelowMinArea();
    TestRouteEmptyProducerYieldsNoCrops();
    TestRestoreBoxMapsRoiLocalToSource();
    TestRestoreBoxIsIdentityForFullFrame();
    TestTrackIdsAreStableAcrossFrames();
    TestRouteCarriesTrackIdIntoRoiRef();
    TestUnrotateCropDropsFullyOutOfBoundsObb();
    TestFace5FallsBackToPlainCropWithTooFewLandmarks();
    TestUnrotateCropProducesNonEmptyCropForInBoundsObb();
    TestUnrotateCropWindowMatchesSrcBoxForPartiallyClippedObb();
    TestStageGraphBuildsTopologicalOrder();
    TestSyncExecutorRunsCascade();
    TestSyncExecutorHandlesEmptyDetection();
    TestSyncExecutorRejectsEmptyFrame();
    TestSyncExecutorIsolatesStageFailure();
    TestStageGraphBuildThrowsModelMissingWhenArtifactAbsent();
    TestBuildFallbackPrintsTheModelZooName();
    TestBuildReportsAnEngineThatWillNotLoadAsModelLoad();
    TestStaticRegistryCarriesModelZooNames();
    TestSyncExecutorMergesRoisFromTwoParents();
    TestStageGraphBuildThrowsModelUnknownWhenRegistryLookupMisses();
    TestOperatorEqualityDetectsFrameDivergence();
    TestOperatorEqualityDetectsVectorDivergence();
    TestOperatorEqualityDetectsBoxesFieldDivergence();
    TestOperatorEqualitySamePayloadDistinctObjectsEqual();
    TestOperatorEqualityComparesObBoxes();
    TestOperatorEqualityComparesInstanceMasks();
    TestOperatorEqualityComparesBoxes3d();
    TestOperatorEqualityEveryShapeComparesIdenticalPayloadsEqual();
    TestOperatorEqualityDetectsRoiResultsPayloadDivergence();
    TestOperatorEqualityDetectsRoiOriginFieldDivergence();

    TestAsyncMatchesSyncOnCascade();
    TestAsyncSortsOutOfOrderRoiCompletions();
    TestAsyncCompletesWhenProducerIsEmpty();
    TestAsyncKeepsTrackIdsFrameOrdered();
    TestAsyncIsolatesStageFailure();
    TestAsyncFanOutRunsAllBranches();
    TestAsyncMatchesSyncOnFanIn();
    TestAsyncMatchesSyncUnderBackPressure();
    TestAsyncMatchesSyncWhenProducerFails();
    TestAsyncMatchesSyncWhenChainedProducerFails();
    TestAsyncMatchesSyncOnMultipleFailures();
    TestAsyncResolvesErrorsInTopologicalOrder();
    TestAsyncLeavesNoPendingWorkAfterFrame();
    TestAsyncIgnoresDoubleDelivery();
    TestAsyncRejectsEmptyFrame();

    TestOperatorEqualityComparesLabelMapDenseMapAndKeypoints();
    TestAsyncOverlapsSiblingNodes();
    TestAsyncMatchesSyncOnHeadlineGraph();
    TestAsyncMatchesSyncOnPerCropFailureMessage();

    TestAsyncMatchesSyncOnClippedRois();
    TestAsyncOverlapsAcrossDepth();
    TestAsyncDrainsNeverDispatchedStage();
    TestValidateRejectsMixedFrameAndRoiInput();
    TestValidateAcceptsTwoRoiParents();

    TestStaticRegistryResolvesKnownModel();
    TestStaticRegistryShapeComesFromFactoryNotTask();
    TestStaticRegistryEveryEntryIsWellFormed();
    TestStaticRegistryCreateStageRejectsUnknownModel();
    TestStaticRegistryCreateStageWrapsRuntimeFailure();
    TestStageDataFromResultsDispatchesBothOverloads();
    TestNodeParamsOverlayReachesModelConfig();
    TestOperatorEqualityComparesScores();
    TestAsyncSurvivesStageThatThrowsFromSubmit();
    TestAsyncSurvivesStageThatThrowsFromFlush();
    TestRealStageRunAndSubmitAgree();
    TestRealStageSyncAsyncParity();
    TestRealStagePipelinedMatchesSyncAcrossFrames();

    TestRestorePointWarpedUndoesAlignment();
    TestRestorePointWarpedIsTranslationForPlainCrop();
    TestRestorePointWarpedIsIdentityForFullFrame();
    TestRenderReportIsDeterministic();
    TestRenderReportHandlesEmptyReport();
    TestRenderReportHandlesEmptySource();
    TestRestoreBoxComposesInvAlign();
    TestRestoreBoxUnderRotationIsAxisAlignedBoundingBox();
    TestRenderReportBlendsInstanceMaskAtItsOwnRegion();
    TestRenderReportResizesInstanceMaskToRoiRegion();
    TestRestoreBoxCornerRoundTripsThroughRealObbCrop();
    TestRestorePointWarpedRoundTripsThroughRealFace5Crop();
    TestRenderReportDrawsRoiResultsNotJustNodeResults();
    TestRenderReportRotatesObBoxByItemAngle();

    TestParseRejectsUnknownRoiKey();
    TestParseRejectsUnknownNodeKey();
    TestParseRejectsUnknownTrackKey();
    TestParseSuggestsTheTransposedKeyNotTheShorterOne();
    TestParseRejectsUnknownTopLevelKey();
    TestParseOffersNoSuggestionForAnUnrelatedKey();
    TestUnsupportedVersionOutranksUnknownKeys();
    TestReservedKeyOutranksUnknownKey();
    TestParseAcceptsEverySchemaKeyAndLeavesParamsOpen();
    TestEveryShippedSampleGraphStillParses();

    TestParseSuggestsForAMistypedRequiredKey();
    TestMissingRequiredKeyStillCarriesLabelAndRemedy();
    TestParseReportsEveryUnknownKeyInOneMessage();
    TestParseRejectsUnknownNodeType();
    TestTopLevelSchemaKeyIsAcceptedAndInert();

    TestValidateRejectsAPlainEdgeThatWouldDiscardItsProducer();
    TestValidateStillAcceptsEveryPlainEdgeFromASource();
    TestValidateSuggestsANearbyModelName();

    TestFakeStagePollDeliversPending();
    TestFakeStageCanWithholdCompletionsFromFlush();
    TestFakeStageShuffleIsDeterministic();

    TestAsyncOptionsDefaults();
    TestAsyncRejectsZeroFramesInFlight();

    TestAsyncCompletesWhenStageDeliversOnlyOnPoll();
    TestAsyncStallTimeoutNamesTheStuckStage();
    TestAsyncReportsStuckStagesOnceWhenInterrupted();
    TestAsyncOptionsWaitDefaults();
    TestAsyncReportsExceptionEscapingPoll();
    TestAsyncSurvivesGraphRebuiltInPlace();

    TestAsyncKeepsSeveralFramesInFlight();
    TestAsyncTrackerGateOrdersReversedCompletions();
    TestAsyncMatchesSyncAcrossFramesUnderShuffledDelivery();
    TestAsyncEmitsReportsInFrameOrder();
    TestAsyncEmptyFrameDoesNotBlockTheTrackerGate();
    TestAsyncGateAdvancesPastFailedTrackedNode();
    TestAsyncNeverAdmitsPastTheFrameCap();
    TestAsyncSubmitsEverythingThenDrainsInOrder();
    TestAsyncRunFrameRefusesWhileFramesAreInFlight();
    TestAsyncRunFrameRefusesWhileAFinishedReportIsUntaken();
    TestAsyncTryNextHandsBackTheReportsOwnFrame();
    TestAsyncRebindsTrackerGatesAfterInPlaceRebuild();

    TestAsyncHeldReportsCountAgainstTheFrameCap();
    TestAsyncRebindsWhileReportsAreUntaken();
    TestAsyncDoesNotNameHealthyStagesWhenInterrupted();
    TestAsyncStuckReportDefault();
    TestAsyncMatchesSyncOnHeadlineAcrossFramesUnderShuffledDelivery();
    TestAsyncMatchesSyncOnFanInAcrossFrames();
    TestAsyncMatchesSyncAcrossFramesWhenTrackedNodeFails();
    TestAsyncMatchesSyncAcrossFramesWithOneJobPerStage();
    TestAsyncMatchesSyncWithCrossThreadDelivery();
    TestHandOffProblemNamesEachUnusableImage();
    TestHandOffViewScalesPerAxis();
    TestHandOffViewComposesAChain();
    TestRestoreHelpersApplyInvAlignForFullFrameResults();
    TestRestorePointOnPlainCropIsUnchanged();
    TestRouteRoisIdentityViewEqualsFrameOverload();
    TestRouteRoisOverScaledViewCutsFromTheViewImage();
    TestRouteRoisOverScaledViewClipsToTheViewImage();
    TestSamePayloadComparesImages();
    TestTargetRegionPlacesScaledCropOnItsSourceBox();

    TestValidateAcceptsImageHandOff();
    TestValidateRejectsHandOffFromPerCropNode();
    TestValidateRejectsHandOffOfNonImage();
    TestValidateRejectsTwoFullFrameImages();
    TestValidateStillAcceptsTwoSourceEdges();
    TestValidateStillRejectsFrameAndRoiIntoOneNode();

    TestSyncHandsOffTheProducersImage();
    TestSyncRoutesCropsFromTheHandedOffImage();
    TestSyncHandOffChainComposesScales();
    TestSyncHandOffErrorIsReportedOnTheConsumer();
    TestSyncHandOffProducerFailureSkipsTheConsumer();
    TestSyncWithoutHandOffIsUnchanged();

    TestAsyncHandOffMatchesSync();
    TestAsyncHandOffMatchesSyncAcrossFramesUnderShuffledDelivery();
    TestAsyncHandOffErrorFrameMidStreamMatchesSync();
    TestAsyncTrackedDetectorAfterHandOffMatchesSync();
    TestTrackedDetectorAfterHandOffKeepsItsOrigin();
    TestRealStageHandOffMatchesSync();

    TestRenderReportBlendsMapsOverAHandedOffImage();
    TestAsyncHandOffFanOutMatchesSyncUnderReversedDelivery();
    TestAsyncHandOffChainComposesScalesLikeSync();
    TestWarpedCropsFromAScaledViewRestoreToTheSource();
    TestRenderReportDrawsAHandOffChildsBoxAtSourceCoordinates();
    TestHandOffErrorAndANodeErrorLeaveTheSameSurvivor();
    TestAsyncHarvestKeepsAConsumersHandOffError();

    TestRenderReportDrawsAnInstancesBoxOnItsRestoredEdge();
    TestRealStageInstanceSegSyncAsyncParity();
    TestRenderReportWarpsDenseResultsOntoARealObbCrop();
    TestRenderReportWarpsDenseResultsOntoARealFace5Crop();
    TestRenderReportWarpsDenseResultsOntoAnObbCropFromAScaledView();
    TestRenderReportWarpsARotatedCropThatCrossesTheCanvasEdge();
    TestRenderReportKeepsPlainCropDenseResultsBitExact();

    TestReportFileWriterWritesWhatTheWholeFileDumpWrote();
    TestReportFileWriterLeavesAFramePrefixAndFinishesOnDestruction();
    TestReportFileWriterReportsAPathItCannotOpen();
    TestReportJsonSpellsNonFiniteNumbersDistinctly();

    TestCliOutputRules();
    TestScopedOpenCvLogLevelRestoresTheLevel();
    TestVideoOutputKeepsEveryFrameAtTheFirstFramesSize();

    TestVideoOutputOpenFailureOffersAviOnlyWhenItIsNotAvi();
    TestCliOutputDirectoryIsCheckedUpFront();
    TestReportFileWriterClosesWhenTheHeaderCannotBeWritten();
    TestAsyncHugeStallTimeoutDoesNotStallAtOnce();

    TestListStreamsHasOneStreamPerSource();
    TestValidateCountsFullFrameImagesPerStream();

    TestStageGraphPlansEachStream();
    TestSyncStreamRunsOnlyItsOwnSubgraph();
    TestSyncStreamsKeepTheirOwnTrackIds();
    TestExecutorsRefuseAnUnnamedStreamOnSeveralSources();
    TestReportsNameTheirStream();

    TestAsyncStreamsMatchSyncUnderReversedDetectorDelivery();
    TestAsyncStreamGatesOrderEachStreamOnItsOwnWhenOneEndsEarly();
    TestAsyncSharedNodeRunsOncePerStreamFrame();
    TestAsyncFrameWindowIsSharedByEveryStream();

    TestReportJsonNamesTheStreamOnlyForSeveralSources();
    TestParseInputOverridesBindsOnlyKnownSourceIds();
    TestResolveStreamsGivesEverySourceItsUri();
    TestOpenStreamKeepsTheOneSourceWording();
    TestStreamReaderReadsInTurnAndStopsEachStreamAtItsLimit();
    TestStreamOutputPaths();
    TestFrameExecutorsSubmitToTheNamedStream();

    TestValidateNamesTheLaterStreamAFullFrameClashIsIn();
    TestValidateReportsTheOrphanOfAnUnreachableFullFrameClash();
    TestStreamReaderSkipsAMiddleStreamThatEndsEarly();
    TestParseInputOverridesBindsTheLongestSourceIdBeforeAnEquals();

    TestParseReadsAnEdgePort();
    TestValidateEdgePorts();
    TestSyncRoutesEachEdgeFromItsPort();
    TestAsyncRoutesPortsLikeSyncUnderReversedDelivery();
    TestTrackedProducerKeepsItsPorts();
    TestRoiEdgeFromABrokenPortRecordsAnError();
    TestReportsDifferingOnlyInAPortAreUnequal();
    TestRoiResultsKeepTheirPortsUnderReversedDelivery();

    TestReportWritesPortsOnlyWhenThereAreSome();
    TestRenderReportDrawsPortsInTheirShapesPass();

    TestVitPoseIsATopDownRoiConsumer();

    TestPanopticConversionKeepsTheBoxesAndAddsBothMasks();
    TestPoseDescriptorsBecomeOneMatrixRowPerKeypoint();
    TestAFailedPortConversionLeavesTheResultEmpty();
    TestHandednessBecomesOneScorePerHand();
    TestFacePoseBecomesThreeFloatsPerFace();
    TestPortProblemNamesAPortTheResultCannotFill();
    TestRealRegistryDeclaresPortsOnlyWhereTheyExist();
    TestRealPortModelsReportTheirPortsAndAgreeAcrossExecutors();

    TestParseAcceptsTextAndListParams();
    TestParseRejectsParamsThatAreNeitherNumberTextNorList();
    TestValidateChecksEachParamAgainstTheModelsKinds();
    TestBuildHandsTextAndListParamsToTheRegistry();
    TestParamsToJsonKeepsTheNumericBytesAndAddsTextAndLists();
    TestParamsOverlayKeepsConfigJsonArrays();
    TestParamsToJsonRoundTripsEveryCharacterThroughModelConfig();
    TestParamsToJsonEscapesControlCharactersAndKeys();
    TestParseGivesTheSwitchHintOnce();
    TestRealRegistryKnowsWhichKeysAreText();

    TestRealTiledSrStageHonoursTheStageContract();
    TestRealTiledSrHandOffMatchesSync();

    TestReadDxnnContainerVersion();
    TestContainerSupportRefusesV9BelowDxrt350();
    TestAStageRefusesAV9FileBeforeOpeningAnEngine();
    TestRegistryResolvesOldNamesAndAliasesToTheVariant();
    TestRegistryCarriesPublishedAndResources();
    TestBuildOffersADownloadOfAPublishedModel();
    TestStageConfigIsTheVariantsConfigJson();
    TestStageConfigPathRefusesAModelInfoWithoutFamilyOrVariant();
    TestARelativeGalleryIsReadAgainstTheRepository();
    TestResolveRepoRelativeKeepsAbsoluteAndJoinsRelative();
    TestMattingAlphaBecomesADenseMapPort();
    TestDepthConversionOwnsItsValues();
    TestGalleryMatchesBecomeAScoresPort();

    TestCpuNodeRejectsAGenericFuseAndABadResultEdge();
    TestHeadPoseMatchesSyncAndAsync();
    TestVolumeProxyMatchesSyncAndAsync();
    TestVolumeSamplesADepthMapOfAnotherSize();
    TestVolumeMapsAHandedOffSegmenterToSourcePixels();
    TestVolumeRefusesARasterThatDoesNotCoverTheFrame();
    TestHeadPosePitchIsNegativeForAHeadTurnedDown();
    TestHeadPoseDrawsItsAxesAndItsAngles();
    TestHeadPoseOverlayReachesTheReportInBothExecutors();
    TestHeadPoseWritesOneRecordPerFace();
    TestDriverAttentionFiresWhenAnyHeadIsDown();
    TestSyncExecutorTimesEveryNode();
    TestAsyncExecutorTimesEveryNode();
    TestReportToJsonWritesNodeMsOnlyWhenAsked();
    TestEngineKeyParsesAndValidates();
    TestEngineBuffersReachTheRegistry();
    TestEngineBufferScopeSetsAndRestores();
    TestSharedNodesUseOneStage();
    TestASharedStageKeepsEachNodesJobsApartUnderPipelining();
    TestBuildRejectsAnUnvalidatedSharerWithOtherSettings();
    TestRealSharedEngineWithBuffers();
    TestSharedEngineRestrictions();
    TestAFailedLoadClosesEveryEngineItOpened();
    TestGraphErrorKeepsItsThreeParts();
    TestLogicOpCatalogDescribesEveryOp();
    TestParseReadsALogicNode();
    TestParseRejectsBadLogicNodes();
    TestValidateTakesALogicNodeAsANonModelNode();
    TestValidateChecksLogicParams();
    TestValidateChecksLogicInputs();
    TestValidateChecksTheGalleryFile();
    TestAGalleryLargerThanItsFileIsRefusedBeforeAllocating();
    TestNodeOutputShapeFollowsFilterZoneAndUser();
    TestFrameTimeCountsFramesOrWallSeconds();
    TestReportsCarryTheFrameTime();
    TestOperatorEqualityComparesTimeEventsAndOverlays();
    TestReportToJsonWritesEventsOnlyWhenThereAreSome();
    TestFilterKeepsClassScoreAndArea();
    TestZoneKeepsTheBoxesInsideAndRaisesEnterAndExit();
    TestBuildGivesEachLogicNodeAStatePerStream();
    TestAsyncLogicMatchesSyncOverFrames();
    TestAsyncListsEventsInNodeOrder();
    TestCpuNodeBehindAFilterGetsOnlyTheFilteredBoxes();
    TestEventDataKeepsEachKeyOnce();
    TestLineCountsCrossingsBothWays();
    TestCountCountsEachInputAndClass();
    TestDwellTimesEachStayAndRaisesOneEvent();
    TestConditionRaisesOnTheRisingEdgeOrEveryCooldown();
    TestAsyncLogicGateOrdersReversedCompletions();
    TestPerCropScoresReachACondition();
    TestGalleryNamesEachCropAndAnnouncesNameChanges();
    TestUserNodePassesItsInputThrough();
    TestRenderReportDrawsLogicOverlaysButNotTheirBoxes();
    TestEventFileWriterWritesOneLinePerEvent();
    TestStreamReaderLoopsAVideo();
    TestConditionOnRecordsNeedsAValue();
    TestABooleanOnALogicNodeGetsItsParamsHint();
    TestCountRejectsTotalAndRepeatedClasses();
    TestOnlyLogicNodesThatDrawHaveAnOverlay();
    TestTimeThresholdsHoldOnTheFrameTheyAreDue();
    TestLogicForgetsATrackAfterFiveSeconds();
    TestTwoStreamsKeepTheirOwnLogicState();
    TestConditionReadsRecordsNumbers();
    TestGalleryMatchesOnlyAVectorWithADirection();
    TestLogicLabelsCoverNeitherBoxLabelsNorTheInArrow();
    TestConditionComparesRecordValuesWithSlack();
    TestARestartedStreamStartsItsTracksAndLogicAgain();
    TestSecondsAreWrittenRoundedLikeTime();
    TestAGalleryNodeQuietsTheTextOfTheNodeItReads();
    TestAVectorPortDrawsNoVecLabel();
    TestEventLogKeepsOffTheLabelsRenderReportDrew();
    TestDrawFilterDrawsOnlyTheNamedNodes();
    TestLayerFromBackdropsRecoversColourAndAlpha();
    TestLayersStackBackToTheMergedImage();
    TestDisplayGridTilesTheStreams();
    TestLayerFileNames();
    TestRuntimeRootIsDxappRootElseTheBuildsRepository();
    TestFindMarkedRootWalksUpToTheMarker();
    TestUserNodeNamesAreCppNamesThatAreNotKeywords();
    TestUserNodesSharingANameTakeOneKind();
    TestUserLogicRegistryMakesOneObjectPerCall();
    TestARegisteredUserClassRunsInPlaceOfThePassThrough();
    TestEachLogicStateHasItsOwnUserObject();
    TestUserClassFailuresPassTheInputThroughAndSayWhy();
    TestAUserClassThatCannotBeMadeOrThrowsAnythingPassesThrough();
    TestAUserNodeWithoutANamePassesThrough();
    TestDxappRootLosesItsTrailingSlash();
    TestProjectArgsFillInTheProjectsGraphAndModels();

    std::printf("%d checks, %d failures, %d skipped\n",
                g_checks, g_failures, g_skipped);
    return g_failures == 0 ? 0 : 1;
}

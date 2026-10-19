#include "common/graph/roi_router.hpp"

#include "common/utility/roi_crop.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace dxapp {
namespace graph {
namespace {

struct ScoredIndex {
    std::size_t index;
    float score;
};

bool HigherScore(const ScoredIndex& a, const ScoredIndex& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.index < b.index;  // stable, so sync and async agree
}

bool ClassAllowed(const RoiSpec& spec, const BoxItem& item) {
    if (spec.classes.empty()) return true;
    for (std::size_t i = 0; i < spec.classes.size(); ++i) {
        if (spec.classes[i] == item.class_name) return true;
    }
    return false;
}

/// Five-point similarity transform onto the ArcFace reference layout.
bool BuildFace5Transform(const std::vector<Keypoint>& landmarks,
                         int output_size, cv::Matx23f* forward) {
    if (landmarks.size() < 5) return false;

    // ArcFace canonical 112x112 landmarks, scaled to output_size.
    static const float kReference[5][2] = {
        {38.2946f, 51.6963f}, {73.5318f, 51.5014f}, {56.0252f, 71.7366f},
        {41.5493f, 92.3655f}, {70.7299f, 92.2041f}};
    const float scale = static_cast<float>(output_size) / 112.f;

    std::vector<cv::Point2f> source;
    std::vector<cv::Point2f> target;
    for (int i = 0; i < 5; ++i) {
        source.push_back(cv::Point2f(landmarks[i].x, landmarks[i].y));
        target.push_back(cv::Point2f(kReference[i][0] * scale,
                                     kReference[i][1] * scale));
    }
    const cv::Mat estimated = cv::estimateAffinePartial2D(source, target);
    if (estimated.empty()) return false;

    cv::Mat as_float;
    estimated.convertTo(as_float, CV_32F);
    *forward = cv::Matx23f(as_float.at<float>(0, 0), as_float.at<float>(0, 1),
                           as_float.at<float>(0, 2), as_float.at<float>(1, 0),
                           as_float.at<float>(1, 1), as_float.at<float>(1, 2));
    return true;
}

/// A pure translation, as a 2x3 affine.
cv::Matx23f Translate(float tx, float ty) {
    return cv::Matx23f(1.f, 0.f, tx, 0.f, 1.f, ty);
}

/// A per-axis scale, as a 2x3 affine.
cv::Matx23f Scale(float sx, float sy) {
    return cv::Matx23f(sx, 0.f, 0.f, 0.f, sy, 0.f);
}

/**
 * @brief Compose two 2x3 affines: result(p) = outer(inner(p)).
 *
 * Used to fold a crop's own translation into its warp so `RoiRef::inv_align`
 * always maps crop-local coordinates directly to the source frame — see
 * RouteRois's own comment on why this composition, and in which order, is
 * required for the OBB branch.
 */
cv::Matx23f ComposeAffine(const cv::Matx23f& outer, const cv::Matx23f& inner) {
    const float r00 = outer(0, 0) * inner(0, 0) + outer(0, 1) * inner(1, 0);
    const float r01 = outer(0, 0) * inner(0, 1) + outer(0, 1) * inner(1, 1);
    const float r10 = outer(1, 0) * inner(0, 0) + outer(1, 1) * inner(1, 0);
    const float r11 = outer(1, 0) * inner(0, 1) + outer(1, 1) * inner(1, 1);
    const float t0 = outer(0, 0) * inner(0, 2) + outer(0, 1) * inner(1, 2) + outer(0, 2);
    const float t1 = outer(1, 0) * inner(0, 2) + outer(1, 1) * inner(1, 2) + outer(1, 2);
    return cv::Matx23f(r00, r01, t0, r10, r11, t1);
}

cv::Matx23f InvertAffine(const cv::Matx23f& forward) {
    cv::Mat as_mat(2, 3, CV_32F);
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 3; ++c) as_mat.at<float>(r, c) = forward(r, c);
    }
    cv::Mat inverted;
    cv::invertAffineTransform(as_mat, inverted);
    cv::Mat as_float;
    inverted.convertTo(as_float, CV_32F);
    return cv::Matx23f(as_float.at<float>(0, 0), as_float.at<float>(0, 1),
                       as_float.at<float>(0, 2), as_float.at<float>(1, 0),
                       as_float.at<float>(1, 1), as_float.at<float>(1, 2));
}

/**
 * @brief Rotate the full frame around the box's own centre, then extract the
 *        already frame-clipped axis-aligned window.
 *
 * Why rotate-then-clip, and not clip-then-rotate: clipping the source to the
 * (unrotated) padded box FIRST and rotating that sub-image would discard
 * exactly the pixels the rotation needs to fill the corners of the final
 * axis-aligned window — cv::warpAffine can only draw from pixels present in
 * its input, so a pre-rotation clip produces incorrect (black/garbage)
 * content near the box edges instead of real source pixels. Rotating the
 * whole frame first and clipping the extraction window afterward is the
 * only way to realize "pad+clip -> un-rotate" for an OBB without losing
 * information the rotation needs — this is still order-faithful to the
 * pinned "pad+clip -> align/un-rotate -> crop" chain: `extraction_window`
 * below is exactly the pad+clip step's own output, just applied to the
 * rotated frame instead of the unrotated one, because clip is naturally the
 * LAST geometric step before the pixels are read out, not something that
 * has to touch memory before rotation runs.
 *
 * @param padded_box the box AFTER pad has already been applied (same centre
 *        as the raw detection, since pad is symmetric) — used only to derive
 *        the rotation centre; the pad+clip step happens before align/un-
 *        rotate in the pinned operator order, so this function must never
 *        be fed the raw, unpadded box.
 * @param extraction_window MUST be the exact same rect RouteRois already
 *        used to build the RoiCrop's own `ref.src_box` (today: `clipped`,
 *        from ClipBoxToFrame(padded_box, source.cols, source.rows)) —
 *        passed in, not recomputed here. That is the actual invariant this
 *        parameter exists to enforce: `crop.image`'s pixel size and
 *        `ref.src_box`'s recorded size must describe the same window,
 *        because everything downstream
 *        reads them as if they always agree — RouteRois folds this same
 *        `clipped`/`ref.src_box` translation into `ref.inv_align` itself
 *        (`ComposeAffine(inverse, Translate(clipped.x, clipped.y))` for
 *        this OBB path), so `ref.inv_align` alone maps a crop-local
 *        coordinate to the source frame; RestoreBox/RestoreBoxCorners
 *        apply nothing beyond `ref.inv_align`. None of that re-derives
 *        the window from the pixels, so a crop and its own RoiRef silently
 *        disagreeing about their window is not something anything downstream
 *        would ever catch.
 *
 *        This is NOT a claim that recomputing the window here from
 *        `centre`/size and re-intersecting it against the rotated frame's
 *        bounds would be numerically identical to `ClipBoxToFrame`'s result —
 *        it would not be, in general: the two clip a fractional box in a
 *        different order (`ClipBoxToFrame` clamps in float, then truncates to
 *        int once; a centre+size recompute truncates each float coordinate
 *        to int first, then intersects), so a padded box straddling a frame
 *        edge with fractional coordinates can differ by a pixel between the
 *        two paths (e.g. `padded.x = 634.7, padded.width = 8.5, cols =
 *        640`: `ClipBoxToFrame` gives `x=634, width=5`; the centre+size
 *        recompute gives `x=634, width=6`). Neither path crashes — both
 *        results are valid rects inside `rotated`, whose size equals
 *        source.size() — which is exactly why a stale recompute here would
 *        pass every existing test silently: it fails on agreement between
 *        the crop and its RoiRef, not on bounds. Passing the caller's own
 *        window closes that gap by construction rather than by argument.
 *        Still guarded defensively below rather than assumed non-empty,
 *        since this function does not otherwise enforce that precondition
 *        on its caller.
 */
bool UnrotateCrop(const cv::Mat& source, const cv::Rect2f& padded_box,
                  float angle_radians, const cv::Rect& extraction_window,
                  cv::Mat* crop, cv::Matx23f* inverse) {
    if (extraction_window.width <= 0 || extraction_window.height <= 0) return false;

    const cv::Point2f centre(padded_box.x + padded_box.width * 0.5f,
                             padded_box.y + padded_box.height * 0.5f);
    const float degrees = angle_radians * 180.f / static_cast<float>(CV_PI);

    const cv::Mat rotation = cv::getRotationMatrix2D(centre, degrees, 1.0);
    cv::Mat rotated;
    cv::warpAffine(source, rotated, rotation, source.size(), cv::INTER_LINEAR);

    *crop = rotated(extraction_window).clone();

    cv::Mat as_float;
    rotation.convertTo(as_float, CV_32F);
    const cv::Matx23f forward(
        as_float.at<float>(0, 0), as_float.at<float>(0, 1),
        as_float.at<float>(0, 2), as_float.at<float>(1, 0),
        as_float.at<float>(1, 1), as_float.at<float>(1, 2));
    *inverse = InvertAffine(forward);
    return true;
}

}  // namespace

bool IsIdentityAffine(const cv::Matx23f& m) {
    return m(0, 0) == 1.f && m(0, 1) == 0.f && m(0, 2) == 0.f &&
           m(1, 0) == 0.f && m(1, 1) == 1.f && m(1, 2) == 0.f;
}

std::string HandOffProblem(const StageDataPtr& data) {
    const ImageData* image =
        data ? dynamic_cast<const ImageData*>(data.get()) : NULL;
    if (image == NULL) return "handed off no image";
    if (image->image.empty()) return "handed off an empty image";
    if (image->image.type() != CV_8UC3) return "handed off a non-BGR image";
    return std::string();
}

// Every image model in the registry stretches its WHOLE input per axis
// (restoreSourceGeometry, zero_dce, espcn), so the output-to-input map is a
// per-axis scale. A model that crops or pads would need its own mapping.
FrameView HandOffView(const ImageData& produced, const FrameView& producer_input) {
    FrameView out;
    out.image = produced.image;
    const float sx = static_cast<float>(producer_input.image.cols) / produced.image.cols;
    const float sy = static_cast<float>(producer_input.image.rows) / produced.image.rows;
    out.to_source = ComposeAffine(producer_input.to_source, Scale(sx, sy));
    return out;
}

void ApplyTrackIds(BoxesData* boxes, IouTracker* tracker) {
    if (boxes == NULL || tracker == NULL) return;

    std::vector<std::vector<float> > rects;
    rects.reserve(boxes->items.size());
    for (std::size_t i = 0; i < boxes->items.size(); ++i) {
        const cv::Rect2f& box = boxes->items[i].box;
        std::vector<float> rect;
        rect.push_back(box.x);
        rect.push_back(box.y);
        rect.push_back(box.x + box.width);
        rect.push_back(box.y + box.height);
        rects.push_back(rect);
    }

    const std::vector<int> ids = tracker->update(rects);
    for (std::size_t i = 0; i < boxes->items.size() && i < ids.size(); ++i) {
        boxes->items[i].track_id = ids[i];
    }
}

std::vector<RoiCrop> RouteRois(const BoxesData& boxes, const cv::Mat& source,
                               const RoiSpec& spec,
                               const std::string& parent_node,
                               IouTracker* /*tracker*/, RouteStats* stats) {
    std::vector<RoiCrop> crops;
    RouteStats local;
    RouteStats* out = stats != NULL ? stats : &local;

    if (source.empty()) return crops;

    // 1. classes / min_score / min_area filter
    std::vector<ScoredIndex> kept;
    for (std::size_t i = 0; i < boxes.items.size(); ++i) {
        const BoxItem& item = boxes.items[i];
        const bool passes =
            ClassAllowed(spec, item) &&
            (spec.min_score < 0.f || item.score >= spec.min_score) &&
            (item.box.width * item.box.height >= spec.min_area);
        if (!passes) {
            out->filtered += 1;
            continue;
        }
        ScoredIndex entry;
        entry.index = i;
        entry.score = item.score;
        kept.push_back(entry);
    }

    // 2. max cut, highest score first
    std::sort(kept.begin(), kept.end(), HigherScore);
    if (spec.max >= 0 && kept.size() > static_cast<std::size_t>(spec.max)) {
        out->filtered += static_cast<int>(kept.size() - spec.max);
        kept.resize(spec.max);
    }

    // 3. track ids are already on the items (ApplyTrackIds, frame-ordered) —
    //    this function only reads item.track_id, never assigns it.
    // 4-6. pad + clip, align/un-rotate, crop — in that pinned order.
    int emitted = 0;
    for (std::size_t k = 0; k < kept.size(); ++k) {
        const std::size_t index = kept[k].index;
        const BoxItem& item = boxes.items[index];

        // 4. pad, then clip to the frame (the shared rule in
        //    common/utility/roi_crop.hpp).
        const cv::Rect2f padded = PadBox(item.box, spec.pad);
        const cv::Rect clipped = ClipBoxToFrame(padded, source.cols, source.rows);
        if (clipped.width <= 0 || clipped.height <= 0 ||
            static_cast<float>(clipped.width) * clipped.height < spec.min_area) {
            // Review Focus #4: a box whose padded rect falls entirely (or
            // below min_area) outside the frame is dropped and counted here,
            // never reaching cv::resize / cv::Mat::operator(), both of which
            // throw on an empty region.
            out->clipped_away += 1;
            continue;
        }

        RoiCrop crop;
        crop.ref.from_roi = true;
        crop.ref.src_box = cv::Rect2f(
            static_cast<float>(clipped.x), static_cast<float>(clipped.y),
            static_cast<float>(clipped.width), static_cast<float>(clipped.height));
        crop.ref.track_id = item.track_id;
        crop.ref.parent_node = parent_node;
        crop.ref.parent_index = static_cast<int>(index);
        crop.ref.roi_index = emitted;

        // ref.inv_align's domain is normalized here, once, so every
        // consumer downstream can apply it alone: it always maps a
        // crop-local coordinate directly to the source frame, with no
        // separate "+ ref.src_box" step required or permitted anywhere
        // else. The three crop kinds need three different compositions to
        // establish that:
        //   - plain crop: local == source pixels shifted by (clipped.x,
        //     clipped.y), so inv_align is that translation alone.
        //   - OBB: UnrotateCrop rotates the WHOLE FRAME by `inverse`'s
        //     forward (see its own doc comment), THEN clips the window
        //     `clipped` out of the ROTATED frame — so a crop-local point
        //     is first translated into the rotated frame's coordinates
        //     (+clipped.x/y), and ONLY THEN un-rotated back to the source
        //     frame (`inverse`). Composing translate-then-rotate in the
        //     other order (rotate-then-translate) silently returns the
        //     wrong point for any non-identity rotation — verified against
        //     RouteRois's own real output in
        //     TestRestoreBoxCornerRoundTripsThroughRealObbCrop.
        //   - face5: BuildFace5Transform's `forward` warps the SOURCE
        //     directly onto the `side x side` output
        //     (cv::warpAffine(source, warped, forward, ...)), with no
        //     intermediate translation step at all — the crop is not a
        //     sub-window of anything, so `inverse` (== InvertAffine(forward))
        //     is already the complete local-to-source map on its own.
        //
        // 5. align (face5) or un-rotate (OBB), 6. crop.
        if (spec.align == "face5") {
            const int side = std::max(clipped.width, clipped.height);
            cv::Matx23f forward;
            if (BuildFace5Transform(item.landmarks, side, &forward)) {
                cv::Mat as_mat(2, 3, CV_32F);
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 3; ++c) as_mat.at<float>(r, c) = forward(r, c);
                }
                cv::Mat warped;
                cv::warpAffine(source, warped, as_mat, cv::Size(side, side),
                               cv::INTER_LINEAR);
                crop.image = warped;
                crop.ref.inv_align = InvertAffine(forward);  // no src_box term
            } else {
                // Not enough landmarks: fall back to the plain crop rather
                // than dropping a detection the user asked to process.
                crop.image = source(clipped).clone();
                crop.ref.inv_align =
                    Translate(crop.ref.src_box.x, crop.ref.src_box.y);
            }
        } else if (boxes.shape() == Shape::kObBoxes && item.angle != 0.f) {
            cv::Mat unrotated;
            cv::Matx23f inverse;
            if (!UnrotateCrop(source, padded, item.angle, clipped, &unrotated, &inverse)) {
                out->clipped_away += 1;
                continue;
            }
            crop.image = unrotated;
            crop.ref.inv_align = ComposeAffine(
                inverse, Translate(crop.ref.src_box.x, crop.ref.src_box.y));
        } else {
            crop.image = source(clipped).clone();
            crop.ref.inv_align = Translate(crop.ref.src_box.x, crop.ref.src_box.y);
        }

        if (crop.image.empty()) {
            out->clipped_away += 1;
            continue;
        }
        crop.ref.crop_size = crop.image.size();

        crops.push_back(crop);
        ++emitted;
    }

    return crops;
}

std::vector<RoiCrop> RouteRois(const BoxesData& boxes, const FrameView& view,
                               const RoiSpec& spec, const std::string& parent_node,
                               IouTracker* tracker, RouteStats* stats) {
    std::vector<RoiCrop> crops =
        RouteRois(boxes, view.image, spec, parent_node, tracker, stats);
    if (IsIdentityAffine(view.to_source)) return crops;   // bit-exact old path
    // The frame overload built every ref against view.image; re-express it
    // against the source frame. src_box becomes the axis-aligned bounding
    // box of the view-space window in source coordinates, inv_align maps
    // crop-local -> view -> source, and crop_size stays the pixel size.
    RoiRef to_source;                                       // from_roi == false
    to_source.inv_align = view.to_source;
    for (std::size_t i = 0; i < crops.size(); ++i) {
        crops[i].ref.src_box = RestoreBox(crops[i].ref.src_box, to_source);
        crops[i].ref.inv_align = ComposeAffine(view.to_source, crops[i].ref.inv_align);
    }
    return crops;
}

std::vector<cv::Point2f> RestoreBoxCorners(const cv::Rect2f& roi_local,
                                           const RoiRef& ref) {
    std::vector<cv::Point2f> corners;
    corners.reserve(4);
    const float xs[2] = {roi_local.x, roi_local.x + roi_local.width};
    const float ys[2] = {roi_local.y, roi_local.y + roi_local.height};

    // Identity fast path: a full-frame result on the source frame is
    // returned bit-exact, without arithmetic. A full-frame result from a
    // handed-off image (from_roi == false, inv_align = the view's scale
    // back to the source) falls through and is mapped like any crop.
    if (!ref.from_roi && IsIdentityAffine(ref.inv_align)) {
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) corners.push_back(cv::Point2f(xs[i], ys[j]));
        }
        return corners;
    }

    // ref.inv_align alone maps a crop-local (or handed-off-image) coordinate
    // to the source frame — RouteRois folds the crop's own translation into
    // inv_align at construction time (see its own comment), so nothing here
    // adds ref.src_box a second time.
    const cv::Matx23f& m = ref.inv_align;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            const float lx = xs[i];
            const float ly = ys[j];
            corners.push_back(cv::Point2f(m(0, 0) * lx + m(0, 1) * ly + m(0, 2),
                                          m(1, 0) * lx + m(1, 1) * ly + m(1, 2)));
        }
    }
    return corners;
}

cv::Rect2f RestoreBox(const cv::Rect2f& roi_local, const RoiRef& ref) {
    // Bit-exact for a full-frame result on the source frame; a result from
    // a handed-off image carries its scale in inv_align and is mapped below.
    if (!ref.from_roi && IsIdentityAffine(ref.inv_align)) return roi_local;

    // A box under an affine (ref.inv_align) is in general a quadrilateral,
    // not a box, so this restores the axis-aligned bounding box of the four
    // transformed corners. inv_align defaults to identity (RoiRef's own
    // constructor), so a plain crop's four corners transform to themselves
    // (plus RouteRois's own translation, folded into inv_align) and this
    // reduces exactly to the previous translation-only behaviour.
    const std::vector<cv::Point2f> corners = RestoreBoxCorners(roi_local, ref);
    float min_x = corners[0].x;
    float min_y = corners[0].y;
    float max_x = corners[0].x;
    float max_y = corners[0].y;
    for (std::size_t i = 1; i < corners.size(); ++i) {
        min_x = std::min(min_x, corners[i].x);
        max_x = std::max(max_x, corners[i].x);
        min_y = std::min(min_y, corners[i].y);
        max_y = std::max(max_y, corners[i].y);
    }
    return cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}

Keypoint RestorePoint(const Keypoint& roi_local, const RoiRef& ref) {
    if (!ref.from_roi && IsIdentityAffine(ref.inv_align)) return roi_local;
    // ref.inv_align alone maps local -> source for every crop kind and for a
    // handed-off image. For a plain crop (inv_align = translate(src_box))
    // (1*x + 0*y) + tx is bit-identical to the old x + src_box.x.
    const cv::Matx23f& m = ref.inv_align;
    Keypoint restored = roi_local;
    restored.x = (m(0, 0) * roi_local.x + m(0, 1) * roi_local.y) + m(0, 2);
    restored.y = (m(1, 0) * roi_local.x + m(1, 1) * roi_local.y) + m(1, 2);
    return restored;
}

}  // namespace graph
}  // namespace dxapp

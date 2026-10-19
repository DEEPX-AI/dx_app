#include "common/graph/shape.hpp"

#include "common/graph/graph_error.hpp"

namespace dxapp {
namespace graph {

const char* ToString(Shape shape) {
    switch (shape) {
        case Shape::kFrame:     return "frame";
        case Shape::kBoxes:     return "boxes";
        case Shape::kObBoxes:   return "obboxes";
        case Shape::kInstances: return "instances";
        case Shape::kKeypoints: return "keypoints";
        case Shape::kLabelMap:  return "labelmap";
        case Shape::kDenseMap:  return "densemap";
        case Shape::kImage:     return "image";
        case Shape::kScores:    return "scores";
        case Shape::kVector:    return "vector";
        case Shape::kBoxes3d:   return "boxes3d";
        case Shape::kRecords:   return "records";
    }
    return "unknown";
}

bool ProducesRoi(Shape shape) {
    return shape == Shape::kBoxes ||
           shape == Shape::kObBoxes ||
           shape == Shape::kInstances;
}

const char* ToString(InputContract contract) {
    switch (contract) {
        case InputContract::kFullFrame: return "full_frame";
        case InputContract::kRoi:       return "roi";
        case InputContract::kEither:    return "either";
    }
    return "unknown";
}

bool AcceptsRoi(InputContract contract) {
    return contract == InputContract::kRoi || contract == InputContract::kEither;
}

const char* ToString(GraphErrorCode code) {
    switch (code) {
        case GraphErrorCode::kGraphVersion:  return "GRAPH_VERSION";
        case GraphErrorCode::kGraphSchema:   return "GRAPH_SCHEMA";
        case GraphErrorCode::kGraphEdge:     return "GRAPH_EDGE";
        case GraphErrorCode::kGraphAlign:    return "GRAPH_ALIGN";
        case GraphErrorCode::kGraphCycle:    return "GRAPH_CYCLE";
        case GraphErrorCode::kGraphOrphan:   return "GRAPH_ORPHAN";
        case GraphErrorCode::kGraphReserved: return "GRAPH_RESERVED";
        case GraphErrorCode::kModelUnknown:  return "MODEL_UNKNOWN";
        case GraphErrorCode::kModelNoTask:   return "MODEL_NO_TASK";
        case GraphErrorCode::kModelNotReady: return "MODEL_NOT_READY";
        case GraphErrorCode::kModelMissing:  return "MODEL_MISSING";
        case GraphErrorCode::kModelLoad:     return "MODEL_LOAD";
    }
    return "UNKNOWN";
}

}  // namespace graph
}  // namespace dxapp

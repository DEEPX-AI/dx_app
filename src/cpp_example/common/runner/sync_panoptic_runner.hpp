/**
 * @file sync_panoptic_runner.hpp
 * @brief Synchronous panoptic driving perception runner (e.g. YOLOPv2)
 */

#ifndef SYNC_PANOPTIC_RUNNER_HPP
#define SYNC_PANOPTIC_RUNNER_HPP

#include "sync_detection_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using SyncPanopticRunner = SyncDetectionRunner<FactoryT>;

}  // namespace dxapp

#endif  // SYNC_PANOPTIC_RUNNER_HPP

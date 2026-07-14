/**
 * @file sync_keypoint_runner.hpp
 * @brief Synchronous keypoint detection runner (e.g. SuperPoint)
 */

#ifndef SYNC_KEYPOINT_RUNNER_HPP
#define SYNC_KEYPOINT_RUNNER_HPP

#include "sync_pose_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using SyncKeypointRunner = SyncPoseRunner<FactoryT>;

}  // namespace dxapp

#endif  // SYNC_KEYPOINT_RUNNER_HPP

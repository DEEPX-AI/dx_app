/**
 * @file sync_object_pose_runner.hpp
 * @brief Synchronous object pose estimation runner (e.g. DOPE)
 */

#ifndef SYNC_OBJECT_POSE_RUNNER_HPP
#define SYNC_OBJECT_POSE_RUNNER_HPP

#include "sync_pose_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using SyncObjectPoseRunner = SyncPoseRunner<FactoryT>;

}  // namespace dxapp

#endif  // SYNC_OBJECT_POSE_RUNNER_HPP

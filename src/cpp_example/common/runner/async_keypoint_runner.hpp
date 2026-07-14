/**
 * @file async_keypoint_runner.hpp
 * @brief Asynchronous keypoint detection runner (e.g. SuperPoint)
 */

#ifndef ASYNC_KEYPOINT_RUNNER_HPP
#define ASYNC_KEYPOINT_RUNNER_HPP

#include "async_pose_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using AsyncKeypointRunner = AsyncPoseRunner<FactoryT>;

}  // namespace dxapp

#endif  // ASYNC_KEYPOINT_RUNNER_HPP

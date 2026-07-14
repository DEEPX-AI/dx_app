/**
 * @file async_object_pose_runner.hpp
 * @brief Asynchronous object pose estimation runner (e.g. DOPE)
 */

#ifndef ASYNC_OBJECT_POSE_RUNNER_HPP
#define ASYNC_OBJECT_POSE_RUNNER_HPP

#include "async_pose_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using AsyncObjectPoseRunner = AsyncPoseRunner<FactoryT>;

}  // namespace dxapp

#endif  // ASYNC_OBJECT_POSE_RUNNER_HPP

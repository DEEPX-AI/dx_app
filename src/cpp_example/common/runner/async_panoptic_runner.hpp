/**
 * @file async_panoptic_runner.hpp
 * @brief Asynchronous panoptic driving perception runner (e.g. YOLOPv2)
 */

#ifndef ASYNC_PANOPTIC_RUNNER_HPP
#define ASYNC_PANOPTIC_RUNNER_HPP

#include "async_detection_runner.hpp"

namespace dxapp {

template <typename FactoryT>
using AsyncPanopticRunner = AsyncDetectionRunner<FactoryT>;

}  // namespace dxapp

#endif  // ASYNC_PANOPTIC_RUNNER_HPP

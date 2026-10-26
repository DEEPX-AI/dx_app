/**
 * @file yolov3-gluon_416x416
 * @brief yolov3 sync inference -- one entry point for the whole family.
 *
 * The variant is the .dxnn stem, so the model path alone identifies it and no
 * --variant flag or runner change is needed.
 */

#include "factory/yolov3-gluon_416x416_factory.hpp"
#include "common/runner/sync_detection_runner.hpp"

int main(int argc, char* argv[]) {
    auto factory = std::make_unique<dxapp::v_yolov3_gluon_416x416::Yolov3Factory>();
    dxapp::SyncDetectionRunner<dxapp::v_yolov3_gluon_416x416::Yolov3Factory> runner(std::move(factory));
    return runner.run(argc, argv);
}

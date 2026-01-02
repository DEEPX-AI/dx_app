/*
  camera_v4l2_test.cpp
  Simple example showing low-level V4L2 mmap capture and feeding frames to DXRT
*/

#include <dxrt/dxrt_api.h>

#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// v4l2 low level headers
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>

// Minimal mmap-based V4L2 capture helper (MJPEG / YUYV -> BGR cv::Mat)
class V4L2Capture {
   public:
    struct Buffer {
        void* start;
        size_t length;
    };
    V4L2Capture() : fd_(-1), width_(0), height_(0), pixfmt_(0) {}
    ~V4L2Capture() { closeDevice(); }

    bool openDevice(const std::string& dev, uint32_t width, uint32_t height) {
        device_ = dev;
        width_ = width;
        height_ = height;
        fd_ = ::open(device_.c_str(), O_RDWR | O_NONBLOCK, 0);
        if (fd_ < 0) {
            std::cerr << "[V4L2] open failed: " << strerror(errno) << std::endl;
            return false;
        }

        struct v4l2_format fmt;
        memset(&fmt, 0, sizeof(fmt));
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width_;
        fmt.fmt.pix.height = height_;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
        if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
            fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
                std::cerr << "[V4L2] S_FMT failed: " << strerror(errno) << std::endl;
                closeDevice();
                return false;
            }
        }
        pixfmt_ = fmt.fmt.pix.pixelformat;

        struct v4l2_requestbuffers req;
        memset(&req, 0, sizeof(req));
        req.count = 4;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
            std::cerr << "[V4L2] REQBUFS failed: " << strerror(errno) << std::endl;
            closeDevice();
            return false;
        }

        buffers_.resize(req.count);
        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
                std::cerr << "[V4L2] QUERYBUF failed: " << strerror(errno) << std::endl;
                closeDevice();
                return false;
            }
            buffers_[i].length = buf.length;
            buffers_[i].start =
                mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
            if (buffers_[i].start == MAP_FAILED) {
                std::cerr << "[V4L2] mmap failed: " << strerror(errno) << std::endl;
                closeDevice();
                return false;
            }
        }

        for (size_t i = 0; i < buffers_.size(); ++i) {
            struct v4l2_buffer buf;
            memset(&buf, 0, sizeof(buf));
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;
            buf.index = i;
            if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
                std::cerr << "[V4L2] QBUF failed: " << strerror(errno) << std::endl;
                closeDevice();
                return false;
            }
        }

        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            std::cerr << "[V4L2] STREAMON failed: " << strerror(errno) << std::endl;
            closeDevice();
            return false;
        }
        return true;
    }

    void closeDevice() {
        if (fd_ < 0) return;
        enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        for (auto& b : buffers_)
            if (b.start && b.length) munmap(b.start, b.length);
        buffers_.clear();
        ::close(fd_);
        fd_ = -1;
    }

    bool readFrame(cv::Mat& out) {
        if (fd_ < 0) return false;
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
            if (errno == EAGAIN) return false;
            std::cerr << "[V4L2] DQBUF failed:" << strerror(errno) << std::endl;
            return false;
        }
        uint8_t* data = reinterpret_cast<uint8_t*>(buffers_[buf.index].start);
        size_t len = buf.bytesused ? buf.bytesused : buffers_[buf.index].length;
        if (pixfmt_ == V4L2_PIX_FMT_MJPEG) {
            std::cout << "this v4l2 pixfmt is V4L2_PIX_FMT_MJPEG" << std::endl;
            std::vector<uint8_t> blob(data, data + len);
            out = cv::imdecode(blob, cv::IMREAD_COLOR);
        } else if (pixfmt_ == V4L2_PIX_FMT_YUYV) {
            std::cout << "this v4l2 pixfmt is V4L2_PIX_FMT_YUYV" << std::endl;
            cv::Mat yuyv(height_, width_, CV_8UC2, data);
            cv::cvtColor(yuyv, out, cv::COLOR_YUV2BGR_YUYV);
        } else {
            out = cv::Mat(height_, width_, CV_8UC3, data).clone();
        }
        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            std::cerr << "[V4L2] requeue failed:" << strerror(errno) << std::endl;
        }
        return true;
    }

   private:
    std::string device_;
    int fd_;
    uint32_t width_, height_;
    uint32_t pixfmt_;
    std::vector<Buffer> buffers_;
};

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    uint32_t input_w = 1920, input_h = 1080;
    std::string dev = "/dev/video0";

    cxxopts::Options options("camera_v4l2_test", "v4l2 camera test for DXRT");
    options.add_options()("m,model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "width", "model input width", cxxopts::value<uint32_t>(input_w)->default_value("640"))(
        "height", "model input height", cxxopts::value<uint32_t>(input_h)->default_value("640"))(
        "d,device", "v4l2 device", cxxopts::value<std::string>(dev)->default_value(dev))(
        "h,help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "e.g. ./camera_v4l2 -m model.dxnn -d /dev/video0 --width 640 --height 640"
                  << std::endl;
        return 0;
    }
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path required (-m)" << std::endl;
        return 1;
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);
    if (!dxapp::common::minversionforRTandCompiler(&ie)) {
        std::cerr << "[DXAPP] model/runtime version mismatch" << std::endl;
        return -1;
    }

    std::mutex g_lock;
    std::queue<uint8_t> keyQueue;
    int processCount = 0;
    bool appQuitLocal = false;

    std::function<int(void)> postprocessingThread = [&]() -> int {
        while (keyQueue.size() < 1) std::this_thread::sleep_for(std::chrono::microseconds(10));
        while (!appQuitLocal) {
            std::unique_lock<std::mutex> lk(g_lock);
            if (keyQueue.empty()) {
                lk.unlock();
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                continue;
            }
            auto outputs = ie.Wait(keyQueue.front());
            std::cout << "[DXAPP] [INFO] post processing result: " << outputs.size() << " items"
                      << std::endl;
            keyQueue.pop();
            processCount++;
        }
        return 0;
    };
    std::thread postThread(postprocessingThread);

    V4L2Capture cap;
    if (!cap.openDevice(dev, input_w, input_h)) {
        std::cerr << "[ERROR] open device failed: " << dev << std::endl;
        return -1;
    }
    cv::Mat frame;
    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& t : inputTensors) t = std::vector<uint8_t>(ie.GetInputSize());
    int idx = 0;
    auto s = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Starting V4L2 camera processing. Press 'q' or ESC key to quit."
              << std::endl;

    while (true) {
        if (!cap.readFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (frame.empty()) {
            std::cerr << "[ERROR] empty frame" << std::endl;
            break;
        }
        cv::Mat resized(input_h, input_w, CV_8UC3, inputTensors[idx].data());
        cv::resize(frame, resized, cv::Size(input_w, input_h), cv::INTER_LINEAR);
        keyQueue.push(ie.RunAsync(resized.data));
        idx = (idx + 1) % inputTensors.size();
        cv::imshow("v4l2", frame);
        auto key = cv::waitKey(1);
        if (key == 27 || key == 'q') {  // ESC key
            std::cout << "[INFO] Processing stopped by user (ESC key)" << std::endl;
            break;
        }
    }
    cap.closeDevice();
    cv::destroyAllWindows();

    // Calculate and display performance statistics
    auto e = std::chrono::high_resolution_clock::now();
    auto total_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
    double total_time_sec = total_time_ms / 1000.0;
    double fps_actual = processCount / total_time_sec;

    std::cout << "[INFO] Stopping V4L2 camera processing..." << std::endl;
    std::cout << "[INFO] Total processing time: " << total_time_sec << " seconds" << std::endl;
    std::cout << "[INFO] Processed " << processCount << " frames" << std::endl;
    std::cout << "[INFO] Actual FPS: " << fps_actual << std::endl;

    appQuitLocal = true;
    postThread.join();
    DXRT_TRY_CATCH_END
    return 0;
}
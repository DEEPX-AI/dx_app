/*
  camera_v4l2_v3.cpp
  Simple example showing low-level V4L2 mmap capture and feeding frames to DXRT
  V3 : standalone Inference module
*/

#include <dxrt/dxrt_api.h>

#include <chrono>
#include <common_util.hpp>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// v4l2 low level headers
#include <DX_VENC.h>
#include <Logger.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstring>

// Global atomic variables for thread synchronization
std::atomic<bool> stop_threads{false};
std::atomic<bool> input_finished{false};
std::atomic<bool> output_finished{false};
std::atomic<size_t> frames_fed{0};
std::atomic<size_t> frames_encoded{0};
std::atomic<size_t> frames_dumped{0};

// Dump queue for encoded frames
std::queue<DX_VENC::EncodedStream> dump_queue;
std::mutex dump_mutex;
std::condition_variable dump_cv;
const size_t MAX_DUMP_QUEUE_SIZE = 30;
const size_t TOTAL_FRAMES_TO_FEED = 300;

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

        // Try different pixel formats in order of preference (MJPEG first for Brio 500)
        std::vector<uint32_t> formats = {V4L2_PIX_FMT_MJPEG, V4L2_PIX_FMT_YUYV, V4L2_PIX_FMT_NV12};
        bool format_set = false;

        for (uint32_t format : formats) {
            fmt.fmt.pix.pixelformat = format;
            if (ioctl(fd_, VIDIOC_S_FMT, &fmt) >= 0) {
                pixfmt_ = fmt.fmt.pix.pixelformat;
                format_set = true;
                std::cout << "[V4L2] Using pixel format: 0x" << std::hex << pixfmt_ << std::dec
                          << std::endl;
                break;
            }
        }

        if (!format_set) {
            std::cerr << "[V4L2] S_FMT failed for all formats: " << strerror(errno) << std::endl;
            closeDevice();
            return false;
        }

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
            std::vector<uint8_t> blob(data, data + len);
            out = cv::imdecode(blob, cv::IMREAD_COLOR);
        } else if (pixfmt_ == V4L2_PIX_FMT_YUYV) {
            cv::Mat yuyv(height_, width_, CV_8UC2, data);
            cv::cvtColor(yuyv, out, cv::COLOR_YUV2BGR_YUYV);
        } else if (pixfmt_ == V4L2_PIX_FMT_NV12) {
            // NV12 format: Y plane + interleaved UV plane
            cv::Mat y_plane(height_, width_, CV_8UC1, data);
            cv::Mat uv_plane(height_ / 2, width_ / 2, CV_8UC2, data + width_ * height_);
            cv::Mat yuv_i420(height_ * 3 / 2, width_, CV_8UC1);

            // Copy Y plane
            y_plane.copyTo(yuv_i420(cv::Rect(0, 0, width_, height_)));

            // Convert NV12 UV to I420 U and V planes
            cv::Mat u_plane = yuv_i420(cv::Rect(0, height_, width_ / 2, height_ / 4));
            cv::Mat v_plane = yuv_i420(cv::Rect(width_ / 2, height_, width_ / 2, height_ / 4));

            for (uint32_t i = 0; i < height_ / 2; i++) {
                for (uint32_t j = 0; j < width_ / 2; j++) {
                    cv::Vec2b uv = uv_plane.at<cv::Vec2b>(i, j);
                    u_plane.at<uint8_t>(i, j) = uv[0];
                    v_plane.at<uint8_t>(i, j) = uv[1];
                }
            }

            cv::cvtColor(yuv_i420, out, cv::COLOR_YUV2BGR_I420);
        } else {
            // Default: assume BGR format
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

/**
 * @brief Print CMA (Contiguous Memory Allocator) information
 */
void printCmaInfo() {
    std::cout << "=== CMA Memory Information ===" << std::endl;

    // Read CMA info from /proc/meminfo
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("CmaTotal:") != std::string::npos ||
            line.find("CmaFree:") != std::string::npos) {
            std::cout << line << std::endl;
        }
    }
    std::cout << "===============================" << std::endl;
}

/**
 * @brief Thread function for feeding camera frames to the encoder.
 *
 * @param encoder The DX_VENC instance.
 * @param capture_device The V4L2 capture device path.
 */
void FeedingThread(DX_VENC& encoder, const std::string& capture_device) {
    std::cout << "[INPUT] Feeding thread started" << std::endl;

    try {
        // Use device native resolution (Brio 500: 640x360)
        const size_t capture_width = 640;
        const size_t capture_height = 360;

        // Encoder dimensions (can be different from capture)3
        const size_t encoder_width = 640;
        const size_t encoder_height = 360;
        const size_t y_size_expected = encoder_width * encoder_height;
        const size_t uv_size_expected = encoder_width * encoder_height / 2;
        size_t frames_count = 0;

        // Open V4L2 capture device with native resolution
        V4L2Capture cap;
        if (!cap.openDevice(capture_device, capture_width, capture_height)) {
            std::cerr << "[INPUT] Failed to open capture device: " << capture_device << std::endl;
            stop_threads = true;
            input_finished = true;
            return;
        }
        std::cout << "[INPUT] V4L2 capture device opened: " << capture_device << std::endl;

        cv::Mat frame;
        std::vector<uint8_t> nv12_frame(encoder_width * encoder_height * 3 / 2);

        while (!stop_threads.load() && frames_count < TOTAL_FRAMES_TO_FEED) {
            // Capture frame from camera
            if (!cap.readFrame(frame)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            if (frame.empty()) {
                std::cerr << "[INPUT] Empty frame captured" << std::endl;
                continue;
            }

            // Resize frame to encoder dimensions if needed
            if (static_cast<size_t>(frame.cols) != encoder_width ||
                static_cast<size_t>(frame.rows) != encoder_height) {
                cv::Mat resized;
                cv::resize(
                    frame, resized,
                    cv::Size(static_cast<int>(encoder_width), static_cast<int>(encoder_height)),
                    cv::INTER_LINEAR);
                frame = resized;
            }

            // Convert BGR to NV12
            cv::Mat yuv;
            cv::cvtColor(frame, yuv, cv::COLOR_BGR2YUV_I420);

            // Convert I420 to NV12 format
            uint8_t* y_ptr = nv12_frame.data();
            uint8_t* uv_ptr = nv12_frame.data() + y_size_expected;

            // Copy Y plane
            memcpy(y_ptr, yuv.data, y_size_expected);

            // Interleave U and V planes for NV12
            uint8_t* u_src = yuv.data + y_size_expected;
            uint8_t* v_src = yuv.data + y_size_expected + y_size_expected / 4;
            for (size_t i = 0; i < y_size_expected / 4; ++i) {
                uv_ptr[i * 2] = u_src[i];
                uv_ptr[i * 2 + 1] = v_src[i];
            }

            bool ok = encoder.putframe(y_ptr, uv_ptr, static_cast<uint32_t>(y_size_expected),
                                       static_cast<uint32_t>(uv_size_expected),
                                       static_cast<uint32_t>(encoder_width),
                                       static_cast<uint32_t>(encoder_height), frames_count, false);
            if (ok) {
                frames_count++;
                frames_fed++;
                if (frames_count % (TOTAL_FRAMES_TO_FEED / 10) == 0) {
                    std::cout << "[INPUT] Fed " << frames_count << "/" << TOTAL_FRAMES_TO_FEED
                              << " frames (" << (frames_count * 100 / TOTAL_FRAMES_TO_FEED) << "%)"
                              << std::endl;
                }
            } else {
                std::cerr << "[INPUT] putframe() failed: " << encoder.getLastError() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            // simulate ~60fps
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cap.closeDevice();
        std::cout << "[INPUT] Sending EOS after " << frames_count << " frames" << std::endl;

        const int max_wait_ms = 10000;
        int waited_ms = 0;
        while (!stop_threads.load()) {
            // EOS: width,height 올바르게 전달
            if (encoder.putframe(nullptr, nullptr, 0, 0, static_cast<uint32_t>(encoder_width),
                                 static_cast<uint32_t>(encoder_height), 0, true)) {
                std::cout << "[INPUT] EOS signal sent successfully" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            waited_ms += 10;
            if (waited_ms >= max_wait_ms) {
                std::cerr << "[INPUT] Warning: EOS signal not accepted after " << max_wait_ms
                          << " ms" << std::endl;
                break;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[INPUT] Exception: " << e.what() << std::endl;
        stop_threads = true;
    }
    input_finished = true;
    std::cout << "[INPUT] Input thread finished, frames fed: " << frames_fed.load() << std::endl;
}

/**
 * @brief Thread function for getting encoded frames from the encoder.
 *
 * @param encoder The DX_VENC instance.
 * @param enable_h264_dump Flag to enable H.264 dumping.
 */
void GetStreamThread(DX_VENC& encoder, bool enable_h264_dump) {
    std::cout << "[OUTPUT] Output thread started" << std::endl;

    while (!stop_threads.load()) {
        DX_VENC::EncodedStream frame;
        if (!encoder.getstream(frame, -1)) {
            if (input_finished.load() && encoder.hasError()) {
                std::cout << "[OUTPUT] No more frames and input finished" << std::endl;
                break;
            }
            if (encoder.hasError()) {
                std::cerr << "[OUTPUT] Encoder error: " << encoder.getLastError() << std::endl;
                stop_threads = true;
                break;
            }
            std::cout << "[OUTPUT] No frame available, retrying..." << std::endl;
            continue;
        }

        if (!frame.data) {
            continue;
        }

        if (frame.is_eos) {
            std::cout << "[OUTPUT] End of stream reached" << std::endl;
            encoder.releasestream(frame);
            break;
        }

        if (enable_h264_dump && !frame.is_eos) {
            std::unique_lock<std::mutex> lock(dump_mutex);
            if (dump_queue.size() < MAX_DUMP_QUEUE_SIZE) {
                dump_queue.push(std::move(frame));
                dump_cv.notify_one();
            } else {
                std::cout << "[OUTPUT] Warning: Dump queue full, dropping frame" << std::endl;
            }
        } else {
            encoder.releasestream(frame);
        }
        frames_encoded++;
    }

    output_finished = true;
    dump_cv.notify_all();
    std::cout << "[OUTPUT] Output thread finished, frames encoded: " << frames_encoded.load()
              << std::endl;
}

/**
 * @brief Thread function for dumping encoded frames to a file.
 *
 * @param encoder The DX_VENC instance.
 * @param output_stream The path to the output file.
 */
void fileDumpThread(DX_VENC& encoder, const std::string& output_stream) {
    std::cout << "[DUMP] File dump thread started" << std::endl;
    std::ofstream output_file(output_stream, std::ios::binary);

    if (!output_file.is_open()) {
        std::cerr << "[DUMP] Failed to open output file: " << output_stream << std::endl;
        stop_threads = true;
        return;
    }

    while (!stop_threads.load()) {
        std::unique_lock<std::mutex> lock(dump_mutex);

        dump_cv.wait(lock, [&] {
            return !dump_queue.empty() || output_finished.load() || stop_threads.load();
        });

        while (!dump_queue.empty()) {
            DX_VENC::EncodedStream frame = std::move(dump_queue.front());
            dump_queue.pop();
            lock.unlock();

            // Dump all encoded frames
            output_file.write(reinterpret_cast<const char*>(frame.data), frame.size);

            frames_dumped++;
            std::cout << "Dumped encoded frame: " << frame.size
                      << " bytes, frame count: " << frames_dumped.load() << std::endl;
            encoder.releasestream(frame);
            if (frame.is_eos) {
                goto dump_finished;
            }
            lock.lock();
        }

        if (output_finished.load() && dump_queue.empty()) {
            break;
        }
    }

dump_finished:
    output_file.close();
    std::cout << "[DUMP] File dump thread finished, frames dumped: " << frames_dumped.load()
              << std::endl;
}

int main(int argc, char* argv[]) {
    DXRT_TRY_CATCH_BEGIN
    std::string modelPath = "";
    uint32_t input_w = 1920, input_h = 1080;
    std::string dev = "/dev/video0";
    std::string capture_dev = "/dev/video2";
    std::string output_video = "bs_output.h264";
    uint32_t max_frames = 300;  // Default: 300 frames (10 seconds at 30fps)

    cxxopts::Options options("camera_v4l2_v3", "v4l2 camera video recording for DXRT");
    options.add_options()("m,model_path", "sample model file (.dxnn, required)",
                          cxxopts::value<std::string>(modelPath))(
        "width", "model input width", cxxopts::value<uint32_t>(input_w)->default_value("640"))(
        "height", "model input height", cxxopts::value<uint32_t>(input_h)->default_value("640"))(
        "d,device", "v4l2 device", cxxopts::value<std::string>(dev)->default_value(dev))(
        "c,capture", "v4l2 capture device for frames",
        cxxopts::value<std::string>(capture_dev)->default_value("/dev/video2"))(
        "o,output", "output video file path (e.g. bs_output.h264)",
        cxxopts::value<std::string>(output_video)->default_value(""))(
        "frames", "maximum frames to record",
        cxxopts::value<uint32_t>(max_frames)->default_value("300"))("h,help", "print usage");

    auto cmd = options.parse(argc, argv);
    if (cmd.count("help")) {
        std::cout << options.help() << std::endl;
        std::cout << "e.g. ./camera_v4l2_v3 -m model.dxnn -d /dev/video0 -c /dev/video2 --width "
                     "640 --height 480 "
                     "--frames 300 --output bs_output.h264"
                  << std::endl;
        return 0;
    }
    if (modelPath.empty()) {
        std::cerr << "[ERROR] Model path required (-m)" << std::endl;
        return 1;
    }

    dxrt::InferenceOption io;
    dxrt::InferenceEngine ie(modelPath, io);

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
            keyQueue.pop();
            std::cout << "[DXAPP] [INFO] post processing result: " << outputs.size() << " items"
                      << std::endl;
            processCount++;
        }
        return 0;
    };
    std::thread postThread(postprocessingThread);

    // Set logger to debug level for detailed output
    Logger::setLevel(Logger::Level::DEBUG);
    Logger::enableTimestamp(true);
    printCmaInfo();

    DX_VENC::IOMode io_mode = DX_VENC::stringToIOMode("auto");

    try {
        DX_VENC encoder(io_mode);

        //  if (enable_h264_dump && io_mode == DX_VENC::IOMode::DMABUF) {
        //     encoder.enableDmabufMmap(true);
        //     std::cout << "DMABUF mmap enabled for H.264 dump" << std::endl;
        // }

        if (!encoder.open(dev)) {
            std::cerr << "Failed to open encoder: " << encoder.getLastError() << std::endl;
            return 1;
        }

        int width = 640;
        int height = 360;
        int frame_rate = 30;      // Match camera's native framerate
        int bitrate_kbps = 4096;  // 4 Mbps (appropriate for 640x360)
        int gop_size = 30;

        if (!encoder.prepare(width, height, DX_VENC::InputColorFormat::NV12,
                             DX_VENC::OutputCodecFormat::H264, frame_rate, bitrate_kbps,
                             gop_size)) {
            std::cerr << "Failed to prepare encoder: " << encoder.getLastError() << std::endl;
            return 1;
        }
        if (!encoder.startStreaming()) {
            std::cerr << "Failed to start streaming: " << encoder.getLastError() << std::endl;
            return 1;
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        std::thread input_th(FeedingThread, std::ref(encoder), capture_dev);
        std::thread output_th(GetStreamThread, std::ref(encoder),
                              output_video.empty() ? false : true);
        std::thread dump_th;

        if (output_video.empty() ? false : true) {
            dump_th = std::thread(fileDumpThread, std::ref(encoder), output_video);
        }

        input_th.join();
        output_th.join();
        if (output_video.empty() ? false : true && dump_th.joinable()) {
            dump_th.join();
        }
        encoder.close();
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << std::endl << "=== Threaded Encoding Complete ===" << std::endl;
        std::cout << "Frames fed: " << frames_fed.load() << std::endl;
        std::cout << "Frames encoded: " << frames_encoded.load() << std::endl;
        if (output_video.empty() ? false : true) {
            std::cout << "Frames dumped: " << frames_dumped.load() << std::endl;
        }
        std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;
        if (duration.count() > 0) {
            std::cout << "Encoding FPS: " << (frames_encoded.load() * 1000 / duration.count())
                      << std::endl;
        }
        if (output_video.empty() ? false : true) {
            std::cout << "Output file: " << output_video << std::endl;
        } else {
            std::cout << "Mode: Encode-only (no file output)" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        stop_threads = true;
        return 1;
    }

    /*
    V4L2Capture cap;
    if (!cap.openDevice(dev, input_w, input_h)) {
        std::cerr << "[ERROR] open device failed: " << dev << std::endl;
        return -1;
    }

    // Initialize video writer
    cv::VideoWriter video_writer;
    cv::Mat frame;
    bool video_writer_initialized = false;

    std::vector<std::vector<uint8_t>> inputTensors(10);
    for (auto& t : inputTensors) t = std::vector<uint8_t>(ie.GetInputSize());
    int idx = 0;
    uint32_t frame_count = 0;
    auto s = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Starting V4L2 camera processing and video recording..." << std::endl;
    std::cout << "[INFO] Will record " << max_frames << " frames to: " << output_video << std::endl;

    while (frame_count < max_frames) {
        if (!cap.readFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        if (frame.empty()) {
            std::cerr << "[ERROR] empty frame" << std::endl;
            break;
        }

        // Initialize video writer with actual frame size
        if (!video_writer_initialized) {
            cv::Size frame_size(frame.cols, frame.rows);
            video_writer.open(output_video, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
                              frame_size);
            if (!video_writer.isOpened()) {
                std::cerr << "[ERROR] Could not open video writer: " << output_video << std::endl;
                break;
            }
            video_writer_initialized = true;
            std::cout << "[INFO] Video writer initialized with size: " << frame_size << " @ " << fps
                      << " fps" << std::endl;
        }

        // Save frame to video
        video_writer.write(frame);

        // Process frame for inference
        cv::Mat resized(input_h, input_w, CV_8UC3, inputTensors[idx].data());
        cv::resize(frame, resized, cv::Size(input_w, input_h), cv::INTER_LINEAR);
        keyQueue.push(ie.RunAsync(resized.data));
        idx = (idx + 1) % inputTensors.size();

        frame_count++;

        // Print progress every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "[INFO] Recorded " << frame_count << "/" << max_frames << " frames"
                      << std::endl;
        }
    }

    std::cout << "[INFO] Recording completed. Total frames: " << frame_count << std::endl;
    cap.closeDevice();

    // Release video writer
    if (video_writer.isOpened()) {
        video_writer.release();
        std::cout << "[INFO] Video saved to: " << output_video << std::endl;
    }

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
    */
    DXRT_TRY_CATCH_END
    return 0;
}
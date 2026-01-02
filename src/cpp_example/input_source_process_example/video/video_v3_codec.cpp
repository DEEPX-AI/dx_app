#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>

#include "DX_VDEC.h"

// Global synchronization objects
std::atomic<bool> stop_threads{false};
std::atomic<bool> input_finished{false};
std::atomic<bool> output_finished{false};
std::atomic<int> frames_fed{0};
std::atomic<int> frames_decoded{0};
std::atomic<int> frames_dumped{0};

// File dump queue
std::queue<DX_VDEC::DecodedFrame> dump_queue;
std::mutex dump_mutex;
std::condition_variable dump_cv;
const size_t MAX_DUMP_QUEUE_SIZE = 30;

void printCmaInfo() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        std::cerr << "Error: Could not open /proc/meminfo" << std::endl;
        return;
    }

    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("CmaTotal:") == 0 || line.find("CmaFree:") == 0) {
            std::cout << line << std::endl;
        }
    }
}

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input FILE                Input video stream file" << std::endl;
    std::cout << "  -o, --output FILE               Output YUV dump file" << std::endl;
    std::cout << "  -d, --dump-yuv                  Enable YUV frame dump to file" << std::endl;
    std::cout << "  --capture-io-mode MODE          Set capture I/O mode" << std::endl;
    std::cout << "                                  MODE:" << std::endl;
    std::cout << "                                    auto         (auto-select, uses mmap)"
              << std::endl;
    std::cout << "                                    mmap         (traditional MMAP mode)"
              << std::endl;
    std::cout << "                                    dmabuf       (DMABUF export, zero-copy)"
              << std::endl;
    std::cout << "                                    dmabuf-import (not implemented)" << std::endl;
    std::cout << "  -h, --help                      Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name
              << " --input test.h264                           # Decode only" << std::endl;
    std::cout << "  " << program_name
              << " --input test.h264 --output out.yuv --dump-yuv # Decode and dump" << std::endl;
    std::cout << "  " << program_name
              << " --capture-io-mode dmabuf --input test.h264  # DMABUF mode" << std::endl;
}

/**
 * @brief Thread function for getting decoded frames from the decoder.
 *
 * @param decoder The DX_VDEC instance.
 * @param enable_yuv_dump Flag to enable YUV dumping.
 */
void GetFrameThread(DX_VDEC& decoder, bool enable_yuv_dump) {
    std::cout << "[OUTPUT] Output thread started" << std::endl;

    while (!stop_threads.load()) {
        DX_VDEC::DecodedFrame frame;
        if (!decoder.getframe(frame, 10)) {
            if (input_finished.load() && decoder.hasError()) {
                std::cout << "[OUTPUT] No more frames and input finished" << std::endl;
                break;
            }
            if (decoder.hasError()) {
                std::cerr << "[OUTPUT] Decoder error: " << decoder.getLastError() << std::endl;
                stop_threads = true;
                break;
            }
            continue;
        }

        if (!(frame.y_data && frame.uv_data)) {
            continue;
        }

        if (frame.is_eos) {
            std::cout << "[OUTPUT] End of stream reached" << std::endl;
            decoder.releaseframe(frame);
            break;
        }

        if (enable_yuv_dump && !frame.is_eos) {
            std::unique_lock<std::mutex> lock(dump_mutex);
            if (dump_queue.size() < MAX_DUMP_QUEUE_SIZE) {
                dump_queue.push(std::move(frame));
                dump_cv.notify_one();
            } else {
                std::cout << "[OUTPUT] Warning: Dump queue full, dropping frame" << std::endl;
            }
        } else {
            decoder.releaseframe(frame);
        }
        frames_decoded++;
    }

    output_finished = true;
    dump_cv.notify_all();
    std::cout << "[OUTPUT] Output thread finished, frames decoded: " << frames_decoded.load()
              << std::endl;
}

/**
 * @brief Thread function for dumping decoded frames to a file.
 *
 * @param decoder The DX_VDEC instance.
 * @param output_frame The path to the output file.
 */
void fileDumpThread(DX_VDEC& decoder, const std::string& output_frame) {
    std::cout << "[DUMP] File dump thread started" << std::endl;
    std::ofstream output_file(output_frame, std::ios::binary);

    if (!output_file.is_open()) {
        std::cerr << "[DUMP] Failed to open output file: " << output_frame << std::endl;
        stop_threads = true;
        return;
    }

    while (!stop_threads.load()) {
        std::unique_lock<std::mutex> lock(dump_mutex);

        dump_cv.wait(lock, [&] {
            return !dump_queue.empty() || output_finished.load() || stop_threads.load();
        });

        while (!dump_queue.empty()) {
            DX_VDEC::DecodedFrame frame = std::move(dump_queue.front());
            dump_queue.pop();
            lock.unlock();
            /* dump every 10-frame */
            if (frames_decoded % 30 == 0) {
                size_t y_size = frame.stride_y * frame.height;
                size_t uv_size = frame.stride_uv * (frame.height / 2);
                output_file.write(reinterpret_cast<const char*>(frame.y_data), y_size);
                output_file.write(reinterpret_cast<const char*>(frame.uv_data), uv_size);
                frames_dumped++;
            }

            decoder.releaseframe(frame);
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
    const std::string device_path = "/dev/video0";
    std::string input_stream = "./pf_stream/avc_fhd_15Mbps.h264";
    std::string output_frame = "./yuv_dump/dx_vdec_dump.yuv";
    bool enable_yuv_dump = false;
    std::string capture_io_mode = "auto";

    // Reduce kernel message verbosity to avoid log mixing
    std::cout << "[INFO] Adjusting kernel log level for cleaner output..." << std::endl;
    system("echo 3 > /proc/sys/kernel/printk 2>/dev/null || true");

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_stream = argv[++i];
            } else {
                std::cerr << "Error: --input requires a file path" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_frame = argv[++i];
            } else {
                std::cerr << "Error: --output requires a file path" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-d" || arg == "--dump-yuv") {
            enable_yuv_dump = true;
        } else if (arg == "--capture-io-mode") {
            if (i + 1 < argc) {
                capture_io_mode = argv[++i];
            } else {
                std::cerr << "Error: --capture-io-mode requires a value" << std::endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    Logger::setLevel(Logger::Level::DEBUG);
    Logger::enableTimestamp(true);

    DX_VDEC::IOMode io_mode = DX_VDEC::stringToIOMode(capture_io_mode);

    printCmaInfo();

    try {
        DX_VDEC decoder(io_mode);

        if (enable_yuv_dump && io_mode == DX_VDEC::IOMode::DMABUF_EXPORT) {
            decoder.enableDmabufMmap(true);
            std::cout << "DMABUF mmap enabled for YUV dump" << std::endl;
        }

        if (!decoder.open(device_path)) {
            std::cerr << "Failed to open decoder: " << decoder.getLastError() << std::endl;
            return 1;
        }

        if (!decoder.prepare(1920, 1080, DX_VDEC::InputCodecFormat::H264,
                             DX_VDEC::OutputColorFormat::NV12)) {
            std::cerr << "Failed to prepare decoder: " << decoder.getLastError() << std::endl;
            return 1;
        }

        if (!decoder.startStreaming()) {
            std::cerr << "Failed to start streaming: " << decoder.getLastError() << std::endl;
            return 1;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        std::thread output_th(GetFrameThread, std::ref(decoder), enable_yuv_dump);
        std::thread dump_th;

        if (enable_yuv_dump) {
            dump_th = std::thread(fileDumpThread, std::ref(decoder), output_frame);
        }

        output_th.join();
        if (enable_yuv_dump && dump_th.joinable()) {
            dump_th.join();
        }
        decoder.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << std::endl << "=== Threaded Decoding Complete ===" << std::endl;
        std::cout << "Frames fed: " << frames_fed.load() << std::endl;
        std::cout << "Frames decoded: " << frames_decoded.load() << std::endl;
        if (enable_yuv_dump) {
            std::cout << "Frames dumped: " << frames_dumped.load() << std::endl;
        }
        std::cout << "Total processing time: " << duration.count() << " ms" << std::endl;
        if (duration.count() > 0) {
            std::cout << "Decoding FPS: " << (frames_decoded.load() * 1000 / duration.count())
                      << std::endl;
        }
        if (enable_yuv_dump) {
            std::cout << "Output file: " << output_frame << std::endl;
        } else {
            std::cout << "Mode: Decode-only (no file output)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception caught: " << e.what() << std::endl;
        stop_threads = true;
        return 1;
    }

    printCmaInfo();

    std::cout << "=== API TEST COMPLETED ===" << std::endl;
    return 0;
}

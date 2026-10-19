/**
 * @file graph_report_writer.hpp
 * @brief --report, written frame by frame (SP1 U-04).
 *
 * The CLI used to hold every frame's JSON until the input ended - unbounded
 * on a camera or RTSP source, and all of it lost on SIGTERM or a crash. Now
 * the file is opened when the source opens, each frame is appended the
 * moment its report comes out, and the closing lines are written at the end.
 *
 * THE BYTES ARE TODAY'S. nlohmann::json objects are std::map-ordered, so
 * the old writer's {"graph":..,"frames":[..]}.dump(2) put "frames" first,
 * and dump(2) indents additively: a frame inside the "frames" array is its
 * own dump(2) with four spaces in front of every line. The sync/async
 * parity tests and every consumer compare these files byte for byte, and
 * graph_engine_test pins the equality for 0, 1 and 2 frames.
 *
 * KILL SAFETY. The FILE is unbuffered and each frame is ONE fwrite, so a
 * frame reaches the kernel in one write(). A SIGKILL between frames leaves
 * a prefix ending on a frame's closing brace; appending
 * "\n  ],\n  \"graph\": <name>\n}\n" makes it valid JSON again. Every normal
 * or unwinding exit finalizes: the destructor calls Finish().
 *
 * Text mode ("w"), as the std::ofstream it replaces, so a Windows build
 * keeps its line endings.
 */
#ifndef DXAPP_MULTI_MODEL_GRAPH_GRAPH_REPORT_WRITER_HPP
#define DXAPP_MULTI_MODEL_GRAPH_GRAPH_REPORT_WRITER_HPP

#include <cstddef>
#include <cstdio>
#include <string>

#include "common/third_party/nlohmann_json.hpp"

namespace dxapp {
namespace graph {
namespace cli {

/// `text` with `pad` in front of every line.
inline std::string IndentLines(const std::string& text, const std::string& pad) {
    std::string out;
    out.reserve(text.size() + text.size() / 8 + pad.size());
    out += pad;
    for (std::size_t i = 0; i < text.size(); ++i) {
        out += text[i];
        if (text[i] == '\n') out += pad;
    }
    return out;
}

class ReportFileWriter {
 public:
    ReportFileWriter() : file_(NULL), frames_(0) {}
    ~ReportFileWriter() { Finish(NULL); }

    bool Open(const std::string& path, const std::string& graph_name,
              std::string* error) {
        Finish(NULL);
        path_ = path;
        graph_json_ = nlohmann::json(graph_name).dump();
        frames_ = 0;
        file_ = std::fopen(path.c_str(), "w");
        if (file_ == NULL) return Fail(error);
        std::setvbuf(file_, NULL, _IONBF, 0);
        if (Write("{\n  \"frames\": [", error)) return true;
        // Not open: is_open() says so, and neither Append nor the
        // destructor's Finish writes into a file whose header never landed.
        std::fclose(file_);
        file_ = NULL;
        return false;
    }

    bool Append(const nlohmann::json& frame, std::string* error) {
        if (file_ == NULL) return Fail(error);
        const std::string chunk =
            (frames_ == 0 ? "\n" : ",\n") + IndentLines(frame.dump(2), "    ");
        if (!Write(chunk, error)) return false;
        ++frames_;
        return true;
    }

    bool Finish(std::string* error) {
        if (file_ == NULL) return true;
        const std::string tail = std::string(frames_ == 0 ? "],\n" : "\n  ],\n") +
                                 "  \"graph\": " + graph_json_ + "\n}\n";
        const bool wrote = Write(tail, error);
        const bool closed = std::fclose(file_) == 0;
        file_ = NULL;
        if (wrote && !closed) return Fail(error);
        return wrote;
    }

    bool is_open() const { return file_ != NULL; }
    std::size_t frames_written() const { return frames_; }

 private:
    ReportFileWriter(const ReportFileWriter&) = delete;
    ReportFileWriter& operator=(const ReportFileWriter&) = delete;

    bool Write(const std::string& text, std::string* error) {
        if (std::fwrite(text.data(), 1, text.size(), file_) != text.size()) {
            return Fail(error);
        }
        std::fflush(file_);
        return true;
    }

    bool Fail(std::string* error) const {
        if (error != NULL) *error = "could not write " + path_;
        return false;
    }

    std::FILE* file_;
    std::string path_;
    std::string graph_json_;
    std::size_t frames_;
};

/**
 * @brief --events: one JSON object per line (JSONL), the same objects the
 *        CLI prints after "[EVENT] ".
 *
 * Unbuffered, one fwrite per line, so a reader tailing the file (Studio,
 * Live) never sees half an event, and every event written before a kill is
 * whole. Opened before the first frame, closed by the destructor.
 */
class EventFileWriter {
 public:
    EventFileWriter() : file_(NULL) {}
    ~EventFileWriter() { Close(); }

    bool Open(const std::string& path, std::string* error) {
        Close();
        path_ = path;
        file_ = std::fopen(path.c_str(), "w");
        if (file_ == NULL) return Fail(error);
        std::setvbuf(file_, NULL, _IONBF, 0);
        return true;
    }

    bool Append(const std::string& line, std::string* error) {
        if (file_ == NULL) return Fail(error);
        const std::string text = line + "\n";
        if (std::fwrite(text.data(), 1, text.size(), file_) != text.size()) return Fail(error);
        return true;
    }

    void Close() {
        if (file_ == NULL) return;
        std::fclose(file_);
        file_ = NULL;
    }

    bool is_open() const { return file_ != NULL; }

 private:
    EventFileWriter(const EventFileWriter&) = delete;
    EventFileWriter& operator=(const EventFileWriter&) = delete;

    bool Fail(std::string* error) const {
        if (error != NULL) *error = "could not write " + path_;
        return false;
    }

    std::FILE* file_;
    std::string path_;
};

}  // namespace cli
}  // namespace graph
}  // namespace dxapp

#endif  // DXAPP_MULTI_MODEL_GRAPH_GRAPH_REPORT_WRITER_HPP

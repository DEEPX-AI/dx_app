// Test-only mock of the tiny OpenCV surface DisplayPump uses.
// Placed on the include path via -Itests/display/mock_include so that
// display_pump.hpp compiles UNMODIFIED (it includes <opencv2/opencv.hpp>).
#ifndef DXAPP_MOCK_OPENCV_HPP
#define DXAPP_MOCK_OPENCV_HPP

#include <atomic>
#include <memory>
#include <string>
#include <vector>

namespace cvmock {
struct Counters {
    std::atomic<int> named_window{0};
    std::atomic<int> resize_window{0};
    std::atomic<int> imshow{0};
    std::atomic<int> wait_key{0};
    std::atomic<int> get_window_prop{0};
    std::atomic<int> destroy_all{0};
    std::atomic<int> next_key{-1};        // value waitKey() returns
    std::atomic<double> window_visible{1.0};
    std::atomic<int> last_resize_w{0};      // last resizeWindow() width
    std::atomic<int> last_resize_h{0};      // last resizeWindow() height
    void reset() {
        named_window = 0; resize_window = 0; imshow = 0; wait_key = 0;
        get_window_prop = 0; destroy_all = 0; next_key = -1; window_visible = 1.0;
        last_resize_w = 0; last_resize_h = 0;
    }
};
inline Counters& counters() { static Counters c; return c; }
}  // namespace cvmock

namespace cv {

enum { WINDOW_NORMAL = 0 };
enum { WND_PROP_VISIBLE = 4 };

class Exception : public std::exception {
public:
    const char* what() const noexcept override { return "cv::Exception"; }
};

// Refcounted like the real cv::Mat: copy/assign shares pixels (O(1)),
// clone() deep-copies. Tests rely on this to prove offer() does not memcpy.
class Mat {
public:
    Mat() = default;
    Mat(int r, int c) : rows(r), cols(c),
        data_(std::make_shared<std::vector<unsigned char>>(
            static_cast<size_t>(r) * c * 3, 0)) {}
    bool empty() const { return !data_ || data_->empty(); }
    Mat clone() const {
        Mat m; m.rows = rows; m.cols = cols;
        if (data_) {
            m.data_ = std::make_shared<std::vector<unsigned char>>(*data_);
            ++clone_count();
        }
        return m;
    }
    // identity of the underlying pixel buffer — lets a test assert "same buffer"
    const void* buffer() const { return data_ ? static_cast<const void*>(data_->data()) : nullptr; }
    static int& clone_count() { static int n = 0; return n; }

    int rows = 0;
    int cols = 0;
private:
    std::shared_ptr<std::vector<unsigned char>> data_;
};

inline void namedWindow(const std::string&, int) { ++cvmock::counters().named_window; }
inline void resizeWindow(const std::string&, int w, int h) {
    ++cvmock::counters().resize_window;
    cvmock::counters().last_resize_w = w;
    cvmock::counters().last_resize_h = h;
}
inline void imshow(const std::string&, const Mat&) { ++cvmock::counters().imshow; }
inline int  waitKey(int) { ++cvmock::counters().wait_key; return cvmock::counters().next_key.load(); }
inline double getWindowProperty(const std::string&, int) {
    ++cvmock::counters().get_window_prop;
    return cvmock::counters().window_visible.load();
}
inline void destroyAllWindows() { ++cvmock::counters().destroy_all; }
inline void destroyWindow(const std::string&) { ++cvmock::counters().destroy_all; }

}  // namespace cv
#endif

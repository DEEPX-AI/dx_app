#include "utils.h"
#include "face_preprocess.h"

double GetTimestamp(void)
{
    double timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    return timestamp;
}

cv::Mat preprocess(cv::Mat image, cv::Size size)
{
    cv::Mat resized, converted;
    cv::resize(image, resized, size);
    cv::cvtColor(resized, converted, cv::COLOR_BGR2RGB);
    return converted;
}

void data_pre_processing(uint8_t* src, uint8_t* dst, int shape, int align)
{
    int copy_size = (int)(shape * 3);
    for(int y=0; y<shape; ++y)
    {
        memcpy(&dst[y * (copy_size + align)],
                &src[y * copy_size], 
                copy_size
                );
    }
};

cv::Rect get_rect(float *box, int image_w, int image_h)
{
    float x1 = box[0] * image_w;
    float y1 = box[1] * image_h;
    float x2 = box[2] * image_w;
    float y2 = box[3] * image_h;

    // expand bbox
    float ratio = 0.5;
    float bw = (x2 - x1) / 2 * ratio;
    float bh = (y2 - y1) / 2 * ratio;

    float xx1 = x1 - bw;
    float yy1 = y1 - bh;
    float xx2 = x2 + bw;
    float yy2 = y2 + bh;

    cv::Rect frame_window(0, 0, image_w, image_h);
    cv::Rect crop_window(xx1, yy1, xx2 - xx1, yy2 - yy1);
    cv::Rect rect = crop_window & frame_window;
    return rect;
}

float get_iou(cv::Rect rect1, cv::Rect rect2)
{
    float box[4] = {(float)rect1.x, (float)rect1.y, (float)rect1.x + rect1.width, (float)rect1.y + rect1.height};
    float truth[4] = {(float)rect2.x, (float)rect2.y, (float)rect2.x + rect2.width, (float)rect2.y + rect2.height};

    float ovr_left = std::max(box[0], truth[0]);
    float ovr_right = std::min(box[2], truth[2]);
    float ovr_top = std::max(box[1], truth[1]);
    float ovr_bottom = std::min(box[3], truth[3]);
    float ovr_width = ovr_right - ovr_left;
    float ovr_height = ovr_bottom - ovr_top;
    if (ovr_width < 0 || ovr_height < 0)
        return 0;
    float overlap_area = ovr_width * ovr_height;
    float union_area =
        (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - overlap_area;
    return overlap_area * 1.0 / union_area;
}

static std::vector<std::vector<int>> face_landmark_links = {
    // outlier
    {1, 9},
    {9, 10},
    {10, 11},
    {11, 12},
    {12, 13},
    {13, 14},
    {14, 15},
    {15, 16},
    {16, 2},
    {2, 3},
    {3, 4},
    {4, 5},
    {5, 6},
    {6, 7},
    {7, 8},
    {8, 0},
    {0, 24},
    {24, 23},
    {23, 22},
    {22, 21},
    {21, 20},
    {20, 19},
    {19, 18},
    {18, 32},
    {32, 31},
    {31, 30},
    {30, 29},
    {29, 28},
    {28, 27},
    {27, 26},
    {26, 25},
    {25, 17},
    // lips out
    {52, 55},
    {55, 56},
    {56, 53},
    {53, 59},
    {59, 58},
    {58, 61},
    {61, 68},
    {68, 67},
    {67, 71},
    {71, 63},
    {63, 64},
    {64, 52},
    // lips in
    {65, 54},
    {54, 60},
    {60, 57},
    {57, 69},
    {69, 70},
    {70, 62},
    {62, 66},
    {66, 65},
    // nose
    {72, 73},
    {73, 74},
    {74, 86},
    {75, 76},
    {76, 77},
    {77, 78},
    {78, 79},
    {79, 80},
    {80, 85},
    {85, 84},
    {84, 83},
    {83, 82},
    {82, 81},
    // eye right
    {35, 36},
    {36, 33},
    {33, 37},
    {37, 39},
    {39, 42},
    {42, 40},
    {40, 41},
    {41, 35},
    // eye left
    {89, 90},
    {90, 87},
    {87, 91},
    {91, 93},
    {93, 96},
    {96, 94},
    {94, 95},
    {95, 89},
    // blow right
    {43, 44},
    {44, 45},
    {45, 47},
    {47, 46},
    {46, 50},
    {50, 51},
    {51, 49},
    {49, 48},
    {48, 43},
    // blow left
    {101, 100},
    {100, 99},
    {99, 98},
    {98, 97},
    {97, 102},
    {102, 103},
    {103, 104},
    {104, 105},
    {105, 101},
};

std::vector<cv::Point2f> get_landmark(float *fl_data, int w_crop, int h_crop, float x, float y)
{
    std::vector<cv::Point2f> landmark;
    int fl_length = 106;
    for (int j = 0; j < fl_length; j++)
    {
        float lx = (fl_data[j * 2] + 1) * 0.5 * w_crop + x;
        float ly = (fl_data[j * 2 + 1] + 1) * 0.5 * h_crop + y;
        landmark.emplace_back(cv::Point2f(lx, ly));
    }
    return landmark;
}

void visualize_landmark(cv::Mat image, std::vector<cv::Point2f> landmark)
{
    for (size_t j = 0; j < face_landmark_links.size(); j++)
    {
        auto p0 = landmark[face_landmark_links[j][0]];
        auto p1 = landmark[face_landmark_links[j][1]];
        cv::line(image, p0, p1, cv::Scalar(255, 255, 0), 1);
    }
}

cv::Mat warp(cv::Mat image, std::vector<cv::Point2f> landmark)
{
    float src_landmark[5][2] = {
        {landmark[38].x, landmark[38].y},
        {landmark[88].x, landmark[88].y},
        {landmark[86].x, landmark[86].y},
        {landmark[52].x, landmark[52].y},
        {landmark[61].x, landmark[61].y}};
    float dst_landmark[5][2] = {
        {38.2946, 51.6963},
        {73.5318, 51.5014},
        {56.0252, 71.7366},
        {41.5493, 92.3655},
        {70.7299, 92.2041}};
    cv::Mat src(5, 2, CV_32F, src_landmark);
    cv::Mat dst(5, 2, CV_32F, dst_landmark);
    auto M = FacePreprocess::similarTransform(src, dst);
    cv::Mat transform_matrix(2, 3, CV_32F, M.data);
    cv::Mat warped_image;
    cv::warpAffine(image, warped_image, transform_matrix, cv::Size(112, 112));
    return warped_image;
}

inline static float np_dot_1d(float *a, float *b, int s)
{
    float sum = 0;
    for (int i = 0; i < s; i++)
        sum += (a[i] * b[i]);
    return sum;
}
inline static float np_linalg_norm(float *d, int s)
{
    float sum = 0;
    for (int i = 0; i < s; i++)
        sum += (d[i] * d[i]);
    return sqrt(sum);
}
float cos_sim(float *a, float *b, int s)
{
    float f1 = np_dot_1d(a, b, s);
    float f2 = np_linalg_norm(a, s);
    float f3 = np_linalg_norm(b, s);
    float result = f1 / (f2 * f3);
    return result;
}

Instance::Instance()
{
    id = 0;
    box = cv::Rect(0, 0, 0, 0);
}

Instance::Instance(int _id, cv::Rect _box)
{
    id = _id;
    box = _box;
}

Tracker::Tracker(float _iou_threshold)
{
    id = 0;
    iou_threshold = _iou_threshold;
}

void Tracker::run(std::vector<cv::Rect> D)
{
    float iou_max_threshold = 0.25;
    std::vector<Instance> T_temp;
    for (size_t i = 0; i < T.size(); i++)
    {
        Instance tracked = T[i];
        float iou_max = 0;
        int iou_max_index = -1;
        for (size_t j = 0; j < D.size(); j++)
        {
            float iou = get_iou(tracked.box, D[j]);
            if (iou_max < iou)
            {
                iou_max = iou;
                iou_max_index = j;
            }
        }

        if (iou_max > iou_max_threshold)
        {
            tracked.box = D[iou_max_index];
            T_temp.emplace_back(tracked);
            D.erase(D.begin() + iou_max_index);
        }
    }
    for (size_t j = 0; j < D.size(); j++)
    {
        T_temp.emplace_back(Instance(id, D[j]));
        id++;
    }
    T = T_temp;
}

FaceData::FaceData(int _id, cv::Mat _image, float *_feature_vector)
{
    id = _id;
    image = _image;
    memcpy(feature_vector, _feature_vector, sizeof(float) * 512);
}
FaceData::~FaceData() {}
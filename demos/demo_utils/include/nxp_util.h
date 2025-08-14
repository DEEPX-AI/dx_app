#pragma once

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "bbox.h"

#define VDMA_CNN_BUF_MAX	4
#define VDMA_CNN_IMG_HSIZE  640		//1920
#define VDMA_CNN_IMG_VSIZE  480		//1080

class FrameBuffer
{
public:
    FrameBuffer(std::string dev, uint32_t numBuf_);
    void Clear(void);
    void DrawBoxes(int bufId, std::vector<BoundingBox> &result, std::vector<cv::Scalar> &colors, float OriginHeight, float OriginWidth);
    void EraseBoxes(int bufId, std::vector<BoundingBox> &result, float OriginHeight, float OriginWidth);
    void Show();
    void* get_data(){return data[0];};
private:
    int fd;
    uint32_t numBuf;
    uint32_t bpp; /* Bytes Per Pixel */
    void *data[8];
    struct fb_var_screeninfo vinfo;
    struct fb_fix_screeninfo finfo;
    uint32_t xres;
    uint32_t yres;
    long int screenSize;
};

namespace nxp
{
    int print_caps(int fd);
    int init_mmap(int fd, uint8_t* buffer[]);
    int capture_image(int fd, uint8_t* buffer[], uint8_t *dst); 
}
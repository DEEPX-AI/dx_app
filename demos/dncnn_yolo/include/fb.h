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
#include "bbox.h"

class FrameBuffer
{
public:
    FrameBuffer(std::string dev, uint32_t numBuf_);
    void Clear(void);
    void DrawBoxes(int bufId, std::vector<BoundingBox> &result, float OriginHeight, float OriginWidth);
    void EraseBoxes(int bufId, std::vector<BoundingBox> &result, float OriginHeight, float OriginWidth);
    void Show();
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
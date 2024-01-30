#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h>
#include <string>
#include <vector>
#include "fb.h"

using namespace std;

FrameBuffer::FrameBuffer(string dev, uint32_t numBuf_)
:numBuf(numBuf_)
{
    void *ret = nullptr;
    fd = open((char*)dev.c_str(), O_RDWR);
    if (fd == -1) {
        perror("Error: cannot open framebuffer device");
        exit(1);
    }

    if (ioctl(fd, FBIOGET_FSCREENINFO, &finfo) == -1) {
        perror("Error reading fixed information");
        exit(2);
    }

    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        perror("Error reading variable information");
        exit(3);
    }
    screenSize = vinfo.xres * vinfo.yres * vinfo.bits_per_pixel / 8;
    xres = vinfo.xres;
    yres = vinfo.yres;
    bpp = vinfo.bits_per_pixel/8;

    printf("    - Framebuffer Resolution: %dx%d, %d bytes per pixel\n", vinfo.xres, vinfo.yres, bpp);
    printf("    - Framebuffer size: %ld bytes x %d\n", screenSize, numBuf);

    data[0] = mmap(0, screenSize*numBuf, PROT_WRITE , MAP_SHARED, fd, 0);
    if (data[0] == (void *)-1) {
        perror("Error: failed to mmap framebuffer device to memory");
        exit(4);
    }
    for(int i=1;i<numBuf;i++)
    {
        data[i] = data[0] + screenSize*i;
    }
    cout << "    - Framebuffer Pointer: " << hex << (uint64_t)data[0] << dec << endl;
}
void FrameBuffer::Clear()
{
    char *buf = (char*)data[0];
    for(int i=0;i<screenSize*numBuf;i++)
    {
        buf[i] = 0;
    }
}
uint8_t zeros[4] = { 0, 0, 0, 0 };
uint8_t colorRed[4] = { 255, 0, 0, 255 };
uint8_t colorGreen[4] = { 0, 255, 0, 255 };
uint8_t colorBlue[4] = { 0, 0, 255, 255 };
void FrameBuffer::DrawBoxes(int bufId, vector<BoundingBox> &result, float OriginHeight, float OriginWidth)
{
    float rx = OriginWidth/xres;
    float ry = OriginHeight/yres;
    uint32_t x1, y1, x2, y2;
    void *buf = data[bufId];
    for(auto &bbox:result)
    {
        x1 = bbox.box[0]/rx;
        y1 = bbox.box[1]/ry;
        x2 = bbox.box[2]/rx;
        y2 = bbox.box[3]/ry;
        // cout << "    ++ (" << bbox.labelname << ") (" << x1 << ", " << y1 << ")" << ", (" << x2 << ", " << y2 << ")" << endl;
        for(int i = x1; i<x2; i++)
        {
            memcpy((void*)(buf + bpp * (y1*xres + i)), colorGreen, 4);
            memcpy((void*)(buf + bpp * (y2*xres + i)), colorGreen, 4);
        }
        for(int i = y1; i<y2; i++)
        {
            memcpy((void*)(buf + bpp * (i*xres + x1)), colorGreen, 4);
            memcpy((void*)(buf + bpp * (i*xres + x2)), colorGreen, 4);
        }
    }
    vinfo.yoffset = yres * bufId;
    if (ioctl(fd, FBIOPAN_DISPLAY, &vinfo)) {
		perror("Error panning display");
		exit(5);
	}
}
void FrameBuffer::EraseBoxes(int bufId, vector<BoundingBox> &result, float OriginHeight, float OriginWidth)
{
    float rx = OriginWidth/xres;
    float ry = OriginHeight/yres;
    uint32_t x1, y1, x2, y2;
    char *buf = (char*)data[bufId];
    for(auto &bbox:result)
    {
        x1 = bbox.box[0]/rx;
        y1 = bbox.box[1]/ry;
        x2 = bbox.box[2]/rx;
        y2 = bbox.box[3]/ry;
        // cout << "    -- (" << bbox.labelname << ") (" << x1 << ", " << y1 << ")" << ", (" << x2 << ", " << y2 << ")" << endl;
        for(int i = x1; i<x2; i++)
        {
            buf[ bpp * (y1*xres + i) + 3 ] = 0;
            buf[ bpp * (y2*xres + i) + 3 ] = 0;
        }
        for(int i = y1; i<y2; i++)
        {
            buf[ bpp * (i*xres + x1) + 3 ] = 0;
            buf[ bpp * (i*xres + x2) + 3 ] = 0;
        }
    }
}
void FrameBuffer::Show()
{
    /* TODO */
}
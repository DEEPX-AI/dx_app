#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <string.h> // for memcpy
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "nxp.h"

FrameBuffer::FrameBuffer(string dev, uint32_t numBuf_)
:numBuf(numBuf_)
{
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
    for(int i=1;i<(int)numBuf;i++)
    {
        data[i] = (FrameBuffer*)((uintptr_t)data[0] + screenSize*i);
    }
    std::cout << "    - Framebuffer Pointer: " << std::hex << (uint64_t)data[0] << std::dec << std::endl;
}
void FrameBuffer::Clear()
{
    char *buf = (char*)data[0];
    for(int i=0;i<screenSize*numBuf;i++)
    {
        buf[i] = 0;
    }
}

void FrameBuffer::DrawBoxes(int bufId, std::vector<BoundingBox> &result, std::vector<cv::Scalar> &colors, float OriginHeight, float OriginWidth)
{
    void *buf = data[bufId];
    cv::Mat image(yres, xres, CV_8UC4, buf, xres * bpp);
    float rx = OriginWidth/xres;
    float ry = OriginHeight/yres;
    uint32_t x1, y1, x2, y2;

    for(auto &bbox:result)
    {
        x1 = bbox.box[0]/rx < 0.f ? 0: (uint32_t)(bbox.box[0]/rx);
        y1 = bbox.box[1]/ry < 0.f ? 0: (uint32_t)(bbox.box[1]/ry);
        x2 = bbox.box[2]/rx > xres ? xres: (uint32_t)(bbox.box[2]/rx);
        y2 = bbox.box[3]/ry > yres ? yres: (uint32_t)(bbox.box[3]/ry);
        char label_score[64] = {0, };
        sprintf(label_score, "%s %.4lf", bbox.labelname, bbox.score);
        cv::rectangle(image, 
            cv::Point(x1, y1), 
            cv::Point(x2, y2), 
            cv::Scalar(colors[bbox.label][0], colors[bbox.label][1], colors[bbox.label][2], 255), 
            2, 8, 0);
        cv::putText(image, label_score, cv::Point(x1, y1-5),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                cv::Scalar(colors[bbox.label][0], colors[bbox.label][1], colors[bbox.label][2], 255), 
                2);
    }
    vinfo.yoffset = yres * bufId;
    if (ioctl(fd, FBIOPAN_DISPLAY, &vinfo)) {
		perror("Error panning display");
		exit(5);
	}
}
void FrameBuffer::EraseBoxes(int bufId, std::vector<BoundingBox> &result, float OriginHeight, float OriginWidth)
{
    char *buf = (char*)data[bufId];
    memset(buf, 0, screenSize); 
}


static int xioctl(int fd, int request, void *arg)
{
        int r;

        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
}

int print_caps(int fd)
{
        struct v4l2_capability caps = {};

        if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &caps))
        {
                perror("Querying Capabilities");
                return 1;
        }

        printf( "Driver Caps:\n"
                "  Driver: \"%s\"\n"
                "  Card: \"%s\"\n"
                "  Bus: \"%s\"\n"
                "  Version: %d.%d\n"
                "  Capabilities: %08x\n",
                caps.driver,
                caps.card,
                caps.bus_info,
                (caps.version>>16)&&0xff,
                (caps.version>>24)&&0xff,
                caps.capabilities);


        struct v4l2_cropcap cropcap = {0};
        cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (-1 == xioctl (fd, VIDIOC_CROPCAP, &cropcap))
        {
                perror("Querying Cropping Capabilities");
                //return 1;
        }

        printf( "Camera Cropping:\n"
                "  Bounds: %dx%d+%d+%d\n"
                "  Default: %dx%d+%d+%d\n"
                "  Aspect: %d/%d\n",
                cropcap.bounds.width, cropcap.bounds.height, cropcap.bounds.left, cropcap.bounds.top,
                cropcap.defrect.width, cropcap.defrect.height, cropcap.defrect.left, cropcap.defrect.top,
                cropcap.pixelaspect.numerator, cropcap.pixelaspect.denominator);

        int support_grbg10 = 0;

        struct v4l2_fmtdesc fmtdesc = {0};
        fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        char fourcc[5] = {0};
        char c, e;
        printf("  FMT : CE Desc\n--------------------\n");
        while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
        {
        	strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
            if (fmtdesc.pixelformat == V4L2_PIX_FMT_SGRBG10)
                support_grbg10 = 1;
            c = fmtdesc.flags & 1? 'C' : ' ';
            e = fmtdesc.flags & 2? 'E' : ' ';
            printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
            fmtdesc.index++;
        }
        /*
        if (!support_grbg10)
        {
            printf("Doesn't support GRBG10.\n");
            return 1;
        }*/

        struct v4l2_format fmt = {0};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_G_FMT, &fmt) < 0)
        {
                 printf("get format failed\n");
                 return -1;
        }
        else
        {
        	printf("Width = %d\n", fmt.fmt.pix.width);
            printf("Height = %d\n", fmt.fmt.pix.height);
            printf("Image size = %d\n", fmt.fmt.pix.sizeimage);
            printf("pixelformat = %d\n", fmt.fmt.pix.pixelformat);
        }


        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = VDMA_CNN_IMG_HSIZE;
        fmt.fmt.pix.height = VDMA_CNN_IMG_VSIZE;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
        // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;
        fmt.fmt.pix.field = V4L2_FIELD_NONE;

        // fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        // fmt.fmt.pix.width = 1280;
        // fmt.fmt.pix.height = 720;
        // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32;
        // fmt.fmt.pix.field = V4L2_FIELD_NONE;

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            perror("Setting Pixel Format");
            return 1;
        }

        strncpy(fourcc, (char *)&fmt.fmt.pix.pixelformat, 4);
        printf( "Selected Camera Mode:\n"
                "  Width: %d\n"
                "  Height: %d\n"
                "  PixFmt: %s\n"
                "  Field: %d\n",
                fmt.fmt.pix.width,
                fmt.fmt.pix.height,
                fourcc,
                fmt.fmt.pix.field);
        return 0;
}

int init_mmap(int fd, uint8_t* buffer[])
{
    struct v4l2_requestbuffers req = {0};
   	struct v4l2_buffer buf = {0};
	int i;

    req.count = VDMA_CNN_BUF_MAX;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

	printf("#### %s : VIDIOC_REQBUFS ####\n", __func__);
    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req))
    {
        perror("Requesting Buffer");
        return 1;
    }

	for(i=0;i<VDMA_CNN_BUF_MAX;i++)
	{
		printf(" #### VIDIOC_QUERYBUF ####\n", __func__);
    	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    	buf.memory = V4L2_MEMORY_MMAP;
    	buf.index = i;
    	if(-1 == xioctl(fd, VIDIOC_QUERYBUF, &buf))
    	{
    	    perror("Querying Buffer");
    	    return 1;
    	}

    	printf("buf.length: %d\n", buf.length);
    	printf("buf.imagelength: %d\n", buf.bytesused);
		printf("buf.m.offset = %x\n", buf.m.offset);

    	buffer[i] = (uint8_t*)mmap (NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    	printf("buffer[%d] address: %p\n", i, buffer[i]);


    	if(-1 == xioctl(fd, VIDIOC_QBUF, &buf))
    	{
    	    perror("Query Buffer");
    	    return 1;
    	}
	}

	printf("--------------------------------------------------------\n");

    if(-1 == xioctl(fd, VIDIOC_STREAMON, &buf.type))
    {
        perror("Start Capture");
        return 1;
    }


    return 0;
}

int capture_image(int fd, uint8_t* buffer[], uint8_t *dst)
{
    static struct v4l2_buffer buf = {0};
	static unsigned int cnt = 0;
	int i;
	char *start, *end;

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    // printf ("DQBUF image\n");
    if(-1 == xioctl(fd, VIDIOC_DQBUF, &buf))
    {
        perror("Retrieving Frame");
        return 1;
    }

    // printf("capture video image\n");
    memcpy(dst, buffer[buf.index],buf.length);

    if (ioctl (fd, VIDIOC_QBUF, &buf) < 0) {
    	printf("VIDIOC_QBUF failed\n");
   		return 1;
    }

    return 0;
}

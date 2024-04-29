#include "v4l2.h"
#include "dxrt/dxrt_api.h"

using namespace std;

#define VDMA_CNN_BUF_MAX	4

static int xioctl(int fd, int request, void *arg)
{
        int r;

        do r = ioctl (fd, request, arg);
        while (-1 == r && EINTR == errno);

        return r;
}
int SetVideoFormat(int fd, int height, int width)
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

    struct v4l2_cropcap cropcap = {};
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

    struct v4l2_fmtdesc fmtdesc = {};
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    char fourcc[5] = {0,};
    char c, e;
    printf("  FMT : CE Desc\n--------------------\n");
    while (0 == xioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc))
    {
        strncpy(fourcc, (char *)&fmtdesc.pixelformat, 4);
        c = fmtdesc.flags & 1? 'C' : ' ';
        e = fmtdesc.flags & 2? 'E' : ' ';
        printf("  %s: %c%c %s\n", fourcc, c, e, fmtdesc.description);
        fmtdesc.index++;
    }

    struct v4l2_format fmt = {};
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
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    // fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB24;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    //fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    //fmt.fmt.pix.width = 1280;
    //fmt.fmt.pix.height = 720;
    //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_RGB32;
    //fmt.fmt.pix.field = V4L2_FIELD_NONE;

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

V4L2CaptureWorker::V4L2CaptureWorker(string dev, int height_, int width_, int numBuf_, vector<unsigned long> userPtr_)
: height(height_), width(width_), numBuf(numBuf_), userPtr(userPtr_)
{
    struct v4l2_requestbuffers req = {};
   	struct v4l2_buffer buf = {};
    volatile int ret;
    devFd = open(dev.c_str(), O_RDWR);
    DXRT_ASSERT(devFd>0, "Fail to open V4L2 device.");
    SetVideoFormat(devFd, height, width);

    req.count = numBuf;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

	printf("#### %s : VIDIOC_REQBUFS ####\n", __func__);
    ret = xioctl(devFd, VIDIOC_REQBUFS, &req);
    DXRT_ASSERT(ret!=-1, "V4L2 Error in VIDIOC_REQBUFS");
    userPtr = userPtr_;

	for(int i=0;i<(int)userPtr.size();i++)
	{
		printf(" #### %s : VIDIOC_QUERYBUF ####\n", __func__);
    	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    	buf.memory = V4L2_MEMORY_USERPTR;
    	buf.index = i;
        ret = xioctl(devFd, VIDIOC_QUERYBUF, &buf);
        DXRT_ASSERT(ret!=-1, "V4L2 Error in VIDIOC_QUERYBUF");
        buf.m.userptr = (unsigned long) userPtr[i];
    	printf("buf.length: %d\n", buf.length);
    	printf("buf.imagelength: %d\n", buf.bytesused);
		printf("buf.m.offset = %x\n", buf.m.offset);
		printf("buf.m.userptr = %lx\n", buf.m.userptr);
        ret = xioctl(devFd, VIDIOC_QBUF, &buf);
        DXRT_ASSERT(ret!=-1, "V4L2 Error in VIDIOC_QBUF");
	}

	printf("--------------------------------------------------------\n");
    ret = xioctl(devFd, VIDIOC_STREAMON, &buf.type);
    DXRT_ASSERT(ret!=-1, "V4L2 Error in VIDIOC_STREAMON");
    captureThread = thread(&V4L2CaptureWorker::CaptureV4L2FrameThread, this);
}
V4L2CaptureWorker::~V4L2CaptureWorker() {}
void V4L2CaptureWorker::PushFrameId(int id)
{
    unique_lock<mutex> lk(frameIdLock);
    frameIdList.push_back(id);
    if(frameIdList.size()>20)
    {
        frameIdList.erase(frameIdList.begin());
    }
}
int V4L2CaptureWorker::GetFrameId(void)
{
    unique_lock<mutex> lk(frameIdLock);
    int ret;
    if(frameIdList.empty()) return -1;
    ret = frameIdList.back();
    return ret;
}
void V4L2CaptureWorker::Stop()
{
    cout << "==== Stop to capture frame" << endl;
    stop = true;
    captureThread.join();
}
void V4L2CaptureWorker::CaptureV4L2FrameThread()
{
    static unsigned int cnt = 0;
    stop = false;
    auto& profiler = dxrt::Profiler::GetInstance();
    while(!stop)
    {
        profiler.Start("capture");
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        buf.index = 0;

        profiler.Start("dqbuf");
        if(-1 == xioctl(devFd, VIDIOC_DQBUF, &buf))
        {
            perror("Retrieving Frame");
            return;
        }
        profiler.End("dqbuf");      
        // {
        //     fp = fopen(("frame"+to_string(cnt++)+".raw").c_str(), "wb");
        //     fwrite((uint8_t*)buf.m.userptr, buf.length, 1, fp);
        //     fclose(fp);
        //     system("sync");
        // }
        
        PushFrameId(buf.index);
        if (ioctl (devFd, VIDIOC_QBUF, &buf) < 0) {
            printf("VIDIOC_QBUF failed\n");
            return;
        }
        cnt++;
        profiler.End("capture");
    }
    profiler.Show();
}

#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <iostream>
#include <mutex>
#include <atomic>
#include <signal.h>
#include <thread>
#include <vector>

class V4L2CaptureWorker
{
public:
    V4L2CaptureWorker(std::string dev, int height_, int width_, int numBuf_, std::vector<unsigned long> userPtr_);
    ~V4L2CaptureWorker();
    void PushFrameId(int id);
    int GetFrameId();
    void Stop();
private:
    int devFd;
    int height;
    int width;
    int numBuf;
    std::vector<unsigned long> userPtr;
    std::atomic<bool> stop;
    std::mutex frameIdLock;    
    std::vector<int> frameIdList;
    std::thread captureThread;
    void CaptureV4L2FrameThread(void);
};
#include "segmentation.h"
#include <iostream>
void Segmentation(uint16_t *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses)
{
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int cls = input[cols*h + w];
            if(cls<numClasses)
            {
                output[3*cols*h + 3*w + 0] = cfg[cls].colorB;
                output[3*cols*h + 3*w + 1] = cfg[cls].colorG;
                output[3*cols*h + 3*w + 2] = cfg[cls].colorR;
            }
        }
    }
}


void Segmentation(float *input, uint8_t *output, int rows, int cols, SegmentationParam *cfg, int numClasses, const std::vector<int64_t>& shape)
{
    /**
     *  The provided DeepLabV3 model has different output data formats depending on use-ort mode.
     *  When need_transpose is False (use-ort off mode), the output data format is [1, 640, 640, 64].
     *  When need_transpose is True (use-ort on mode), the output data format is [1, 19, 640, 640].
     *  Test results show that FPS improves by 1.5x when use-ort is off.
     */
    bool need_transpose = shape[1] == numClasses? true : false;
    int compare_max_idx, compare_channel_idx;
    int pitch = shape[3];
    for(int h=0;h<rows;h++)
    {
        for(int w=0;w<cols;w++)
        {
            int maxIdx = 0;
            for (int c=0;c<numClasses;c++)
            {
                if(need_transpose)
                {
                    compare_max_idx = w + (cols * h) + (maxIdx * rows * cols);
                    compare_channel_idx = w + (cols * h) + (c * rows * cols);
                }
                else
                {
                    compare_max_idx = maxIdx + ((cols * h) + w) * pitch;
                    compare_channel_idx = c + ((cols * h) + w) * pitch;
                }
                if(input[compare_max_idx] < input[compare_channel_idx])
                {
                    maxIdx = c;
                }
            }
            output[3*cols*h + 3*w + 2] = cfg[maxIdx].colorB;
            output[3*cols*h + 3*w + 1] = cfg[maxIdx].colorG;
            output[3*cols*h + 3*w + 0] = cfg[maxIdx].colorR;
        }
    }
}
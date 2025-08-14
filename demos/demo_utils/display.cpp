#include "display.h"
#include <utils/color_table.hpp>
#include <cstdio>

void DisplayBoundingBox(cv::Mat& frame, 
                       std::vector<BoundingBox>& result, 
                       float OriginHeight, 
                       float OriginWidth, 
                       std::vector<cv::Scalar> ObjectColors, 
                       PostProcType type, 
                       bool ImageCenterAligned, 
                       float InputWidth, 
                       float InputHeight)
{
    std::map<std::string, int> numObjects;
    float x1, y1, x2, y2, w_pad = 0, h_pad = 0;
    float w = (float)frame.cols;  // Target Frame Width
    float h = (float)frame.rows;  // Target Frame Height
    float r = std::min(OriginWidth/w, OriginHeight/h);
    bool reformatting = false;
    float reformatting_ratio_width = 1.f, reformatting_ratio_height = 1.f;
    int txtBaseline = 0;

    if (InputWidth > 0) 
    {
        if (w/h == InputWidth/InputHeight) 
        {
            reformatting = false;
        } 
        else
        {
            reformatting = true;
            reformatting_ratio_width = w/InputWidth;
            reformatting_ratio_height = h/InputHeight;
        }
    }

    if (reformatting) 
    {
        r = std::min(OriginWidth/InputWidth, OriginHeight/InputHeight);
        w_pad = ImageCenterAligned ? (OriginWidth - InputWidth*r)/2. : 0;
        h_pad = ImageCenterAligned ? (OriginHeight - InputHeight*r)/2. : 0;
    } 
    else 
    {
        r = std::min(OriginWidth/w, OriginHeight/h);
        w_pad = ImageCenterAligned ? (OriginWidth - w*r)/2. : 0;
        h_pad = ImageCenterAligned ? (OriginHeight - h*r)/2. : 0;
    }

    for (auto& bbox : result) 
    {
        x1 = (bbox.box[0] - w_pad)/r;
        x2 = (bbox.box[2] - w_pad)/r;
        y1 = (bbox.box[1] - h_pad)/r;
        y2 = (bbox.box[3] - h_pad)/r;
        
        if (reformatting) {
            x1 *= reformatting_ratio_width;
            x2 *= reformatting_ratio_width;
            y1 *= reformatting_ratio_height;
            y2 *= reformatting_ratio_height;
        }
        
        x1 = std::min((float)w, std::max((float)0.0, x1));
        x2 = std::min((float)w, std::max((float)0.0, x2));
        y1 = std::min((float)h, std::max((float)0.0, y1));
        y2 = std::min((float)h, std::max((float)0.0, y2));

        // Create label text with score (rounded to 2 decimal places)
        char scoreStr[16];
        snprintf(scoreStr, sizeof(scoreStr), "%.2f", bbox.score);
        std::string labelWithScore = bbox.labelname + "=" + scoreStr;
        
        // Get text size for the combined label and score
        auto textSize = cv::getTextSize(labelWithScore, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &txtBaseline);
        
        // Draw bounding box
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), ObjectColors[bbox.label], 2);
        
        // Draw label background
        cv::rectangle(frame, 
                     cv::Point(x1, y1-textSize.height), 
                     cv::Point(x1 + textSize.width, y1), 
                     ObjectColors[bbox.label], 
                     cv::FILLED);
        
        // Draw label text with score
        cv::putText(frame, labelWithScore, cv::Point(x1, y1), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255));

        if (type == PostProcType::POSE) 
        {
            std::vector<cv::Point> points;
            for (int k = 0; k < 17; k++) 
            {
                float kx = (bbox.kpt[k*3+0] - w_pad) / r;
                float ky = (bbox.kpt[k*3+1] - h_pad) / r;
                float ks = bbox.kpt[k*3+2];
                if (ks > 0.5) 
                    points.emplace_back(cv::Point(kx, ky));
                else
                    points.emplace_back(cv::Point(-1, -1));
            }
            
            for (int index=0; index < (int)dxapp::common::skeleton_nodes.size(); index++) 
            {
                auto pp = dxapp::common::skeleton_nodes[index];
                auto kp0 = points[pp[0]];
                auto kp1 = points[pp[1]];
                if (kp0.x >= 0 && kp1.x >= 0)
                    cv::line(frame, kp0, kp1, dxapp::common::pose_limb_color[index], 2, cv::LINE_AA);
            }

            for (int index=0; index < (int)points.size(); index++) 
            {
                cv::circle(frame, points[index], 3, dxapp::common::pose_kpt_color[index], -1, cv::LINE_AA);
            }
        }
        else if (type == PostProcType::FACE) 
        {
            std::vector<cv::Point> points;
            for (int k = 0; k < 5; k++) 
            {
                float kx = (bbox.kpt[k*3+0] - w_pad) / r;
                float ky = (bbox.kpt[k*3+1] - h_pad) / r;
                float ks = bbox.kpt[k*3+2];
                if (ks >= 0.5) 
                    points.emplace_back(cv::Point(kx, ky));
                else
                    points.emplace_back(cv::Point(-1, -1));
            }
            
            for (int index=0; index < (int)points.size(); index++) 
            {
                cv::circle(frame, points[index], 3, dxapp::common::pose_kpt_color[index], -1, cv::LINE_AA);
            }
        }
    }
}
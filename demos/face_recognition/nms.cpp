#include <algorithm>
#include "nms.h"
#include "bbox.h"

using namespace std;
static bool compare(BoundingBox &r1, BoundingBox &r2)
{
    return r1.score > r2.score;
}

float CalcIOU(BoundingBox &a, BoundingBox &b)
{
    return CalcIOU(a.box, b.box);
}

float CalcIOU(float *box, float *truth)
{
    float ovr_left = max(box[0], truth[0]);
    float ovr_right = min(box[2], truth[2]);
    float ovr_top = max(box[1], truth[1]);
    float ovr_bottom = min(box[3], truth[3]);
    float ovr_width = ovr_right - ovr_left;
    float ovr_height = ovr_bottom - ovr_top;
    if (ovr_width < 0 || ovr_height < 0)
        return 0;
    float overlap_area = ovr_width * ovr_height;
    float union_area =
        (box[2] - box[0]) * (box[3] - box[1]) + (truth[2] - truth[0]) * (truth[3] - truth[1]) - overlap_area;
    return overlap_area * 1.0 / union_area;
}

void NmsOneClass(
    unsigned int cls,
    vector<string> &ClassNames,
    vector<vector<pair<float, int>>> &ScoreIndices,
    float *Boxes, float IouThreshold,
    vector<BoundingBox> &Result)
{
    float iou;
    int i, j;
    int numCandidates = ScoreIndices[cls].size();
    bool valid[numCandidates];
    fill_n(valid, numCandidates, true);
    for (i = 0; i < numCandidates; i++)
    {
        if (!valid[i])
        {
            continue;
        }
        auto box = BoundingBox(cls, (char *)ClassNames[cls].c_str(), ScoreIndices[cls][i].first,
                               Boxes[4 * ScoreIndices[cls][i].second],
                               Boxes[4 * ScoreIndices[cls][i].second + 1],
                               Boxes[4 * ScoreIndices[cls][i].second + 2],
                               Boxes[4 * ScoreIndices[cls][i].second + 3]);
        Result.emplace_back(box);
        for (j = i + 1; j < numCandidates; j++)
        {
            if (!valid[j])
            {
                continue;
            }
            iou = CalcIOU(
                &Boxes[4 * ScoreIndices[cls][j].second],
                &Boxes[4 * ScoreIndices[cls][i].second]);
            if (iou > IouThreshold)
            {
                valid[j] = false;
            }
        }
    }
}

void Nms(
    const size_t &numClass,
    const int &numDetectTotal,
    vector<string> &ClassNames,
    vector<vector<pair<float, int>>> &ScoreIndices,
    float *Boxes, const float &IouThreshold,
    vector<BoundingBox> &Result,
    int startClass)
{
    for (size_t cls = startClass; cls < numClass; cls++)
    {
        NmsOneClass(cls, ClassNames, ScoreIndices, Boxes, IouThreshold, Result);
    }
    sort(Result.begin(), Result.end(), compare);
    if (numDetectTotal > 0 && Result.size() > (size_t)numDetectTotal)
    {
        Result.resize(numDetectTotal);
    }
}
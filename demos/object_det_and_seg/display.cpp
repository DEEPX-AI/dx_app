#include "display.h"

using namespace std;
using namespace cv;

list<string> displayObjList[2]{
    {
        "person",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "train",
        "trafficlight",
        "stopsign",
    },
    {
        "person",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "train",
        "trafficlight",
        "stopsign",
        "handbag",
        "airplane",
        "chair",
        "backpack",
        "pottedplant",
        "firehydrant",
    },
};

void DisplayBoundingBox(cv::Mat &frame, vector<BoundingBox> &result, 
    float OriginWidth, float OriginHeight, string frameTitle, string frameText, 
    cv::Scalar UniformColor, vector<Scalar> ObjectColors,
    string OutputImgFile, int DisplayDuration, int Category, bool ImageCenterAligned)
{    
    map<string, int> numObjects;
    float x1, y1, x2, y2, r, w_pad, h_pad;
    float w = (float)frame.cols; /* Target Frame Width */
    float h = (float)frame.rows; /* Target Frame Height */
    int txtBaseline = 0;
    if(OriginWidth!=-1)
    {
        r = min(OriginWidth/w, OriginHeight/h);
        w_pad = ImageCenterAligned?(OriginWidth - w*r)/2.:0;
        h_pad = ImageCenterAligned?(OriginHeight - h*r)/2.:0;
    }
    for(auto &bbox:result)
    {
        if(OriginWidth!=-1)
        {
            x1 = (bbox.box[0] - w_pad)/r;
            x2 = (bbox.box[2] - w_pad)/r;
            y1 = (bbox.box[1] - h_pad)/r;
            y2 = (bbox.box[3] - h_pad)/r;
            x1 = min((float)w, max((float)0.0, x1) );
            x2 = min((float)w, max((float)0.0, x2) );
            y1 = min((float)h, max((float)0.0, y1) );
            y2 = min((float)h, max((float)0.0, y2) );
        }
        else
        {
            int _w = frame.cols;
            int _h = frame.rows;
            x1 = _w * bbox.box[0];
            x2 = _w * bbox.box[2];
            y1 = _h * bbox.box[1];
            y2 = _h * bbox.box[3];
        }
        auto textSize = cv::getTextSize(bbox.labelname, FONT_HERSHEY_SIMPLEX, 0.4, 1, &txtBaseline);
        cv::rectangle(
            frame, Point( x1, y1 ), Point( x2, y2 ), 
            ObjectColors[bbox.label], 2);
            // object_colors[bbox.label], frame.cols>1280?2:1);
        cv::rectangle( 
            frame, 
            Point( x1, y1-textSize.height ), 
            Point( x1 + textSize.width, y1 ), 
            ObjectColors[bbox.label], 
            // dev==0?(object_colors[bbox.label]):Scalar(0, 0, 255), 
            // UniformColor,
            cv::FILLED);
        cv::putText(
            frame, bbox.labelname, Point( x1, y1 ), 
            FONT_HERSHEY_SIMPLEX, 0.4, Scalar(255,255,255));
        if( Category!=-1 && \
            find(displayObjList[Category].begin(), displayObjList[Category].end(), bbox.labelname) \
                != displayObjList[Category].end() )
        {
            numObjects[bbox.labelname]++;
        }
        else
        {
            numObjects["others"]++;
        }
        // cv::putText( frame, txtBuf, Point( x1, 12 + y1 ), FONT_HERSHEY_SIMPLEX, 0.5, object_colors[bbox.label]);
        // bbox.Show();
        // cout << "      (" << x1 << ", " << y1 << ")" << ", (" << x2 << ", " << y2 << ")" << endl;
    }
#if 1
    if(!frameTitle.empty())
    {
        /* Model Caption Area */    
        int textScale = 2;
        int textThickness = 3;
        int boxSizeOffset = 60;    
        auto textSize = cv::getTextSize(frameTitle+" / "+frameText, FONT_HERSHEY_SIMPLEX, textScale, textThickness, &txtBaseline);
        int boxHeight = 70;
        x1 = frame.cols/2. - textSize.width/2. - boxSizeOffset;
        x2 = frame.cols/2. + textSize.width/2. + boxSizeOffset;
        y1 = 40;
        y2 = y1 + boxHeight;
        cv::rectangle(frame, Point(x1, y1), Point(x2, y2), UniformColor, cv::FILLED);
        cv::putText(frame, frameTitle+" / "+frameText, Point(x1 + boxSizeOffset, y1 + textSize.height + 8), 
            FONT_HERSHEY_SIMPLEX, textScale, Scalar(255,255,255), textThickness );
        cv::putText(frame, "A: Dual            1: SSD only           2: YOLO only        3: toggle CAMERA", 
                    Point(5, 15), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1 );
        cv::putText(frame, "4: CCTV            5: Driving            6: Drone            7: Street", 
                    Point(5, 34), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 1 );
        y1 = y2 + 30;
        {
            string txt = "Total: "+ to_string(result.size());
            x1 = 5;
            auto textSize = cv::getTextSize(txt, FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
            cv::rectangle( frame, Point( x1, y1 - textSize.height - 5 ), Point( x1 + textSize.width + 20, y1 + 5 ), 
                            Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, txt, Point(5, y1), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255,255,255), 2 );
            y1 += 25;
        }
        if(Category!=-1)
        {
            for(auto &obj : displayObjList[Category])
            {
                string txt = "   " + obj + ": "+ to_string(numObjects[obj]);
                x1 = 5;
                auto textSize = cv::getTextSize(txt, FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
                cv::rectangle( frame, Point( x1, y1 - textSize.height - 5 ), Point( x1 + textSize.width + 10, y1 + 5 ), 
                                Scalar(0, 0, 0), cv::FILLED);
                cv::putText(frame, txt, Point(5, y1), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2 );
                y1 += 25;
            }
        }
        {
            string txt = "   Others: "+ to_string(numObjects["others"]);
            x1 = 5;
            auto textSize = cv::getTextSize(txt, FONT_HERSHEY_SIMPLEX, 0.6, 2, &txtBaseline);
            cv::rectangle( frame, Point( x1, y1 - textSize.height - 5 ), Point( x1 + textSize.width + 10, y1 + 5 ), 
                            Scalar(0, 0, 0), cv::FILLED);
            cv::putText(frame, txt, Point(5, y1), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255,255,255), 2 );
            y1 += 25;
        }
    }
#endif
    if(DisplayDuration>0)
    {
        imshow("OD Result", frame);
        waitKey(DisplayDuration);
    }
    if(!OutputImgFile.empty())
        imwrite(OutputImgFile, frame);
}

vector<Scalar> GetObjectColors(int type)
{
    vector<Scalar> ObjectColors;
    if(type==0)
    {
        ObjectColors = {
            Scalar(    113    ,    129    ,    39    ),
            Scalar(    133    ,    80    ,    164   ),
            Scalar(    114    ,    122    ,    83    ),
            Scalar(    172    ,    81    ,    99    ),
            Scalar(    104    ,    56    ,    95    ),
            Scalar(    86    ,    84    ,    37        ),
            Scalar(    122    ,    89    ,    14    ),
            Scalar(    65    ,    7    ,    80        ),
            Scalar(    25    ,    102    ,    10        ),
            Scalar(    109    ,    185    ,    90    ),
            Scalar(    132    ,    110    ,    106   ),
            Scalar(    85    ,    158    ,    169       ),
            Scalar(    26    ,    185    ,    188       ),
            Scalar(    17    ,    1    ,    103       ),
            Scalar(    81    ,    144    ,    82        ),
            Scalar(    184    ,    7    ,    92    ),
            Scalar(    155    ,    81    ,    49    ),
            Scalar(    69    ,    177    ,    179       ),
            Scalar(    158    ,    187    ,    93    ),
            Scalar(    73    ,    39    ,    13        ),
            Scalar(    60    ,    50    ,    12        ),
            Scalar(    33    ,    179    ,    16        ),
            Scalar(    165    ,    69    ,    112   ),
            Scalar(    63    ,    139    ,    15        ),
            Scalar(    159    ,    191    ,    33    ),
            Scalar(    32    ,    173    ,    182       ),
            Scalar(    133    ,    113    ,    34    ),
            Scalar(    34    ,    135    ,    90        ),
            Scalar(    86    ,    34    ,    53        ),
            Scalar(    190    ,    35    ,    141   ),
            Scalar(    8    ,    171    ,    6         ),
            Scalar(    112    ,    76    ,    118   ),
            Scalar(    55    ,    60    ,    89        ),
            Scalar(    88    ,    54    ,    15        ),
            Scalar(    181    ,    75    ,    112   ),
            Scalar(    38    ,    147    ,    42        ),
            Scalar(    63    ,    52    ,    138       ),
            Scalar(    149    ,    65    ,    128   ),
            Scalar(    24    ,    103    ,    106       ),
            Scalar(    45    ,    33    ,    168       ),
            Scalar(    135    ,    136    ,    28    ),
            Scalar(    108    ,    91    ,    86    ),
            Scalar(    76    ,    11    ,    52        ),
            Scalar(    189    ,    6    ,    142   ),
            Scalar(    168    ,    81    ,    57    ),
            Scalar(    148    ,    19    ,    55    ),
            Scalar(    89    ,    101    ,    182       ),
            Scalar(    179    ,    65    ,    44    ),
            Scalar(    26    ,    33    ,    1         ),
            Scalar(    26    ,    164    ,    122       ),
            Scalar(    134    ,    63    ,    70    ),
            Scalar(    82    ,    106    ,    137       ),
            Scalar(    52    ,    118    ,    120       ),
            Scalar(    42    ,    74    ,    129       ),
            Scalar(    112    ,    147    ,    182   ),
            Scalar(    50    ,    157    ,    22        ),
            Scalar(    20    ,    50    ,    56        ),
            Scalar(    177    ,    22    ,    2     ),
            Scalar(    106    ,    100    ,    156   ),
            Scalar(    42    ,    35    ,    21        ),
            Scalar(    121    ,    8    ,    13    ),
            Scalar(    28    ,    92    ,    142       ),
            Scalar(    33    ,    118    ,    45        ),
            Scalar(    30    ,    118    ,    105       ),
            Scalar(    124    ,    185    ,    7     ),
            Scalar(    146    ,    34    ,    46    ),
            Scalar(    169    ,    184    ,    105   ),
            Scalar(    5    ,    18    ,    22        ),
            Scalar(    73    ,    71    ,    147       ),
            Scalar(    91    ,    64    ,    181       ),
            Scalar(    184    ,    39    ,    31    ),
            Scalar(    33    ,    179    ,    164       ),
            Scalar(    18    ,    50    ,    96        ),
            Scalar(    106    ,    15    ,    95    ),
            Scalar(    54    ,    68    ,    113       ),
            Scalar(    112    ,    116    ,    136   ),
            Scalar(    130    ,    139    ,    119   ),
            Scalar(    34    ,    139    ,    31        ),
            Scalar(    127    ,    6    ,    66    ),
            Scalar(    2    ,    39    ,    62        ),
            Scalar(    180    ,    99    ,    49    ),
            Scalar(    155    ,    119    ,    49    ),
            Scalar(    183    ,    50    ,    153   ),
            Scalar(    3    ,    38    ,    125       ),
            Scalar(    143    ,    87    ,    129   ),
            Scalar(    40    ,    87    ,    49        ),
            Scalar(    120    ,    62    ,    128   ),
            Scalar(    148    ,    85    ,    73    ),
            Scalar(    118    ,    144    ,    28    ),
            Scalar(    24    ,    9    ,    29        ),
            Scalar(    108    ,    45    ,    175   ),
            Scalar(    64    ,    175    ,    81        ),
            Scalar(    157    ,    19    ,    178   ),
            Scalar(    190    ,    188    ,    74    ),
            Scalar(    2    ,    114    ,    18        ),
            Scalar(    96    ,    128    ,    62        ),
            Scalar(    150    ,    3    ,    21    ),
            Scalar(    95    ,    6    ,    0         ),
            Scalar(    184    ,    20    ,    2     ),
            Scalar(    185    ,    37    ,    122   ),
        };
    }
    if(type==1)
    {
        ObjectColors = vector<Scalar>(80, Scalar(255,0,0));
    }
    else if(type==2)
    {
        ObjectColors = vector<Scalar>(80, Scalar(0,255,0));
    }
    return ObjectColors;
}
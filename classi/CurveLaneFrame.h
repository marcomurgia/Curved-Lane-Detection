#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef _CURVE_LANE_FRAME_H_
#define _CURVE_LANE_FRAME_H_

class CurveLaneFrame
{
private:

    Mat input_frame;
    Mat output_frame;

    void curve_lane();

public:

    CurveLaneFrame(){ };

    virtual ~CurveLaneFrame();

    void FindLane(const Mat &frame, Mat &output_fram_find);
};

#endif

#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

#ifndef _CURVE_LANE_FRAME_H_
#define _CURVE_LANE_FRAME_H_

class CurveLaneFrame
{
private:

    Mat input;
    Mat input_crop;
    Mat warp;
    Mat sobel;
    Mat src;
    Mat u_hist;
    int midpoint;

    float ratio;

    void first_crop_image();
    void warp_image();
    void sobel_operator();
    void second_crop_image();
    void histogram();
    void curve_fit_and_lane();

    Mat output_frame;

public:

    CurveLaneFrame(){ };

    virtual ~CurveLaneFrame();

    void FindLane(const Mat &frame, Mat &output_find_lane);
};

#endif

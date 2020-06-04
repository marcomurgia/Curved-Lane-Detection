#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class CurveLaneFrame
{
private:
    Mat input;
    Mat  input_crop, warp, sobel, src, u_hist;
    float ratio = ((float)input.rows/input.cols);


    void first_crop_image(Mat &input, float &ratio, Mat &input_crop);
    void warp_image(float &ratio, Mat &input_crop,  float &output_factor, Mat &warp);
    void sobel_operator(Mat &warp, Mat &sobel);
    void second_crop_image(float &ratio,Mat &sobel, Mat &src);
    void histogram(Mat &src, Mat &u_hist);
    void curve_fit_and_lane(Mat &input, Mat &u_hist, Mat &src, Mat &input_crop, float &output_factor, Mat &output_frame);

public:
    Mat frame;
    Mat output_frame;
    CurveLaneFrame(){
        //cout <<"prova"<<endl;
        input=frame;
    };

    ~CurveLaneFrame();

};

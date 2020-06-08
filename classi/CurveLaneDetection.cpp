#include <opencv2/opencv.hpp>
#include "CurveLaneFrame.h"

#define average_ratio 0.65 // between 9:16 and 3:4
#define first_factor_9_16 16/3
#define first_factor_3_4 13/3
#define input_factor 1000  //both 9:16 and 3:4
#define output_factor_9_16 600
#define output_factor_3_4 100
#define second_factor_9_16 64/15
#define second_factor_3_4 50/15

using namespace cv;
using namespace std;

void cvPolyfit(Mat &src_x, Mat &src_y, Mat &dst, int order)
{
Mat X = Mat::zeros(src_x.rows, order+1,CV_32FC1);
Mat copy;
for(int i = 0; i <=order;i++)
{
    copy = src_x.clone();
    pow(copy,i,copy);
    Mat M1 = X.col(i);
    copy.col(0).copyTo(M1);
}
Mat X_t, X_inv;
transpose(X,X_t);
Mat temp = X_t*X;
Mat temp2;
invert (temp,temp2);
Mat temp3 = temp2*X_t;
Mat W = temp3*src_y;

dst = W.clone();
}

void inv_warp(Mat &input, Mat &output,  float &output_factor ){

    // Input Quadilateral or Image plane coordinates
    Point2f inputQuad[4], outputQuad[4];

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input
    inputQuad[0] = Point2f( -output_factor,0 );
    inputQuad[1] = Point2f( input.cols+output_factor,0);
    inputQuad[2] = Point2f( input.cols,input.rows);
    inputQuad[3] = Point2f( 0,input.rows);

    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( 0,0 );
    outputQuad[1] = Point2f( input.cols,0);
    outputQuad[2] = Point2f( input.cols+input_factor,input.rows);
    outputQuad[3] = Point2f( -input_factor,input.rows);

    // Get the Perspective Transform Matrix i.e. lambda
    Mat lambda = getPerspectiveTransform( inputQuad, outputQuad );
    warpPerspective(input,output,lambda,output.size() );
}


void CurveLaneFrame::curve_lane()
{

    /// Load the image
    Mat input = this->input_frame;

    /// First CROP image
    float ratio=((float)input.rows/input.cols);
    float factor_roi;
    if(ratio<average_ratio){
        factor_roi = ratio*((float)first_factor_9_16); //9:16
    } else {
        factor_roi = ratio*((float)first_factor_3_4); //3:4
    }

    Rect roi_1;
    roi_1.x = 0;
    roi_1.y = input.size().height- input.size().height/factor_roi;
    roi_1.width = input.size().width;
    roi_1.height = input.size().height/factor_roi;

    Mat input_crop = input(roi_1);

    /// WARP the image
    // Input and Output Quadilateral or Image plane coordinates
    Point2f inputQuad[4], outputQuad[4];

    // The 4 points that select quadilateral on the input , from top-left in clockwise order
    // These four pts are the sides of the rect box used as input
    float  output_factor;
    if(ratio<average_ratio){
        output_factor=output_factor_9_16; //9:16
    } else {
        output_factor=output_factor_3_4; //3:4
    }
    //input_factor=input_factor_both;

    inputQuad[0] = Point2f( 0,0 );
    inputQuad[1] = Point2f( input_crop.cols, 0);
    inputQuad[2] = Point2f( input_crop.cols + input_factor, input_crop.rows);
    inputQuad[3] = Point2f( -input_factor,input_crop.rows);

    // The 4 points where the mapping is to be done , from top-left in clockwise order
    outputQuad[0] = Point2f( -output_factor, 0 );
    outputQuad[1] = Point2f( input_crop.cols+output_factor, 0);
    outputQuad[2] = Point2f( input_crop.cols,input_crop.rows);
    outputQuad[3] = Point2f( 0,input_crop.rows);

    // Get the Perspective Transform Matrix and apply it to the input image
    Mat lambda = getPerspectiveTransform( inputQuad, outputQuad );
    Mat warp;
    warpPerspective(input_crop, warp, lambda, warp.size() );

    /// Sobel operator
    Mat gaussian, gaussian_gray, sobel;

    // Remove noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur(warp, gaussian, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(gaussian, gaussian_gray, COLOR_BGR2GRAY); // Convert the image to grayscale

    // Sobel with Gradient X and Y
    Mat grad_x, grad_y, abs_grad_x, abs_grad_y;

    int scale = 1, delta = 0, ddepth = CV_16S;
    Sobel(gaussian_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(gaussian_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);

    // Converting back to CV_8U
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);

    // Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);

    /// Second CROP image
    float factor_roi_2;
    if(ratio < average_ratio){
        factor_roi_2 = ratio*((float)second_factor_9_16); //9:16
    } else {
        factor_roi_2 = ratio*((float)second_factor_3_4); //3:4
    }

    Rect roi_2;
    roi_2.x = (sobel.size().width/2) - (sobel.size().width/(2*factor_roi_2));
    roi_2.y = 0;
    roi_2.width = sobel.size().width/factor_roi_2;
    roi_2.height = sobel.size().height;

    Mat src = sobel(roi_2);

    /// Draw the histograms
    Mat u_hist;
    for (int i = 0; i<src.cols; i++)    {
        u_hist.push_back(sum(src.col(i))[0]);
    }

    int hist_w = src.size().width, hist_h =src.size().height;
    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar(0,0,0) );

    // Normalize the result to ( 0, histImage.rows )
    normalize(u_hist, u_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    // Draw for each channel
    for( int i = 1; i<src.size().width; i++ )
    {
        line( histImage, Point( (i-1), hist_h - (u_hist.at<double>(i-1)) ),
              Point( (i), hist_h - (u_hist.at<double>(i)) ),
              Scalar( 255,0 , 0), 2, 8, 0  );
    }

    /// Setting for the drawing window
    // Find peaks of left and right halves
    int midpoint = histImage.cols/2;
    int max_f=0, max_f_1=0, max_loc_f,max_loc_f_1;
    int leftx_base, rightx_base;
    for (int i=0; i<midpoint; i++) {
        if (u_hist.at<double>(i)>=max_f){
            max_f=u_hist.at<double>(i);
            max_loc_f=i;
        }
        if (u_hist.at<double>(i+midpoint)>=max_f_1){
            max_f_1=u_hist.at<double>(i+midpoint);
            max_loc_f_1=i+midpoint;
        }
    }

    leftx_base=max_loc_f;
    rightx_base=max_loc_f_1;

    //Set height of windows
    int nwindows=9;
    int window_height=src.rows/nwindows;

    //Identify the x and y positions of all nonzero pixels in the image
    Mat nonZeroCoord;
    //cvtColor(src,src1,COLOR_BGR2GRAY,0);

    // Image filter with threshold
    typedef Vec<uchar, 1> Vec1b;
    Vec1b threshold=30;
    Vec1b temp;

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            temp=src.at< Vec<uchar, 1> >(i,j);
            if(temp[0]<threshold[0])
                src.at< Vec<uchar, 1> >(i,j)=0;
        }
    }

    findNonZero(src , nonZeroCoord);
    int dim_nonzero = nonZeroCoord.rows;
    int nonzerox[dim_nonzero], nonzeroy[dim_nonzero];

    for (int i = 0; i < dim_nonzero; i++ ) {
        nonzerox[i]=nonZeroCoord.at<Point>(i).x ;
        nonzeroy[i]=nonZeroCoord.at<Point>(i).y ;
    }

    // Current positions to be updated for each window
    int  leftx_current = leftx_base, rightx_current = rightx_base;

    // Create empty lists to receive left and right lane pixel indices
    int i_lane_left=0, i_lane_right=0, dim_tot=dim_nonzero*(nwindows);
    int *left_lane_inds, *right_lane_inds;
    left_lane_inds  = (int *) malloc(dim_tot * sizeof(int));
    right_lane_inds = (int *) malloc(dim_tot * sizeof(int));

    //Step through the windows one by one
    int margin=src.cols/11; // half window width
    Mat src_r = Mat::zeros(src.rows, src.cols, CV_32FC3);

    /// Draw windows

    for (int n=0; n<nwindows; n++) {

        int win_y_low = src.rows - (n+1)*window_height;
        int win_y_high = src.rows - n*window_height;
        int win_xleft_low = leftx_current - margin;
        int win_xleft_high = leftx_current + margin;
        int win_xright_low = rightx_current - margin;
        int win_xright_high = rightx_current + margin;

        //if draw_windows==True
        Point p1(win_xleft_low, win_y_low), p2(win_xleft_high, win_y_high);
        Point p3(win_xright_low, win_y_low), p4(win_xright_high, win_y_high);

        rectangle(src_r, p1, p2, Scalar(0, 255, 0), 2, 8, 0);
        rectangle(src_r, p3, p4, Scalar(0, 255, 0), 2, 8, 0);

        // Identify the nonzero pixels in x and y within the window
        int i_left=0,i_right=0;
        int good_left_inds[dim_nonzero],good_right_inds[dim_nonzero];

        for (int i=0; i<dim_nonzero; i++) {
            if ((nonzeroy[i] >= win_y_low) & (nonzeroy[i] < win_y_high) &
                    (nonzerox[i] >= win_xleft_low) &  (nonzerox[i] < win_xleft_high)){
                good_left_inds[i_left]=i;
                left_lane_inds[i_lane_left]=i;
                i_left++; i_lane_left++;
            }

            if ((nonzeroy[i] >= win_y_low) & (nonzeroy[i] < win_y_high) &
                    (nonzerox[i] >= win_xright_low) &  (nonzerox[i] < win_xright_high)){
                good_right_inds[i_right]=i;
                right_lane_inds[i_lane_right]=i;
                i_right++;i_lane_right++;
            }
        }

        // If you found > minpix pixels, recenter next window on their mean position
        int line_cost=100; // solution to the dotted line
        int minpix=1, s_left=0, s_right=0;

        if (i_left > minpix){
            for (int i=0; i<i_left; i++)
                s_left+=nonzerox[good_left_inds[i]];
        }
        if(s_left>line_cost)
            leftx_current=s_left/i_left;

        if (i_right > minpix){
            for (int i=0; i<i_right; i++)
                s_right+=nonzerox[good_right_inds[i]];
        }
        if(s_right>line_cost)
            rightx_current=s_right/i_right;
    }

    //Extract left and right line pixel positions
    float leftx[i_lane_left],lefty[i_lane_left];
    float rightx[i_lane_right],righty[i_lane_right];
    Point pt1,pt2;

    for (int i=0; i<i_lane_left; i++) {
        leftx[i]=nonzerox[left_lane_inds[i]];
        pt1.x=leftx[i];

        lefty[i]=nonzeroy[left_lane_inds[i]];
        pt1.y=lefty[i];

        circle(src_r, pt1, 0.1, Scalar(255,0,0), 1, 8, 0);
    }

    for (int i=0; i<i_lane_right; i++) {
        rightx[i] = nonzerox[right_lane_inds[i]];
        pt2.x = rightx[i];

        righty[i] = nonzeroy[right_lane_inds[i]];
        pt2.y = righty[i];

        circle(src_r, pt2, 0.1, Scalar(0,0,255), 1, 8, 0);
    }

    /// Curve fit
    Mat leftx_m = Mat((i_lane_left), 1, CV_32FC1, &leftx);
    Mat lefty_m = Mat((i_lane_left), 1, CV_32FC1 ,&lefty);
    Mat rightx_m = Mat((i_lane_right), 1, CV_32FC1, &rightx);
    Mat righty_m = Mat((i_lane_right), 1, CV_32FC1 ,&righty);

    int fit_order = 2;
    Mat coef_left(fit_order+1,1,CV_32FC1);
    Mat coef_right(fit_order+1,1,CV_32FC1);

    cvPolyfit(lefty_m, leftx_m, coef_left, fit_order);
    cvPolyfit(righty_m, rightx_m, coef_right, fit_order);

    vector<Point> pt_l(src.rows),pt_r(src.rows);
    vector<Point> pt_r_inv(src.rows);
    for (int i=0; i<src.rows; i++) {
        pt_l[i].y=i;
        pt_l[i].x=(coef_left.at<float>(2))*i*i+(coef_left.at<float>(1))*i+(coef_left.at<float>(0));

        pt_r[i].y=i;
        pt_r[i].x=(coef_right.at<float>(2))*i*i+(coef_right.at<float>(1))*i+(coef_right.at<float>(0));
    }
    polylines(src_r, pt_l, false, Scalar(0,200,255), 3, 8,0);
    polylines(src_r, pt_r, false, Scalar(0,200,255), 3, 8,0);

    /// Fill lane area
    vector<vector<Point> > vpts, vpts1;
    vpts.push_back(pt_r);
    vpts1.push_back(pt_l);

    Mat lane_area( src.size(), CV_8UC3);
    fillPoly(lane_area, vpts, Scalar(255,200,150),8,0);
    floodFill(lane_area, Point(0,0), Scalar(255,200,150));

    Mat lane_area1( src.size(), CV_8UC3);
    fillPoly(lane_area1, vpts1, Scalar(255,200,150),8,0);
    floodFill(lane_area1, Point(0,0), Scalar(255,200,150));

    Mat area_final=lane_area-lane_area1;

    Mat curve_fit( src.size(), CV_8UC3);
    polylines(curve_fit, pt_l, false, Scalar(0,255,255), 3, 8,0);
    polylines(curve_fit, pt_r, false, Scalar(0,255,255), 3, 8,0);

    curve_fit=curve_fit+area_final;

    // Traslate polylines
    int width_first=input.cols;
    int traslate_x=width_first/2-curve_fit.cols/2;
    Mat curve_fit_traslate, warpGround_x;
    warpGround_x =(Mat_<float>(2,3) << 1, 0, traslate_x, 0, 1, 0);
    warpAffine(curve_fit,curve_fit_traslate, warpGround_x,Size(input.cols,input_crop.rows));

    Mat inv_curve_fit;
    inv_warp(curve_fit_traslate, inv_curve_fit,  output_factor);

    int traslate_y=input.rows-inv_curve_fit.rows;
    Mat inv_curve_fit_2, warpGround_y;
    warpGround_y =(Mat_<float>(2,3) << 1, 0, 0, 0, 1, traslate_y);
    warpAffine(inv_curve_fit,inv_curve_fit_2, warpGround_y,Size(input.cols,input.rows));

    Mat final_image=inv_curve_fit_2+input;

    this->output_frame=final_image;
}


CurveLaneFrame::~CurveLaneFrame(){}

void CurveLaneFrame::FindLane(const Mat &frame, Mat &output_fram_find )
{
    this->input_frame = frame;
    this->curve_lane();
    output_fram_find = this->output_frame;
}

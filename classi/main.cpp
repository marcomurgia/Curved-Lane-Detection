#include <opencv2/opencv.hpp>
#include "CurveLaneFrame.h"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    // Initialize VideoCapture and open the default webcam
    VideoCapture cap;
    cap.open(0);

    int deviceID = 0;
    // Automatically reads api by default
    int apiID = CAP_ANY;
    // Open the room with the selected api
    cap.open(deviceID + apiID);
    //Check if it is open, if not mold error
    if (!cap.isOpened()) {
      cerr << "ERROR! Unable to open camera\n";
      return -1;
    }
    cout << "Start" << endl
         << "Press any key to terminate" << endl;


    CurveLaneFrame lane_identification;

    for (;;)
    {
      Mat frame;
      Mat output_frame;

      // Wait for the new frame from the webcam and keep it in frame
      cap.read(frame);

      if (frame.empty()) {
         cerr << "ERROR! Blank frame grabbed\n";
         break;
      }

      lane_identification.FindLane(frame, output_frame);

      waitKey(5);

      // Watch the live webcam
      imshow("Live", output_frame);

      if (waitKey(5) >= 0)
          break;
    }

    return 0;
}


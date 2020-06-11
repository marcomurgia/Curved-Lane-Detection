#include <opencv2/opencv.hpp>
#include "CurveLaneFrame.h"
#include <time.h>
#include <unistd.h>

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
    double media = 0.0;
    int n = 0;

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

      clock_t start,end;
      double tempo;
      start=clock();

      lane_identification.FindLane(frame, output_frame);
      //usleep(1000000);
      end=clock();
      tempo=((double)(end-start))/CLOCKS_PER_SEC;

      cout<<tempo<<endl;

      media += tempo;
      n++;

      // Watch the live webcam
      imshow("Live", output_frame);

      if (waitKey(5) >= 0)
          break;
    }

    media /= n;
    cout<<"media = "<<media<<endl;

    return 0;
}


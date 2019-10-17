#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  //create a black 256x256, 8bit, 3channel BGR image in a matrix container
  Mat image = imread("mandrill0.jpg", 1);

  //set pixels to create colour pattern
  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     uchar pixelBlue = image.at<Vec3b>(y,x)[0];
     uchar pixelGreen = image.at<Vec3b>(y,x)[1];
     uchar pixelRed = image.at<Vec3b>(y,x)[2];

     image.at<Vec3b>(y,x)[0] = pixelRed;
     image.at<Vec3b>(y,x)[1] = pixelBlue;
     image.at<Vec3b>(y,x)[2] = pixelGreen;
     

 } }


  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", image);

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image
  image.release();

  return 0;
}

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  //create a black 256x256, 8bit, 3channel BGR image in a matrix container
  Mat image = imread("mandrill1.jpg", 1);
  Mat image_out(image.cols, image.cols, CV_8UC3, Scalar(0, 0, 0));
  //set pixels to create colour pattern
  for(int y=0; y<image.rows; y++) {
   for(int x=0; x<image.cols; x++) {
     int yshift = y - 30;
     if (yshift < 0) {
       yshift = image.rows + yshift;
     }
     int xshift = x - 30;
     if (xshift < 0) {
       xshift = image.cols + xshift;
     }
     uchar pixelRed = image.at<Vec3b>(yshift,xshift)[2];

     image_out.at<Vec3b>(y,x)[2] = pixelRed;

     image_out.at<Vec3b>(y,x)[0] = image.at<Vec3b>(y,x)[0];
     image_out.at<Vec3b>(y,x)[1] = image.at<Vec3b>(y,x)[1];



 } }


  //construct a window for image display
  namedWindow("Display window", CV_WINDOW_AUTOSIZE);

  //visualise the loaded image in the window
  imshow("Display window", image_out);

  //wait for a key press until returning from the program
  waitKey(0);

  //free memory occupied by image
  image.release();

  return 0;
}

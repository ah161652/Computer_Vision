#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int main() {

  // Read image from file
  Mat singleblur_image = imread("reverted_blur.png", 1);

  Mat doubleblur_image = imread("blur.jpg", 1);

  Mat gray_image = imread("reverted_blur.png", 1);
  Mat gray_image_blur = imread("blur.jpg", 1);
  // cvtColor(singleblur_image, gray_image, CV_BGR2GRAY);
  // cvtColor(doubleblur_image, gray_image_blur, CV_BGR2GRAY);

  for(int y=0; y<gray_image.rows; y++) {
   for(int x=0; x<gray_image.cols; x++) {
     int original = gray_image.at<uchar>(y,x);
     int doubleblur = gray_image_blur.at<uchar>(y,x);
     int final_value = 2*original - doubleblur;
     if (final_value > 255) final_value = 255;
     if (final_value < 0) final_value = 0;

     gray_image.at<uchar>(y,x) = final_value;

     //std::cout << (2*gray_image.at<int>(x,y) - doubleblur_image.at<int>(x,y));
     // int toprow = (input_image.at<uchar>(mod(y-1), mod(x-1))) + (input_image.at<uchar>(mod(y-1), mod(x))) + (input_image.at<uchar>(mod(y-1), mod(x+1)));
     //
     // int midrow = (input_image.at<uchar>(mod(y), mod(x))) + (input_image.at<uchar>(mod(y), mod(x-1))) + (input_image.at<uchar>(mod(y), mod(x+1)));
     //
     // int bottomrow = (input_image.at<uchar>(mod(y+1), mod(x-1))) + (input_image.at<uchar>(mod(y+1), mod(x))) + (input_image.at<uchar>(mod(y+1), mod(x+1)));
     //
     // int total = (toprow + midrow + bottomrow) - ;
     //
     // input_image.at<uchar>(y, x) = total;



  }}

  //Save thresholded image
  imwrite("final.jpg", gray_image);

  return 0;
}

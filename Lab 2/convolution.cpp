#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup

using namespace cv;

int mod(int a){
  int b = 512;
  return (a%b+b)%b;
}
int main() {

  // Read image from file
  Mat gray_image = imread("mandrill.jpg", 1);
  for(int y=0; y<gray_image.rows; y++) {
   for(int x=0; x<gray_image.cols; x++) {

     int toprow = (gray_image.at<uchar>(mod(y-1), mod(x-1))) + (gray_image.at<uchar>(mod(y-1), mod(x))) + (gray_image.at<uchar>(mod(y-1), mod(x+1)));

     int midrow = (gray_image.at<uchar>(mod(y), mod(x))) + (gray_image.at<uchar>(mod(y), mod(x-1))) + (gray_image.at<uchar>(mod(y), mod(x+1)));

     int bottomrow = (gray_image.at<uchar>(mod(y+1), mod(x-1))) + (gray_image.at<uchar>(mod(y+1), mod(x))) + (gray_image.at<uchar>(mod(y+1), mod(x+1)));

     int total = (toprow + midrow + bottomrow) / 9;

     gray_image.at<uchar>(y, x) = total;



  }}

  //Save thresholded image
  imwrite("conv.jpg", gray_image);

  return 0;
}

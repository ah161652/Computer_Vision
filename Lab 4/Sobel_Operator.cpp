#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
using namespace cv;

int mod(int a, int b){
  return (a%b+b)%b;
}

int main() {
  // Read image from file
  Mat input_image = imread("coins1.png", 1);
  Mat input_gray;

  cvtColor(input_image, input_gray, CV_BGR2GRAY);
  // Mat dx = input_gray;
  // Mat dy = input_gray;
  int cols = input_gray.cols;
  int rows = input_gray.rows;
  Mat dx(rows, cols, CV_8UC1, Scalar(0));
  Mat dy(rows, cols, CV_8UC1, Scalar(0));
  Mat grad(rows, cols, CV_8UC1, Scalar(0));
  Mat orientation(rows, cols, CV_8UC1, Scalar(0));

  // int[3][3] Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  // int[3][3] Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {

     int toprow = (input_gray.at<uchar>(mod(y-1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y-1,rows), mod(x,cols)))*0 + (input_gray.at<uchar>(mod(y-1,rows), mod(x+1,cols)))*1;

     int midrow = (input_gray.at<uchar>(mod(y,rows), mod(x,cols)))*(-2) + (input_gray.at<uchar>(mod(y,rows), mod(x-1,cols)))*0 + (input_gray.at<uchar>(mod(y,rows), mod(x+1,cols)))*2;

     int bottomrow = (input_gray.at<uchar>(mod(y+1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y+1,rows), mod(x,cols)))*0 + (input_gray.at<uchar>(mod(y+1,rows), mod(x+1,cols)))*1;

     int total_dx = (toprow + midrow + bottomrow);

     if (total_dx > 255) total_dx = 255;
     if (total_dx < 0) total_dx = 0;

     dx.at<uchar>(y, x) = total_dx;

        }
    }

imwrite("coins_dx.jpg", dx);

for(int y=0; y<input_gray.rows; y++) {
 for(int x=0; x<input_gray.cols; x++) {

   int toprow_dy = (input_gray.at<uchar>(mod(y-1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y-1,rows), mod(x,cols)))*(-2) + (input_gray.at<uchar>(mod(y-1,rows), mod(x+1,cols)))*(-1);

   int midrow_dy = (input_gray.at<uchar>(mod(y,rows), mod(x,cols)))*(0) + (input_gray.at<uchar>(mod(y,rows), mod(x-1,cols)))*0 + (input_gray.at<uchar>(mod(y,rows), mod(x+1,cols)))*0;

   int bottomrow_dy = (input_gray.at<uchar>(mod(y+1,rows), mod(x-1,cols)))*(1) + (input_gray.at<uchar>(mod(y+1,rows), mod(x,cols)))*2 + (input_gray.at<uchar>(mod(y+1,rows), mod(x+1,cols)))*1;

   int total_dy = (toprow_dy + midrow_dy + bottomrow_dy);
   if (total_dy > 255) total_dy = 255;
   if (total_dy < 0) total_dy = 0;
   dy.at<uchar>(y, x) = total_dy;

      }
  }

  imwrite("coins_dy.jpg", dy);

  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {

     double total_grad = sqrt(pow(dy.at<uchar>(y,x),2) + pow(dx.at<uchar>(y,x),2));
     if (total_grad > 70) total_grad = 255;
     if (total_grad < 0) total_grad = 0;
     grad.at<uchar>(y, x) = total_grad;

        }
    }

  imwrite("coins_grad.jpg", grad);

  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {
     double div = dy.at<uchar>(y,x)/(dx.at<uchar>(y,x)+1);
     double final_orientation = atan(div);
     if (final_orientation > 255) final_orientation = 255;
     if (final_orientation < 0) final_orientation = 0;
     orientation.at<uchar>(y,x) = final_orientation;
        }
    }
  imwrite("coins_orientation.jpg",orientation);
  return 0;

}

#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
using namespace cv;

int mod(int a, int b){
  return (a%b+b)%b;
}

Mat gradientDirection( Mat input_image );
Mat gradientMagnitude( Mat input_image );


int main() {
  // Read image from file
  Mat input_image = imread("coins1.png", 1);
  Mat input_gray;

  cvtColor(input_image, input_gray, CV_BGR2GRAY);


  int cols = input_gray.cols;
  int rows = input_gray.rows;
  Mat dx(rows, cols,  CV_64FC1, Scalar(0));
  Mat dx_normalize(rows, cols,  CV_8UC1, Scalar(0));
  Mat dy(rows, cols, CV_64FC1, Scalar(0));
  Mat dy_normalize(rows, cols,  CV_8UC1, Scalar(0));
  Mat grad(rows, cols, CV_64FC1, Scalar(0));
  Mat grad_normalize(rows, cols, CV_8UC1, Scalar(0));
  Mat orientation(rows, cols, CV_64FC1, Scalar(0));
  Mat orientation_normalize(rows, cols, CV_8UC1, Scalar(0));

  Mat threshold_magnitude(rows, cols,  CV_8UC1, Scalar(0));

  // int[3][3] Gx = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
  // int[3][3] Gy = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {

     double toprow = (input_gray.at<uchar>(mod(y-1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y-1,rows), mod(x,cols)))*0 + (input_gray.at<uchar>(mod(y-1,rows), mod(x+1,cols)))*1;

     double midrow = (input_gray.at<uchar>(mod(y,rows), mod(x,cols)))*(-2) + (input_gray.at<uchar>(mod(y,rows), mod(x-1,cols)))*0 + (input_gray.at<uchar>(mod(y,rows), mod(x+1,cols)))*2;

     double bottomrow = (input_gray.at<uchar>(mod(y+1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y+1,rows), mod(x,cols)))*0 + (input_gray.at<uchar>(mod(y+1,rows), mod(x+1,cols)))*1;

     double total_dx = (toprow + midrow + bottomrow);


     dx.at<double>(y, x) = total_dx;

        }
    }
normalize(dx, dx_normalize, 0, 255,NORM_MINMAX, CV_8UC1);
imwrite("coins_dx.jpg", dx_normalize);

for(int y=0; y<input_gray.rows; y++) {
 for(int x=0; x<input_gray.cols; x++) {

   double toprow_dy = (input_gray.at<uchar>(mod(y-1,rows), mod(x-1,cols)))*(-1) + (input_gray.at<uchar>(mod(y-1,rows), mod(x,cols)))*(-2) + (input_gray.at<uchar>(mod(y-1,rows), mod(x+1,cols)))*(-1);

   double midrow_dy = (input_gray.at<uchar>(mod(y,rows), mod(x,cols)))*(0) + (input_gray.at<uchar>(mod(y,rows), mod(x-1,cols)))*0 + (input_gray.at<uchar>(mod(y,rows), mod(x+1,cols)))*0;

   double bottomrow_dy = (input_gray.at<uchar>(mod(y+1,rows), mod(x-1,cols)))*(1) + (input_gray.at<uchar>(mod(y+1,rows), mod(x,cols)))*2 + (input_gray.at<uchar>(mod(y+1,rows), mod(x+1,cols)))*1;

   double total_dy = (toprow_dy + midrow_dy + bottomrow_dy);

   dy.at<double>(y, x) = total_dy;

      }
  }
  normalize(dy, dy_normalize, 0, 255,NORM_MINMAX, CV_8UC1);
  imwrite("coins_dy.jpg", dy_normalize);

  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {

     double total_grad = sqrt(pow(dy.at<double>(y,x),2) + pow(dx.at<double>(y,x),2));
     grad.at<double>(y, x) = total_grad;

        }
    }
  normalize(grad, grad_normalize, 0, 255,NORM_MINMAX, CV_8UC1);
  imwrite("coins_grad.jpg", grad_normalize);

  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {
     double val_y = dy.at<double>(y,x);
     double val_x = dx.at<double>(y,x);
     double final_orientation = atan2(val_y,val_x);
     orientation.at<double>(y,x) = final_orientation;
        }
    }
  normalize(orientation, orientation_normalize, 0, 255,NORM_MINMAX, CV_8UC1);
  imwrite("coins_orientation.jpg",orientation_normalize);


  for(int y=0; y<input_gray.rows; y++) {
   for(int x=0; x<input_gray.cols; x++) {
     if (grad_normalize.at<uchar>(y,x) >= 65) threshold_magnitude.at<uchar>(y,x) = 255;
     else threshold_magnitude.at<uchar>(y,x) = 0;
        }
    }
    imwrite("threshold.jpg",threshold_magnitude);

  // vector<Vec3d> circles;
  // HoughCircles(threshold_magnitude, circles, CV_HOUGH_GRADIENT, 1, threshold_magnitude.rows/8,200,100,0,0);
  // for(size_t i = 0; i<circles.size(); i++){
  //   Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
  //   int radius = cvRound(circles[i][2]);
  //   circle(input_image, center, radius, Scalar(0,0,255), 3, 8, 0);
  // }
  // imwrite("final.jpg",input_image);
return 0;
}

Mat gradientDirection( Mat input_image ){

}
Mat gradientMagnitude( Mat input_image ){

}

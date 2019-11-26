#include "Sobel.h"
#include <stdio.h>
#include <opencv/cv.h>        //you may need to
#include <opencv/highgui.h>   //adjust import locations
#include <opencv/cxcore.h>    //depending on your machine setup
#include <cmath>
using namespace cv;

Mat gradientMagnitude( Mat input_image ){
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  Mat gradMag;

  int scale = 1;
  int delta = 0;

  /// Gradient X
  Sobel( input_image, grad_x, CV_64FC1, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  Sobel( input_image, grad_y, CV_64FC1, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  convertScaleAbs( grad_x, abs_grad_x );
  convertScaleAbs( grad_y, abs_grad_y );

  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradMag );

  return gradMag;
}


Mat gradientDirection( Mat input_image ){
  Mat grad_x, grad_y;
  Mat gradOrientation(input_image.rows, input_image.cols, CV_64FC1, Scalar(0));
  Mat abs_grad_x, abs_grad_y;
  int scale = 1;
  int delta = 0;

  /// Gradient X
  Sobel( input_image, grad_x, CV_64FC1, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  Sobel( input_image, grad_y, CV_64FC1, 0, 1, 3, scale, delta, BORDER_DEFAULT );

  convertScaleAbs( grad_x, abs_grad_x );
  convertScaleAbs( grad_y, abs_grad_y );

  for(int y = 0; y < grad_y.rows; y++) {
   for(int x = 0; x < grad_x.cols; x++) {

     double val_y = grad_y.at<double>(y,x);
     double val_x = grad_x.at<double>(y,x);
     double final_orientation = atan2(val_y,val_x);

     gradOrientation.at<double>(y,x) = final_orientation;
        }
    }

  return gradOrientation;
}

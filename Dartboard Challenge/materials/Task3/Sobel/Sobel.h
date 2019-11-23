#ifndef SOBEL_H
#define SOBEL_H

#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <cstdlib>

using namespace std;
using namespace cv;

Mat gradientMagnitude( cv::Mat input_image );
Mat gradientDirection( cv::Mat input_image );


#endif

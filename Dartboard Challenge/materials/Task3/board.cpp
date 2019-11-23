/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - board.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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

/** Function Headers */
void detectAndDisplayViola( Mat frame);
void groundTruth (Mat frame, int fileNumber);
double tp_count(vector<Rect> boards, int fileNumber);
double f1Score(double count_true_positive, double	count_false_positive, double false_negative);
Mat gradientMagnitude( Mat input_image );
Mat gradientDirection( Mat input_image );
Mat houghSpace(Mat gradientMagnitudeImage, Mat gradientDirectionImage );
Mat thresholdValue(Mat image, int value);
Mat drawLines(Mat houghSpace, Mat image);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
std::vector<Rect> boards;
const double pi = 3.14159265358979323846;

int coordinates[20][4] = {{459,36,573,172},{218,152,369,299},{112,109,178,174},{332,157,380,211},{207,120,396,281},{446,151,529,235},{219,124,266,172},{271,187,386,299},{73,263,117,327},{857,231,943,321},{227,73,406,253},{102,116,175,198},{590,139,633,201},{921,158,947,206},{182,115,226,166},{162,96,208,198},{288,138,388,236},{135,116,229,212},{1002,112,1096,206},{171,75,273,178}};
int board_count[16]={1,1,1,1,1,1,1,1,2,1,3,1,1,1,2,1};
int first_board_index[16] = {0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,19};


/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image & Convert to a grey
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
  cvtColor(frame, input_gray, CV_BGR2GRAY);

	//Get file names and numbers for groundTruth comparisons later
	String fileName = argv[1];
	int fileNumber = atoi(argv[2]);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading Cascade\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplayViola(frame);

	// 4. Get required images for HOUGH TRANSFORMS
	Mat gradientMagnitudeImage = gradientMagnitude(input_gray);
	Mat gradientDirectionImage = gradientDirection(input_gray);

	//Apply threshold to magnitude image
	Mat thresholdMagnitude;
	thresholdMagnitude = thresholdValue(gradientMagnitudeImage, 160);
	imwrite( "threshold.jpg", thresholdMagnitude );

	//Get hough Space
	Mat houghSpaceIMG;
	houghSpaceIMG = houghSpace(thresholdMagnitude, gradientDirectionImage);
	imwrite( "houghSpace.jpg", houghSpaceIMG );

	//Try and draw lines
	Mat houghLines;
	houghLines = drawLines(houghSpaceIMG,frame);
	imwrite( "houghLines.jpg", houghLines );

	//Draw all ground truth rectangles
	groundTruth(frame, fileNumber);

	//Call tpr function with full set of detected faces
	double count_true_positive = tp_count(boards, fileNumber);
	double false_negative = board_count[fileNumber] - count_true_positive;
	double count_false_positive = boards.size()-count_true_positive;

	//Calculate F1 score with true positive count and false positive count
	double f1 = f1Score(count_true_positive, count_false_positive, false_negative);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplayViola( Mat frame){
	Mat frame_gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(300,300) );

       // 3. Print number of Faces found
	std::cout << "Number of boards detected: " << boards.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < boards.size(); i++ )
	{
		rectangle(frame, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
	}
}

void groundTruth (Mat frame, int fileNumber){
	//dependant on file name draw all True boards using global arrays
	//If no boards dont draw any
	if (board_count[fileNumber] == 0) {
		return;
	}
	//Gather indexes specific for that file
	int i = first_board_index[fileNumber];
	int limit = board_count[fileNumber] + i;

	for (i; i < limit; i++) {
		rectangle(frame, Point(coordinates[i][0], coordinates[i][1]), Point(coordinates[i][2], coordinates[i][3]), Scalar(0,0,255), 2);
	}
}

double iou(Rect_<int> current_true, Rect_<int> current_detected){
	//Uses Rect funtions to calculate the intersection
	Rect_<int> intersection = current_true & current_detected;

	//Union is trivial if intersection and total area is known
	double union_of_area = current_detected.area() + current_true.area() - intersection.area();
	double iou_val = (intersection.area())/(union_of_area);

	return iou_val;
}


double tp_count(vector<Rect> boards, int fileNumber){
	double count_true_positive = 0;
	double tpr;
	//If no ground truth faces, true positive is N/A;
	if (board_count[fileNumber] == 0) {
		std::cout << "TPR: NaN " << '\n';
		return std::numeric_limits<double>::quiet_NaN();
	}
	//Gather indexes specific for that file
	int i = first_board_index[fileNumber];
	int limit = board_count[fileNumber] + i;

	//Iterate through all true positives and detected boards and calculate intersection/union.
	for (i; i < limit; i++) {
		for( int j = 0; j < boards.size(); j++ ){
				Rect_<int> current_true(coordinates[i][0],coordinates[i][1],coordinates[i][2]-coordinates[i][0],coordinates[i][3]-coordinates[i][1]);
				if(iou(current_true,boards[j]) > 0.45) count_true_positive++;
		}
	}

	//Calculate true positive ratio
	tpr = count_true_positive/board_count[fileNumber];

	std::cout << "TPR:  "<< tpr << '\n';
	return count_true_positive;
}

//Calculate the F1 Score
double f1Score(double count_true_positive, double	count_false_positive, double false_negative){
	double f1;

	//Required variables using TP and FP count
	double precision = (count_true_positive)/(count_true_positive + count_false_positive);
	double recall = (count_true_positive)/(count_true_positive + false_negative);
	if (precision + recall == 0) {
		f1 = 0;
	}
	else{
		f1 = 2 * (precision*recall) / (precision + recall);
	}

	std::cout << "F1 Score:  "<< f1 << '\n';
	return f1;
}

Mat gradientMagnitude(cv::Mat input_image ){
  Mat dx, dy;
  Mat abs_dx, abs_dy;
  Mat magnitudeIMG;

  /// Gradient X
  Sobel( input_image, dx, CV_64FC1, 1, 0, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( dx, abs_dx );
  /// Gradient Y
  Sobel( input_image, dy, CV_64FC1, 0, 1, 3, 1, 0, BORDER_DEFAULT );
	convertScaleAbs( dy, abs_dy );

	addWeighted( abs_dx, 0.5, abs_dy, 0.5, 0, magnitudeIMG );
  return magnitudeIMG;
}


Mat gradientDirection(cv::Mat input_image ){
  Mat grad_x, grad_y;
  Mat gradOrientation(input_image.rows, input_image.cols, CV_64FC1, Scalar(0));
  Mat abs_grad_x, abs_grad_y;
  int scale = 1;
  int delta = 0;

  /// Gradient X
  Sobel( input_image, grad_x, CV_64FC1, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  /// Gradient Y
  Sobel( input_image, grad_y, CV_64FC1, 0, 1, 3, scale, delta, BORDER_DEFAULT );

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

Mat thresholdValue(cv::Mat image, int value){
	for(int y=0; y<image.rows; y++) {
	 for(int x=0; x<image.cols; x++) {
		 if (image.at<uchar>(y,x) >= value) image.at<uchar>(y,x) = 255;
		 else image.at<uchar>(y,x) = 0;
				}
		}
		return image;
}

Mat houghSpace(Mat gradientMagnitudeImage, Mat gradientDirectionImage ){
	Mat houghSpace(2*(gradientMagnitudeImage.cols + gradientMagnitudeImage.rows), 360, CV_32SC1, Scalar(0));

	for (size_t y = 0; y < gradientMagnitudeImage.rows; y++) {
		for (size_t x = 0; x < gradientMagnitudeImage.cols; x++) {
			if (gradientMagnitudeImage.at<uchar>(y,x) == 255) {
				for (size_t pheta = 0; pheta < 360; pheta++) {
					double ro = x*cos(pheta*pi/180) + y*sin(pheta*pi/180) + gradientMagnitudeImage.rows + gradientMagnitudeImage.cols;
					houghSpace.at<int>(ro,pheta)++;
				}
			}
		}
	}
	//Normalize then return the houghspace
	normalize(houghSpace, houghSpace, 0, 255, NORM_MINMAX, CV_32SC1);
	return houghSpace;
}

Mat drawLines(Mat houghSpace, Mat image){
	for (size_t y = 0; y < houghSpace.rows; y++) {
		for (size_t x = 0; x < houghSpace.cols; x++) {
			if (houghSpace.at<int>(y,x) > 160) {
				int ro = y - image.rows - image.cols;
				Point p1, p2;
				double x0 = cos(x) * ro;
				double y0 = sin(x) * ro;
				p1.x = x0+1000*(-sin(x));
				p1.y = y0+1000*cos(x);
				p2.x = x0-1000*(-sin(x));
				p2.y = y0-1000*cos(x);
				line(image, p1, p2, Scalar(0,0,255), 2);
			}
		}
	}
	return image;
}

/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - board.cpp Part 3
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
void detectViola( Mat frame);
void groundTruth (Mat frame, int fileNumber);
double tp_count(vector<Rect> boards, int fileNumber);
double f1Score(double count_true_positive, double	count_false_positive, double false_negative);
Mat gradientMagnitude( Mat input_image );
Mat gradientDirection( Mat input_image );
Mat houghSpaceLine(Mat gradientMagnitudeImage, Mat gradientDirectionImage );
Mat houghSpaceCircle(Mat gradientMagnitudeImage, Mat gradientDirectionImage );
Mat thresholdValue(Mat image, int value);
void getLines(Mat houghSpace, Mat image);
void getCircles(Mat houghSpace);
Mat combinationFunc(Mat frame, Mat houghSpaceIMG, Mat houghSpaceIMGCircle);
void intersectionFunc();

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
std::vector<Rect> boards;
std::vector<Rect> trueBoards;
const double pi = 3.14159265358979323846;

int coordinates[20][4] = {{459,36,573,172},{218,152,369,299},{112,109,178,174},{332,157,380,211},{207,120,396,281},{446,151,529,235},{219,124,266,172},{271,187,386,299},{73,263,117,327},{857,231,943,321},{227,73,406,253},{102,116,175,198},{590,139,633,201},{921,158,947,206},{182,115,226,166},{162,96,208,198},{288,138,388,236},{135,116,229,212},{1002,112,1096,206},{171,75,273,178}};
int board_count[16]={1,1,1,1,1,1,1,1,2,1,3,1,1,1,2,1};
int first_board_index[16] = {0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,19};
int lines[10][4];
int circles[10][3];
int intersections[100][2];
int intersectionCount = 0;

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

	// 4. Get required images for HOUGH TRANSFORMS
	Mat gradientMagnitudeImage = gradientMagnitude(input_gray);
	Mat gradientDirectionImage = gradientDirection(input_gray);

	//Apply threshold to magnitude image
	Mat thresholdMagnitude;
	thresholdMagnitude = thresholdValue(gradientMagnitudeImage, 140);
	//imwrite( "threshold.jpg", thresholdMagnitude );

	//Get hough Space
	Mat houghSpaceIMG;
	houghSpaceIMG = houghSpaceLine(thresholdMagnitude, gradientDirectionImage);
	//imwrite( "houghSpaceLine.jpg", houghSpaceIMG );

	//Get hough Space Circle
	Mat houghSpaceIMGCircle;
	houghSpaceIMGCircle = houghSpaceCircle(thresholdMagnitude, gradientDirectionImage);

	getLines(houghSpaceIMG, frame);
	getCircles(houghSpaceIMGCircle);

	intersectionFunc();

	detectViola(frame);

	Mat finalImage;
	finalImage = combinationFunc(frame, houghSpaceIMG, houghSpaceIMGCircle);

	//Draw all ground truth rectangles
	groundTruth(frame, fileNumber);

	//Call tpr function with full set of detected boards
	double count_true_positive = tp_count(trueBoards, fileNumber);
	double false_negative = board_count[fileNumber] - count_true_positive;
	double count_false_positive = trueBoards.size()-count_true_positive;

	//Calculate F1 score with true positive count and false positive count
	double f1 = f1Score(count_true_positive, count_false_positive, false_negative);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}


/** @function detectAndDisplay */
void detectViola( Mat frame){
	Mat frame_gray;
	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(300,300) );

       // 3. Print number of boards found
	std::cout << "Number of boards detected: " << boards.size() << std::endl;

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
	//If no ground truth boards, true positive is N/A;
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
				if(iou(current_true,boards[j]) > 0.35) count_true_positive++;
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

Mat houghSpaceLine(Mat gradientMagnitudeImage, Mat gradientDirectionImage ){
	Mat houghSpace(2*(gradientMagnitudeImage.cols + gradientMagnitudeImage.rows), 360, CV_32SC1, Scalar(0));
	for (int y = 0; y < gradientMagnitudeImage.rows; y++) {
		for (int x = 0; x < gradientMagnitudeImage.cols; x++) {
			if (gradientMagnitudeImage.at<uchar>(y,x) == 255) {
					//double pheta = gradientDirectionImage.at<double>(y,x);
					for (double i = 0; i < 360; i+= 1) {
						double ro = x*cos(i*pi/180) + y*sin(i*pi/180) + gradientMagnitudeImage.cols + gradientMagnitudeImage.rows;
						houghSpace.at<int>(ro,i)++;
					}
			}
		}
	}
	//Normalize then return the houghspace
	//normalize(houghSpace, houghSpace, 0, 255, NORM_MINMAX, CV_32SC1);
	return houghSpace;
}

void getLines(Mat houghSpace, Mat image){
	double min , tempMax;
	double max = 1000;
	int locationMax[2];

	for (int n = 0; n < 10; n++) {
		minMaxIdx(houghSpace, &min, &max, NULL, locationMax);
		int y = locationMax[0];
		int x = locationMax[1];

		int ro = y - (image.cols + image.rows);
		double x0 = cos(x*pi/180) * ro;
		double y0 = sin(x*pi/180) * ro;
		lines[n][0] = round(x0+1000*(-sin(x*pi/180)));
		lines[n][1] = round(y0+1000*cos(x*pi/180));
		lines[n][2] = round(x0+(-1000)*(-sin(x*pi/180)));
		lines[n][3] = round(y0+(-1000)*cos(x*pi/180));

		for (double j = y - 5; j < y + 5; j++) {
			for (double i = x - 5; i < x + 5; i++) {
				houghSpace.at<int>(j,i) = 0;
			}
		}

	}
}

Mat houghSpaceCircle(Mat gradientMagnitudeImage, Mat gradientDirectionImage ){
	int num_radius = round(max(gradientMagnitudeImage.rows, gradientMagnitudeImage.cols) * 0.3);
	int dimensions[3] = {gradientMagnitudeImage.rows, gradientMagnitudeImage.cols, num_radius};
	Mat houghSpaceCircle(3, dimensions, CV_32SC1, cv::Scalar(0));
	for (size_t y = 0; y < gradientMagnitudeImage.rows; y++) {
		for (size_t x = 0; x < gradientMagnitudeImage.cols; x++) {
			for(size_t r = 5; r < num_radius; r += 1){
				if (gradientMagnitudeImage.at<uchar>(y,x) == 255) {
					double pheta = gradientDirectionImage.at<double>(y,x);
					for (double i = pheta - 0.1; i < pheta + 0.1; i+= 0.03) {

						int a = round(x - (r * cos(i)));
						int b = round(y - (r * sin(i)));

						if (a < 0 | b < 0 | a >= dimensions[1] | b >= dimensions[0]) { continue; }
						houghSpaceCircle.at<int>(b,a,r)++;
				}
				}
			}
		}
	}
	return houghSpaceCircle;
}

void getCircles(Mat houghSpace){

	double min , tempMax;
	double max = 1000;
	int locationMax[3];
	//minMaxIdx(houghSpace, &min, &tempMax, NULL, locationMax);
	for (int n = 0; n < 10; n++) {
		minMaxIdx(houghSpace, &min, &max, NULL, locationMax);

		int y = locationMax[0];
		int x = locationMax[1];
		int radius = locationMax[2];

		circles[n][0] = round(x);
		circles[n][1] = round(y);
		circles[n][2] = round(radius);
		for (double j = y - 5; j < y + 5; j++) {
			for (double i = x - 5; i < x + 5; i++) {
				houghSpace.at<int>(j,i,radius) = 0;
			}
		}
	}
}

void intersectionFunc(){
	Point l1p1, l1p2, l2p1, l2p2, x, d1, d2, r;
	float cross;
	double t1;
	for (int n = 0; n < 10; n++) {
		for (int m = 0; m < 10; m++) {
			l1p1 = Point(lines[n][0],lines[n][1]);
			l1p2 = Point(lines[n][2],lines[n][3]);

			l2p1 = Point(lines[m][0],lines[m][1]);
			l2p2 = Point(lines[m][2],lines[m][3]);

			x = l2p1 - l1p1;
			d1 = l1p2 - l1p1;
			d2 = l2p2 - l2p1;

			cross = d1.x*d2.y - d1.y*d2.x;
			//std::cout << cross << '\n';
			if(abs(cross) < 1e-8){continue;}

			t1 = (x.x * d2.y - x.y*d2.x)/cross;
			r = l1p1 + d1 * t1;

			intersections[intersectionCount][0] = round(r.x);
			intersections[intersectionCount][1] = round(r.y);
			intersectionCount++;
		}
	}
	return;
}

Mat combinationFunc(Mat frame, Mat houghSpaceIMG, Mat houghSpaceIMGCircle){
	int trueCounter = 0;
	for( int vjcount = 0; vjcount < boards.size(); vjcount++ ){
		bool newRect = false;
		for( int circlecount = 0; circlecount < 10; circlecount++ ){

			int radius = circles[circlecount][2];
			int x = circles[circlecount][0];
			int y = circles[circlecount][1];
			//circle(frame, Point(x,y), radius, Scalar( 0, 255, 0 ));
			Rect_<int> circleRect(x - radius,y - radius,2*radius, 2*radius);

			if (iou(boards[vjcount], circleRect) > 0.4) {
				if (trueCounter > 0 && iou(circleRect, trueBoards[trueCounter-1]) > 0.3) {
					break;
				}
				trueBoards.push_back(circleRect);
				trueCounter++;
				newRect = true;
				break;
			}
		}
		if (newRect == false) {
			for( int linecount = 0; linecount < intersectionCount; linecount++ ){
				int x = intersections[linecount][0];
				int y = intersections[linecount][1];
				int height = boards[vjcount].height/2;
				int width = boards[vjcount].width/2;
				//circle(frame, Point(x,y), radius, Scalar( 0, 255, 0 ));
				Rect_<int> lineRect(x - width,y - height, width*2, height*2);


				if (iou(boards[vjcount], lineRect) > 0.85) {
					if (trueCounter > 0 && iou(lineRect, trueBoards[trueCounter-1]) > 0.1) {
						break;
					}
					trueBoards.push_back(lineRect);
					trueCounter++;
					break;
				}
			}
		}
	}

	for (int i = 0; i < trueBoards.size(); i++) {
		rectangle(frame, trueBoards[i], Scalar( 255, 255, 0 ), 3);
	}

	return frame;
}

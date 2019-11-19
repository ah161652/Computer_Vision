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
void detectAndDisplay( Mat frame, String fileName);
void groundTruth (Mat frame, int fileNumber);
double tp_count(vector<Rect> boards, int fileNumber);
double f1Score(double count_true_positive, double	count_false_positive);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
std::vector<Rect> boards;

int coordinates[16][4] = {{426,0,625,220},{167,104,417,353},{90,86,202,197},{313,137,397,228},{158,67,385,328},{415,127,552,265},{204,107,281,188},{234,152,406,337},{0,0,0,0},{167,18,465,317},{0,0,0,0},{168,95,239,194},{153,58,223,233},{257,104,419,270},{0,0,0,0},{128,36,301,213}};
int coordinates8[2][4] = {{62,241,133,350},{830,203,973,353}};
int coordinates10[3][4] = {{76,88,197,228},{578,119,647,225},{914,143,954,221}};
int coordinates14[2][4] = {{105,85,261,244},{973,80,1126,236}};

/** @function main */
int main( int argc, const char** argv )
{
  // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	String fileName = argv[1];
	int fileNumber = atoi(argv[2]);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay(frame, fileName);

	//Draw all ground truth rectangles
	groundTruth(frame, fileNumber);

	//Call tpr function with full set of detected faces
	double count_true_positive = tp_count(boards, fileNumber);

	//Calculate F1 score with true positive count and false positive count
	double f1 = f1Score(count_true_positive, (boards.size()-count_true_positive));

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, String fileName)
{

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
	//dependant on file name draw all True faces using global arrays
	if (fileNumber == 8){
		for (size_t i = 0; i < 2; i++) {
			rectangle(frame, Point(coordinates8[i][0], coordinates8[i][1]), Point(coordinates8[i][2], coordinates8[i][3]), Scalar(0,0,255), 2);
		}
	}
	else if (fileNumber == 10){
		for (size_t i = 0; i < 3; i++) {
			rectangle(frame, Point(coordinates10[i][0], coordinates10[i][1]), Point(coordinates10[i][2], coordinates10[i][3]), Scalar(0,0,255), 2);
		}
	}
	else if (fileNumber == 14){
		for (size_t i = 0; i < 2; i++) {
			rectangle(frame, Point(coordinates14[i][0], coordinates14[i][1]), Point(coordinates14[i][2], coordinates14[i][3]), Scalar(0,0,255), 2);
		}
	}
	else{
		rectangle(frame, Point(coordinates[fileNumber][0], coordinates[fileNumber][1]), Point(coordinates[fileNumber][2], coordinates[fileNumber][3]), Scalar(0,0,255), 2);
	}
}

double iou(Rect_<int> current_true, Rect_<int> current_detected){
	//Uses Rect funtions to calculate the intersection
	Rect_<int> intersection = current_true & current_detected;

	//Union is trivial if intersection and total area is known
	double union_of = current_detected.area() + current_true.area() - intersection.area();
	double iou_val = (intersection.area())/(union_of);

	return iou_val;
}


double tp_count(vector<Rect> boards, int fileNumber){
	double count_true_positive = 0;
	double tpr;

	//dependant on file name draw all True faces using global arrays
	if (fileNumber == 8){
		for (size_t i = 0; i < 2; i++) {
			for( int j = 0; j < boards.size(); j++ ){
					Rect_<int> current_true(coordinates8[i][0],coordinates8[i][1],coordinates8[i][2]-coordinates8[i][0],coordinates8[i][3]-coordinates8[i][1]);
					if(iou(current_true,boards[j]) > 0.5) count_true_positive++;
			}
			tpr = count_true_positive/2;
		}
	}
	else if (fileNumber == 10){
		for (size_t i = 0; i < 3; i++) {
			for( int j = 0; j < boards.size(); j++ ){
					Rect_<int> current_true(coordinates10[i][0],coordinates10[i][1],coordinates10[i][2]-coordinates10[i][0],coordinates10[i][3]-coordinates10[i][1]);
					if(iou(current_true,boards[j]) > 0.5) count_true_positive++;
			}
			tpr = count_true_positive/3;
		}
	}
	else if (fileNumber == 14){
		for (size_t i = 0; i < 2; i++) {
			for( int j = 0; j < boards.size(); j++ ){
					Rect_<int> current_true(coordinates14[i][0],coordinates14[i][1],coordinates14[i][2]-coordinates14[i][0],coordinates14[i][3]-coordinates14[i][1]);
					if(iou(current_true,boards[j]) > 0.5) count_true_positive++;
			}
			tpr = count_true_positive/2;
		}
	}
	else{
		for( int j = 0; j < boards.size(); j++ ){
				Rect_<int> current_true(coordinates[fileNumber][0],coordinates[fileNumber][1],coordinates[fileNumber][2]-coordinates[fileNumber][0],coordinates[fileNumber][3]-coordinates[fileNumber][1]);
				if(iou(current_true,boards[j]) > 0.5) count_true_positive++;
		}
		tpr = count_true_positive/1;
	}
	std::cout << "TPR:  "<< tpr << '\n';
	return count_true_positive;
}

//Calculate the F1 Score
double f1Score(double count_true_positive, double	count_false_positive){
	//Required variables using TP and FP count
	double precision = (count_true_positive)/(count_true_positive + count_false_positive);
	double recall = (count_true_positive)/(count_true_positive + 0);

	double f1 = 2 * (precision*recall) / (precision + recall);
	std::cout << "F1 Score:  "<< f1 << '\n';
	return f1;
}

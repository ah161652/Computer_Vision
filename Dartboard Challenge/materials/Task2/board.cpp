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
void detectAndDisplay( Mat frame);
void groundTruth (Mat frame, int fileNumber);
double tp_count(vector<Rect> boards, int fileNumber);
double f1Score(double count_true_positive, double	count_false_positive);

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;
std::vector<Rect> boards;

int coordinates[20][4] = {{459,36,573,172},{218,152,369,299},{112,109,178,174},{332,157,380,211},{207,120,396,281},{446,151,529,235},{219,124,266,172},{271,187,386,299},{73,263,117,327},{857,231,943,321},{227,73,406,253},{102,116,175,198},{590,139,633,201},{921,158,947,206},{182,115,226,166},{162,96,208,198},{288,138,388,236},{135,116,229,212},{1002,112,1096,206},{171,75,273,178}};
int coordinates8[2][4] = {{73,263,117,327},{857,231,943,321}};
int coordinates10[3][4] = {{102,116,175,198},{590,139,633,201},{921,158,947,206}};
int coordinates14[2][4] = {{135,116,229,212},{1002,112,1096,206}};

int board_count[16]={1,1,1,1,1,1,1,1,2,1,3,1,1,1,2,1};
int first_board_index[16] = {0,1,2,3,4,5,6,7,8,10,11,14,15,16,17,19};
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
	detectAndDisplay(frame);

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
void detectAndDisplay( Mat frame)
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
	//dependant on file name draw all True boards using global arrays
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
	double union_of = current_detected.area() + current_true.area() - intersection.area();
	double iou_val = (intersection.area())/(union_of);

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
	for (i; i < limit; i++) {
		for( int j = 0; j < boards.size(); j++ ){
				Rect_<int> current_true(coordinates[i][0],coordinates[i][1],coordinates[i][2]-coordinates[i][0],coordinates[i][3]-coordinates[i][1]);
				if(iou(current_true,boards[j]) > 0.4) count_true_positive++;
		}
	}

	//Calculate true positive ratio
	tpr = count_true_positive/board_count[fileNumber];

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

/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
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

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, int fileNumber);
void groundTruth (Mat frame, int fileNumber);
double tp_count(vector<Rect> faces, int fileNumber);
double f1Score(double count_true_positive, double	count_false_positive);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

int coordinates[19][4] ={{346, 129, 476, 269}, {55,259,111,321}, {191,223,251,286}, {294,251,347,314}, {428,243,489,311}, {564,253,620,317}, {683,252,735,316}, {63,147,122,210}, {248,171,309,235}, {377,199,450,250}, {515,190,583,240}, {644,196,712,254}, {289,104,324,156}, {347,205,423,285}, {89,222,194,345}, {321,85,383,149}, {420,142,524,259}, {472,225,549,325}, {720,205,834,296}};
int face_count[16]={0,0,0,0,1,11,1,1,0,1,0,1,0,1,2};
int first_face_index[16] = {0,0,0,0,0,1,12,13,13,14,14,15,15,16,17};

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
	detectAndDisplay(frame, fileNumber);

	// Draw ground truth boxes
	groundTruth(frame, fileNumber);

	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, int fileNumber)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

 	// 3. Print number of Faces found
	std::cout << "Number of faces detected: " << faces.size() << std::endl;

  // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	//Call tpr function with ful set of detected faces
	double count_true_positive = tp_count(faces, fileNumber);

	//Calculate f1Score with true positives and false positives
	double f1 = f1Score(count_true_positive, (faces.size()-count_true_positive));
}


void groundTruth (Mat frame, int fileNumber){
	//dependant on file name draw all True faces using global arrays
	if (face_count[fileNumber] == 0) {
		return;
	}

	//Gather indexes specific for that file
	int i = first_face_index[fileNumber];
	int limit = face_count[fileNumber] + i;

	for (i; i < limit; i++) {
		rectangle(frame, Point(coordinates[i][0], coordinates[i][1]), Point(coordinates[i][2], coordinates[i][3]), Scalar(0,0,255), 2);
	}
}

//Calculate intersection over union
double iou(Rect_<int> current_true, Rect_<int> current_detected){
	Rect_<int> intersection = current_true & current_detected;
	double union_of = current_detected.area() + current_true.area() - intersection.area();
	double iou_val = (intersection.area())/(union_of);
	return iou_val;
}


double tp_count(vector<Rect> faces, int fileNumber){
	double count_true_positive = 0;
	double tpr;
	//If no ground truth faces, true positive is N/A;
	if (face_count[fileNumber] == 0) {
		std::cout << "TPR: NaN " << '\n';
		return std::numeric_limits<double>::quiet_NaN();
	}
	//Gather indexes specific for that file
	int i = first_face_index[fileNumber];
	int limit = face_count[fileNumber] + i;
	for (i; i < limit; i++) {
		for( int j = 0; j < faces.size(); j++ ){
				Rect_<int> current_true(coordinates[i][0],coordinates[i][1],coordinates[i][2]-coordinates[i][0],coordinates[i][3]-coordinates[i][1]);
				if(iou(current_true,faces[j]) > 0.5) count_true_positive++;
		}
	}

	//Calculate true positive ratio
	tpr = count_true_positive/face_count[fileNumber];

	std::cout << "TPR:  "<< tpr << '\n';
	return count_true_positive;
}

double f1Score(double count_true_positive, double	count_false_positive){
	double precision = (count_true_positive)/(count_true_positive + count_false_positive);
	double recall = (count_true_positive)/(count_true_positive + 0);

	double f1 = 2 * (precision*recall) / (precision + recall);
	std::cout << "F1 Score:  "<< f1 << '\n';
	return f1;
}

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
void detectAndDisplay( Mat frame, Int fileNumber );
void groundTruth (Mat frame, String fileName);

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;

int coordinates4[4] = {346, 129, 476, 269};
int coordinates5[11][4] = {{55,259,111,321}, {191,223,251,286}, {294,251,347,314}, {428,243,489,311}, {564,253,620,317}, {683,252,735,316}, {63,147,122,210}, {248,171,309,235}, {377,199,450,250}, {515,190,583,240}, {644,196,712,254}};
int coordinates6[4] = {289,144,324,156};
int coordinates7[4] = {347,205,423,285};
int coordinates9[4] = {89,22,194,145};
int coordinates11[4] = {321,85,383,149};
int coordinates13[4] = {420,142,524,259};
int coordinates14[2][4] = {{472,225,549,325}, {720,205,834,296}};



/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	String fileName = argv[1];
	char fileNumber = fileName.at(4);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame, fileNumber );

	groundTruth(frame, fileName);
ame
	// 4. Save Result Image
	imwrite( "detected.jpg", frame );

	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, Int fileNumber )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}


	tpr(faces, fileNumber);


}

void groundTruth (Mat frame, String fileName){

	if (fileName == "dart4.jpg"){
			rectangle(frame, Point(coordinates4[0], coordinates4[1]), Point(coordinates4[2], coordinates4[3]), Scalar(0,0,255), 2);
	}

	if (fileName == "dart5.jpg"){
		for (size_t i = 0; i < 11; i++) {
			rectangle(frame, Point(coordinates5[i][0], coordinates5[i][1]), Point(coordinates5[i][2], coordinates5[i][3]), Scalar(0,0,255), 2);
		}
	}

	if (fileName == "dart13.jpg"){
			rectangle(frame, Point(coordinates13[0], coordinates13[1]), Point(coordinates13[2], coordinates13[3]), Scalar(0,0,255), 2);
	}

	if (fileName == "dart14.jpg"){
		for (size_t i = 0; i < 2; i++) {
			rectangle(frame, Point(coordinates14[i][0], coordinates14[i][1]), Point(coordinates14[i][2], coordinates14[i][3]), Scalar(0,0,255), 2);
		}
	}

}

double tpr(vector<Rect> faces, Int fileNumber){
	int count = 0;

	if


	for( int i = 0; i < faces.size(); i++ )
	{
	}

	double tpr = count / numberValidFaces;
}

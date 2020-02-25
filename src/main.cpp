#include <iostream>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <cstdlib>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "types.h"

using namespace cv;

/** Global Variables */
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
double fps;
int key;
std::string fpsMessage;
VideoCapture camera;

/** Function Headers */
void videoWithoutDetection(void);
void videoWithDetection(void);
void detectAndDisplay(Mat img);
void *threadproc(void *ptr);

int main(int argc, const char** argv) {
	int err;
	pthread_t tid;

	//create thread to print frames per second
	if (int err = pthread_create(&tid, NULL, &threadproc, NULL))
	{
	  std::cerr << "pthread_create Error. " << strerror(err);
	}

	videoWithoutDetection();
	videoWithDetection();

	return EXIT_SUCCESS;
}

/** Function Definitions */

void videoWithoutDetection(void)
{
        if (!camera.open(0)) {
		printf("Can't find a camera\n");
		return;
	};
	
	// Main loop
	Mat img;

	fpsMessage = "Frames per second without face and eye detection:  ";

	while(1)
	{
		camera >> img;
		imshow("Camera", img);
		key = waitKey(5);
		fps = camera.get(CAP_PROP_FPS);
		if (key == 27 || key == 'q') break;
	}
	camera.release();	
}

void videoWithDetection(void)
{
    String face_cascade_name, eyes_cascade_name;
    
    face_cascade_name = samples::findFile("../data/haarcascades/haarcascade_frontalface_alt.xml");

    if( !face_cascade.load(face_cascade_name) )
    {
        std::cerr << "Error loading face cascade!\n";
        return;
    }
    
    eyes_cascade_name = samples::findFile("../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    
    if( !eyes_cascade.load( eyes_cascade_name ) )
    {
        std::cerr << "Error loading eyes cascade!\n";
        return;
    }
    
    if (!camera.open(0))
    {
        std::cerr << "Error opening video capture!\n";
        return;
    }
    
    //Main loop
    Mat img;
     
    fpsMessage = "Frames per second with face and eye detection:  ";
   
    while(1)
    {
      	camera >> img;
	detectAndDisplay(img);
	key = waitKey(5);
	fps = camera.get(CAP_PROP_FPS);
       	if (key == 27 || key == 'q') break; 
    }
    camera.release();
}

void detectAndDisplay(Mat img)
{
    Mat frame_gray;
    cvtColor(img, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    std::vector<Rect> faces;
    face_cascade.detectMultiScale(frame_gray, faces);

    for (size_t i = 0; i < faces.size(); i++)
    {
        Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
        ellipse(img, center, Size(faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar(255, 0, 255), 4);

        Mat faceROI = frame_gray( faces[i] );
	
        std::vector<Rect> eyes;
        eyes_cascade.detectMultiScale(faceROI, eyes);

        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point eye_center(faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2);
            int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
            circle(img, eye_center, radius, Scalar(255, 0, 0), 4);
        }
    }

    imshow("Camera - Face and Eye Detection", img);
}

void *threadproc(void *ptr)
{
    while(1)
    {
        //print every 2 seconds
        sleep(2);
	std::cout<< fpsMessage << fps << std::endl;
    }
    return EXIT_SUCCESS;
}

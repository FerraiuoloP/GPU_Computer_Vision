// nvcc -std=c++14 -ccbin g++ main.cpp src/cuda_kernel.cu -o main `pkg-config --cflags --libs opencv4`
#define VIDEO
#define CANNY
#define THRESH_GUI_
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "include/utils.h"
#include "include/edge_detection.h"

using namespace cv;
using namespace std;
const int FILTER_WIDTH = 5;
const float FILTER_SIGMA = 1.75f;
const float ALPHA = 0.05;
const float K = 0.05;
int main()
{

    // sobel kernel
    // float sobel_x_kernel[9] = {-1., -2., -1., 0., 0., 0., 1., 2., 1.};
    // float sobel_y_kernel[9] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
    float sobel_x_kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    float sobel_y_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

// opening the image from the address given below.
#ifdef VIDEO
    cv::VideoCapture cap("input/video.mp4");
#endif
// cv::VideoCapture cap(0);
#ifdef VIDEO
    int desired_fps = 60;                       // Puoi cambiare a 60 per 60 FPS
    int target_frame_time = 1000 / desired_fps; // Tempo target per frame in millisecondi
    while (cap.isOpened())
    {
        int64 start_time = cv::getTickCount();
        cv::Mat img;
        cap >> img;
        if (img.empty())
        {
            break;
        }
#else
    cv::Mat img = cv::imread("input/traffic.jpg", cv::IMREAD_COLOR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // cv::Mat img = cv::imread("input/traffic.jpg", cv::IMREAD_COLOR);
#endif

        float *gaussian_kernel = computeGaussianKernel(FILTER_WIDTH, FILTER_SIGMA);
#ifdef CANNY
        img = cannyEdgeDetectionCPU(&img, sobel_x_kernel, sobel_y_kernel, FILTER_WIDTH);
#else
    img = harrisCornerDetectorCPU(&img, gaussian_kernel, sobel_x_kernel, sobel_y_kernel, FILTER_WIDTH);
#endif
        cv::imshow("Original Image with Red Dots", img);
#ifndef VIDEO
        cv::waitKey(0);
#else
    img.release();
    int64 end_time = cv::getTickCount();
    double elapsed_time_ms = (end_time - start_time) * 1000 / cv::getTickFrequency();
    cout << "FPS: " << 1000 / elapsed_time_ms << endl;

    // Calcola il tempo di attesa necessario per raggiungere il frame rate desiderato
    int delay = std::max(1, target_frame_time - static_cast<int>(elapsed_time_ms));
    if (cv::waitKey(delay) == 27) // wait to press 'esc' key
    {
        break;
    }
}
#endif
    }
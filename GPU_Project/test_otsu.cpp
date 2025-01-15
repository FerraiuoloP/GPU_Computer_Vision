// nvcc -std=c++14 -ccbin g++ main.cpp src/cuda_kernel.cu -o main `pkg-config --cflags --libs opencv4`
#define S
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "include/cuda_kernel.cuh"
#include "include/utils.h"
#include "include/edge_detection.h"

using namespace cv;
using namespace std;

int main()
{
	const int FILTER_WIDTH = 5;
	const float FILTER_SIGMA = 1.75f;
	const float ALPHA = 0.05;
	const float K = 0.05;
	// sobel kernel
	// float sobel_x_kernel[9] = {-1., -2., -1., 0., 0., 0., 1., 2., 1.};
	// float sobel_y_kernel[9] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
	float sobel_x_kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	float sobel_y_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

	// opening the image from the address given below.

	cv::Mat img = cv::imread("input/traffic.jpg", cv::IMREAD_COLOR);

	// cv::imshow("Frame", img);

	// invert the red and the blue channels(CV by default reads the image in BGR format)
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

	if (img.empty())
	{
		std::cerr << "Error: Unable to load image." << std::endl;
		return -1;
	}
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	// convert the image to grayscale
	cv::Mat gray_img;
	cv::cvtColor(img, gray_img, cv::COLOR_RGB2GRAY);

	printf("here\n");

	// copy img.data to device
	float *img_h = (float *)malloc(width * height * channels * sizeof(float));
	// convert img.data to float
	for (int i = 0; i < width * height; i++)
	{
		img_h[i] = (float)gray_img.data[i];
	}
	float *img_d;
	cudaMalloc(&img_d, width * height * channels * sizeof(float));
	cudaMemcpy(img_d, img_h, width * height * channels * sizeof(float), cudaMemcpyHostToDevice);
	binarize_img(img_d, img_h, width, height);

	// Copy img_h to img.data using cv methods and then show it
	for (int i = 0; i < width * height; i++)
	{
		gray_img.data[i] = (unsigned char)img_h[i];
	}
	cv::imshow("Gray", gray_img);
	cv::waitKey(0);
	return 0;
}
// g++ -std=c++11 -IC:C:\opencv\opencv\build\include  -LC:C:\opencv\opencv\build\x64\vc15\lib -lopencv_core470 -lopencv_highgui470 -lopencv_imgcodecs470 -lopencv_imgproc470 -o my_program.exe main.cpp
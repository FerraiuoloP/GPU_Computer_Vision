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
#include "include/cuda_kernel.cuh"
#include "include/utils.h"
#include "include/edge_detection.h"

using namespace cv;
using namespace std;

int main()
{

	// sobel kernel
	// float sobel_x_kernel[9] = {-1., -2., -1., 0., 0., 0., 1., 2., 1.};
	// float sobel_y_kernel[9] = {-1., 0., 1., -2., 0., 2., -1., 0., 1.};
	float sobel_x_kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
	float sobel_y_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

#ifdef VIDEO
	cv::VideoCapture cap("input/cars.mp4");
#endif
// cv::VideoCapture cap(0);
#ifdef VIDEO
	int desired_fps = 60;
	int target_frame_time = 1000 / desired_fps;
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
	cv::Mat img = cv::imread("input/lizard.jpg", cv::IMREAD_COLOR);
	// cv::Mat img = cv::imread("input/traffic.jpg", cv::IMREAD_COLOR);
#endif
		// cv::imshow("Frame", img);

		// bgr -> rgb (since opencv loads images in bgr format)
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

		if (img.empty())
		{
			std::cerr << "Error: Unable to load image." << std::endl;
			return -1;
		}
		int width = img.cols;
		int height = img.rows;
		int channels = img.channels();

		size_t img_size_h = width * height * channels * sizeof(unsigned char);
		size_t img_gray_size_h = width * height * sizeof(float);

		unsigned char *img_d;
		float *img_gray_d;
		float *img_blurred;
		float *img_harris_d;
		float *img_sobel_x;
		float *img_sobel_y;

		float *sobel_x_kernel_d;
		float *sobel_y_kernel_d;

		unsigned char *img_h = (unsigned char *)malloc(img_size_h);
		float *img_gray_h = (float *)malloc(img_gray_size_h);
		float *img_grayh2 = (float *)malloc(img_gray_size_h);
		float *img_harris_h = (float *)malloc(img_gray_size_h);

		float *gaussian_kernel = computeGaussianKernel(FILTER_WIDTH, FILTER_SIGMA);
		float *gaussian_kernel_d;

		// cout << "Width: " << width << "\nHeight: " << height << "\nChannels: " << channels << "\nImage Size: " << img_size_h << endl;

		// data to host
		memcpy(img_h, img.data, img_size_h);

		// cuda memory allocations
		cudaMalloc(&img_d, img_size_h);
		cudaMalloc(&img_gray_d, img_gray_size_h);
		cudaMalloc(&img_blurred, img_gray_size_h);
		cudaMalloc(&gaussian_kernel_d, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));
		cudaMalloc(&sobel_x_kernel_d, 3 * 3 * sizeof(float));
		cudaMalloc(&sobel_y_kernel_d, 3 * 3 * sizeof(float));
		cudaMalloc(&img_sobel_x, img_gray_size_h);
		cudaMalloc(&img_sobel_y, img_gray_size_h);
		cudaMalloc(&img_harris_d, img_gray_size_h);

		// cudamemcpys
		cudaMemcpy(img_d, img_h, img_size_h, cudaMemcpyHostToDevice);
		cudaMemcpy(gaussian_kernel_d, gaussian_kernel, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(sobel_x_kernel_d, sobel_x_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(sobel_y_kernel_d, sobel_y_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

		// RGB to Gray
		rgbToGrayKernelWrap(img_d, img_gray_d, width, height);

		// saving image
		float *img_save = (float *)malloc(img_gray_size_h);
		cudaMemcpy(img_save, img_gray_d, img_gray_size_h, cudaMemcpyDeviceToHost);
		cv::Mat img_gray(height, width, CV_32F, img_save);
		cv::imwrite("./img_gray.png", img_gray);

		// Apply Gaussian Blur to grayscale image
		convolutionGPUWrap(img_blurred, img_gray_d, width, height, gaussian_kernel_d);

		// // saving image after gaussian blur
		float *img_save2 = (float *)malloc(img_gray_size_h);
		cudaMemcpy(img_save2, img_blurred, img_gray_size_h, cudaMemcpyDeviceToHost);
		cv::Mat img_blurred_s(height, width, CV_32F, img_save2);
		cv::imwrite("./img_blurred.png", img_blurred_s);

// Otsu thresholding
#ifdef CANNY
		int threshold = otsu_threshold(img_blurred, width, height);
		// int threshold = 100;
#endif

		// Sobel X
		convolutionGPUWrap(img_sobel_x, img_blurred, width, height, sobel_x_kernel_d);
		// saving image after sobel x
		float *img_save3 = (float *)malloc(img_gray_size_h);
		cudaMemcpy(img_save3, img_sobel_x, img_gray_size_h, cudaMemcpyDeviceToHost);
		cv::Mat img_sobelx_s(height, width, CV_32F, img_save3);
		cv::imwrite("./img_sobel_x.png", img_sobelx_s);

		// show the image
		//  printf("Sobel x\n");
		//  cudaMemcpy(img_gray_h, img_sobel_x, img_gray_size_h, cudaMemcpyDeviceToHost);
		//  showImage(height, width, img_gray_h, "Sobel X");

		// Sobel Y
		convolutionGPUWrap(img_sobel_y, img_blurred, width, height, sobel_y_kernel_d);
		// saving image after sobel y
		float *img_save4 = (float *)malloc(img_gray_size_h);
		cudaMemcpy(img_save4, img_sobel_y, img_gray_size_h, cudaMemcpyDeviceToHost);
		cv::Mat img_sobely_s(height, width, CV_32F, img_save4);
		cv::imwrite("./img_sobel_y.png", img_sobely_s);

		// show the image
		//  printf("Sobel Y\n");
		//  cudaMemcpy(img_gray_h, img_sobel_y, img_gray_size_h, cudaMemcpyDeviceToHost);
		//  showImage(height, width, img_gray_h, "Sobel Y");

		// convolutionGPU2Wrap(img_blurred, sobel_x_kernel_d, img_sobel_x, 1, width, height);
		// cudaMemcpy(img_gray_h, img_sobel_x, img_gray_size_h, cudaMemcpyDeviceToHost);
		// showImage(height, width, img_gray_h, "Sobel x");

		// convolutionGPU2Wrap(img_blurred, sobel_y_kernel_d, img_sobel_y, 1, width, height);
		// // //print the image
		// cudaMemcpy(img_gray_h, img_sobel_y, img_gray_size_h, cudaMemcpyDeviceToHost);
		// showImage(height, width, img_gray_h, "Sobel Y");

		// convolutionGPUWrap(img_sobel_x, img_blurred, width, height, sobel_x_kernel_d);
		// show the image
		// cudaMemcpy(img_gray_h, img_sobel_y, img_gray_size_h, cudaMemcpyDeviceToHost);

		// showImage(height, width, img_gray_h, "Sobel X");

		// // Apply Sobel Y kernel to the blurred image
		// gaussianBlurKernelWrap(img_blurred, img_sobel_y, width, height, sobel_y_kernel_d, 3);
#ifndef CANNY
		harrisCornerDetector(&img, img_sobel_x, img_sobel_y, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH);
#endif
#ifdef THRESH_GUI
		int thresh_h = threshold;
		int thresh_l = threshold / 2;
		cv::namedWindow("Original Image with Red Dots", cv::WINDOW_NORMAL);
		cv::createTrackbar("Threshold High", "Original Image with Red Dots", &thresh_h, 255);
		cv::createTrackbar("Threshold Low", "Original Image with Red Dots", &thresh_l, 255);
		while (true)
		{
			cannyEdgeDetector(&img, img_sobel_x, img_sobel_y, width, height, thresh_l, thresh_h, gaussian_kernel_d, FILTER_WIDTH);
			cv::imshow("Original Image with Red Dots", img);
			if (cv::waitKey(1) == 27) // wait to press 'esc' key
			{
				break;
			}
		}
#else
#ifdef CANNY
	int thresh_h = threshold;
	int thresh_l = threshold / 2;
	cannyEdgeDetector(&img, img_sobel_x, img_sobel_y, width, height, thresh_l, thresh_h, gaussian_kernel_d, FILTER_WIDTH);
	// save img
	cv::imwrite("./img_canny.png", img);
#endif
#endif

		// convert to bgr
		cv::cvtColor(img, img, cv::COLOR_RGB2BGR);

		cv::imshow("Output Image", img);
		// save img
		cv::imwrite("./img_harris.png", img);
		// cv::waitKey(0);
		img.release();

		cudaFree(img_d);
		cudaFree(img_gray_d);
		cudaFree(img_blurred);
		cudaFree(gaussian_kernel_d);
		cudaFree(sobel_x_kernel_d);
		cudaFree(sobel_y_kernel_d);
		cudaFree(img_sobel_x);
		cudaFree(img_sobel_y);
		cudaFree(img_harris_d);
		free(img_gray_h);
		free(img_harris_h);
		free(img_grayh2);
		free(gaussian_kernel);
		free(img_h);

#ifndef VIDEO
		cv::waitKey(0);
#else
	int64 end_time = cv::getTickCount();
	double elapsed_time_ms = (end_time - start_time) * 1000 / cv::getTickFrequency();
	cout << "FPS: " << 1000 / elapsed_time_ms << endl;

	int delay = std::max(1, target_frame_time - static_cast<int>(elapsed_time_ms));
	if (cv::waitKey(delay) == 27) // 27=esc key
	{
		break;
	}
}
#endif
		return 0;
	}
	// g++ -std=c++11 -IC:C:\opencv\opencv\build\include  -LC:C:\opencv\opencv\build\x64\vc15\lib -lopencv_core470 -lopencv_highgui470 -lopencv_imgcodecs470 -lopencv_imgproc470 -o my_program.exe main.cpp
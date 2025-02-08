// nvcc -std=c++14 -ccbin g++ main.cpp src/cuda_kernel.cu -o main `pkg-config --cflags --libs opencv4`
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <thread>
#include <cuda_runtime.h>
#include "include/cuda_kernel.cuh"
#include "include/utils.h"
#include "include/edge_detection.h"

using namespace cv;
using namespace std;

const float sobel_x_kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};

const float sobel_x_separable[3] = {1, 2, 1};
const float sobel_x_separable_2[3] = {1, 0, -1};

const float sobel_y_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

const float sobel_y_separable[3] = {1, 0, -1};
const float sobel_y_separable_2[3] = {1, 2, 1};
enum Mode
{
	// -H. Normal Harris Corner Detection
	HARRIS,
	// -S. Harris corner detection with Shi-Tomasi response function
	SHI_TOMASI,
	// -C. Canny Edge Detection with Otsu Thresholding
	CANNY,
	// -C -l=low -h=high. Canny Edge Detection with manual thresholding. Optional
	CANNY_MANUAL,
	// -C -g. Canny Edge Detection with GUI thresholding. Optional
	CANNY_GUI,
	// -O. Otsu thresholding method for image binarization
	OTSU_BIN,
	// -A. All at once
	ALL
};

void save_image(float *img_d, size_t img_size_h, int height, int width, std::string filename)
{
	float *img_save = (float *)malloc(img_size_h);
	cudaMemcpy(img_save, img_d, img_size_h, cudaMemcpyDeviceToHost);
	cv::Mat img_gray(height, width, CV_32F, img_save);
	cv::imwrite(filename, img_gray);
}

/**
 * @brief Handles image processing: RGB to Gray, Gaussian Blur then Harris/ShiTomasi Corner Detection, Canny Edge Detection or Otsu binarization
 *
 * @param mode Execution mode. Can be HARRIS, SHI_TOMASI, CANNY, CANNY_MANUAL, CANNY_GUI, OTSU_BIN
 * @param filename Image filename
 * @param low_threshold Low threshold for Canny Edge Detection Manual mode
 * @param high_threshold High threshold for Canny Edge Detection Manual mode
 */
void handle_image(enum Mode mode, std::string filename, int low_threshold, int high_threshold, bool from_video = false, cv::Mat img_v = cv::Mat())
{
	cv::Mat img;
	if (!from_video)
	{
		img = cv::imread(filename, cv::IMREAD_COLOR);

		if (img.empty())
		{
			std::cerr << "Error: Unable to load image." << std::endl;
			return;
		}
	}
	else
	{
		img = img_v;
	}
	cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);

	// variable declarations
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();
	printf("Channels: %d\n", channels);
	size_t img_size_h = width * height * channels * sizeof(unsigned char);
	size_t img_gray_size_h = width * height * sizeof(float);

	printf("Image size: %d x %d \n", width, height);

	// device variable declarations
	uchar4 *img_d;
	float *img_gray_d;
	float *img_blurred_d;
	float *img_harris_d;
	float *img_sobel_x_d;
	float *img_sobel_y_d;

	// mallocs
	unsigned char *img_h = (unsigned char *)malloc(img_size_h);
	float *img_gray_h = (float *)malloc(img_gray_size_h);
	float *img_grayh2 = (float *)malloc(img_gray_size_h);
	float *img_harris_h = (float *)malloc(img_gray_size_h);

	// kernel devices
	float *sobel_x_kernel_d;
	float *sobel_x_separable_d;
	float *sobel_x_separable_2_d;
	float *sobel_y_kernel_d;
	float *sobel_y_separable_d;
	float *sobel_y_separable_2_d;

	auto start = std::chrono::high_resolution_clock::now();
	float *gaussian_kernel = computeGaussianKernel(FILTER_WIDTH, FILTER_SIGMA);
	float *gaussian_kernel_d;

	// data to host
	// memcpy(img_h, img.data, img_size_h);

	// cuda memory allocations

	cudaMalloc(&img_d, img_size_h);
	cudaMalloc(&img_gray_d, img_gray_size_h);
	cudaMalloc(&img_blurred_d, img_gray_size_h);
	cudaMalloc(&gaussian_kernel_d, FILTER_WIDTH * FILTER_WIDTH * sizeof(float));
	cudaMalloc(&sobel_x_kernel_d, 3 * 3 * sizeof(float));
	cudaMalloc(&sobel_x_separable_d, 3 * sizeof(float));
	cudaMalloc(&sobel_x_separable_2_d, 3 * sizeof(float));
	cudaMalloc(&sobel_y_kernel_d, 3 * 3 * sizeof(float));
	cudaMalloc(&sobel_y_separable_d, 3 * sizeof(float));
	cudaMalloc(&sobel_y_separable_2_d, 3 * sizeof(float));
	cudaMalloc(&img_sobel_x_d, img_gray_size_h);
	cudaMalloc(&img_sobel_y_d, img_gray_size_h);
	cudaMalloc(&img_harris_d, img_gray_size_h);

	// cudamemcpys
	// cudaMemcpy(img_d, img_h, img_size_h, cudaMemcpyHostToDevice);
	cudaHostRegister(img.data, img_size_h, cudaHostRegisterPortable);
	cudaMemcpy(img_d, img.data, img_size_h, cudaMemcpyHostToDevice);
	cudaMemcpy(gaussian_kernel_d, gaussian_kernel, FILTER_WIDTH * FILTER_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(sobel_x_kernel_d, sobel_x_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_y_kernel_d, sobel_y_kernel, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpy(sobel_x_separable_d, sobel_x_separable, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_x_separable_2_d, sobel_x_separable_2, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_y_separable_d, sobel_y_separable, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(sobel_y_separable_2_d, sobel_y_separable_2, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// Commong operations for all modes(except for OTSU_BIN)

	// RGB to Gray
	rgbToGrayKernelWrap(img_d, img_gray_d, width, height);
	// show rgb to gray image
	// cudaMemcpy(img_gray_h, img_gray_d, img_gray_size_h, cudaMemcpyDeviceToHost);
	// saveImage(height, width, img_gray_h, "debug/gray_cuda.jpg");

	// //convert to host
	// cudaMemcpy(img_gray_h, img_gray_d, img_gray_size_h, cudaMemcpyDeviceToHost);
	// showImage(height,width,img_gray_h, "Gray Image");

	// Apply Gaussian Blur to grayscale image
	convolutionGPUWrap(img_blurred_d, img_gray_d, width, height, gaussian_kernel_d, FILTER_WIDTH);
	// save image
	// cudaMemcpy(img_gray_h, img_blurred_d, img_gray_size_h, cudaMemcpyDeviceToHost);
	// saveImage(height, width, img_gray_h, "debug/blurred_cuda.jpg");

	// Sobel X
	convolutionGPUWrap(img_sobel_x_d, img_blurred_d, width, height, sobel_x_kernel_d, 3);
	// separableConvolutionKernelWrap(img_blurred_d, img_sobel_x_d, width, height, sobel_x_separable_2_d, sobel_x_separable_d, 3);
	// float *img_sobel_x_h = (float *)malloc(img_gray_size_h);
	// cudaMemcpy(img_sobel_x_h, img_sobel_x_d, img_gray_size_h, cudaMemcpyDeviceToHost);
	// saveImage(height, width, img_sobel_x_h, "debug/sobel_x_cuda.jpg");

	// Sobel Y
	convolutionGPUWrap(img_sobel_y_d, img_blurred_d, width, height, sobel_y_kernel_d, 3);
	// separableConvolutionKernelWrap(img_blurred_d, img_sobel_y_d, width, height, sobel_x_separable_d, sobel_x_separable_2_d, 3);
	// float *img_sobel_y_h = (float *)malloc(img_gray_size_h);
	// cudaMemcpy(img_sobel_y_h, img_sobel_y_d, img_gray_size_h, cudaMemcpyDeviceToHost);
	// saveImage(height, width, img_sobel_y_h, "debug/sobel_y_cuda.jpg");

	// Exeuting the CV task based on the mode
	switch (mode)
	{
	case HARRIS:

		// harrisCornerDetector(&img, img_sobel_x_d, img_sobel_y_d, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH, false);
		harrisMainKernelWrap((uchar4 *)img.data, img_d, img_sobel_x_d, img_sobel_y_d, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH, false);
		break;
	case SHI_TOMASI:
		// harrisCornerDetector(&img, img_sobel_x_d, img_sobel_y_d, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH, true);
		harrisMainKernelWrap((uchar4 *)img.data, img_d, img_sobel_x_d, img_sobel_y_d, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH, true);
		break;
	case CANNY:
		high_threshold = otsu_threshold(img_blurred_d, width, height);
		low_threshold = high_threshold / 2;
		cannyMainKernelWrap((uchar4 *)img.data, img_d, img_sobel_x_d, img_sobel_y_d, width, height, low_threshold, high_threshold, gaussian_kernel_d, FILTER_WIDTH);
		break;
	case CANNY_MANUAL:
		cannyMainKernelWrap((uchar4 *)img.data, img_d, img_sobel_x_d, img_sobel_y_d, width, height, low_threshold, high_threshold, gaussian_kernel_d, FILTER_WIDTH);
		break;
	case CANNY_GUI:
	{
		int thresh_h = 100;
		int thresh_l = 50;
		cv::namedWindow("Output Image", cv::WINDOW_NORMAL);
		cv::createTrackbar("Threshold High", "Output Image", &thresh_h, 255);
		cv::createTrackbar("Threshold Low", "Output Image", &thresh_l, 255);
		while (true)
		{
			cannyMainKernelWrap((uchar4 *)img.data, img_d, img_sobel_x_d, img_sobel_y_d, width, height, thresh_l, thresh_h, gaussian_kernel_d, FILTER_WIDTH);
			cv::imshow("Output Image", img);
			if (cv::waitKey(1) == 27) // wait to press 'esc' key
			{
				break;
			}
		}
		break;
	}
	case OTSU_BIN:
		int threshold = otsu_threshold(img_gray_d, width, height);
		binarize_img_wrapper(img.data, img_gray_d, width, height, threshold);
		break;
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	cout << "Execution time: " << duration.count() << "ms" << endl;

	// Showing the result
	if (mode != CANNY_GUI)
	{
		cv::Mat img_out;
		// Since otsu binarization is done on the grayscale image, we need to convert it to 8UC1(8 unsigned char 1 channel) before displaying
		if (mode == OTSU_BIN)
		{
			img_out = cv::Mat(height, width, CV_8UC1, img.data);
		}
		else
		{
			img_out = cv::Mat(height, width, CV_8UC4, img.data);
		}
		string window_name = "Output Image " + to_string(mode);
		// string filesave = "debug/" + to_string(mode) + "_cuda.jpg";
		cv::cvtColor(img_out, img_out, cv::COLOR_RGBA2BGR);
		cv::imshow(window_name, img_out);
		// cv::imwrite(filesave, img_out);

		// If not from video, wait for key press
		if (!from_video)
		{
			cv::waitKey(0);
		}
		cudaHostUnregister(img.data);
		img.release();
	}

	// (cuda)memory deallocations
	cudaFree(img_d);

	cudaFree(img_gray_d);
	cudaFree(img_blurred_d);
	cudaFree(gaussian_kernel_d);
	cudaFree(sobel_x_kernel_d);
	cudaFree(sobel_y_kernel_d);
	cudaFree(sobel_x_separable_d);
	cudaFree(sobel_x_separable_2_d);
	cudaFree(img_sobel_x_d);
	cudaFree(img_sobel_y_d);
	cudaFree(img_harris_d);
	// Error checking
	free(img_gray_h);
	free(img_harris_h);
	free(img_grayh2);
	free(gaussian_kernel);
	free(img_h);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Error in main kernels: %s\n", cudaGetErrorString(err));
	}
}
void handle_video(enum Mode mode, std::string filename, int low_threshold, int high_threshold, bool is_all_thread = false)
{
	cv::VideoCapture cap(filename);
	if (!cap.isOpened())
	{
		std::cerr << "Error: Unable to load video." << std::endl;
		return;
	}
	int desired_fps = 60;
	int target_frame_time = 1000 / desired_fps;
	int debug = 0;
	while (cap.isOpened())
	{
		int64 start_time = cv::getTickCount();
		cv::Mat img;
		cap >> img;
		if (img.empty())
		{
			break;
		}
		if (is_all_thread)
		{
			cv::Mat img2 = img.clone();
			cv::Mat img3 = img.clone();
			std::thread t1(
				[img](Mode mode, std::string filename, int low_threshold, int high_threshold)
				{
					handle_image(mode, filename, low_threshold, high_threshold, true, img);
				},
				HARRIS, filename, low_threshold, high_threshold);
			std::thread t2(
				[img2](Mode mode, std::string filename, int low_threshold, int high_threshold)
				{
					handle_image(mode, filename, low_threshold, high_threshold, true, img2);
				},
				CANNY, filename, low_threshold, high_threshold);
			std::thread t3(
				[img3](Mode mode, std::string filename, int low_threshold, int high_threshold)
				{
					handle_image(mode, filename, low_threshold, high_threshold, true, img3);
				},
				OTSU_BIN, filename, low_threshold, high_threshold);
			t1.join();
			t2.join();
			t3.join();
		}
		else
		{
			handle_image(mode, filename, low_threshold, high_threshold, true, img);
			// free(img.data);
		}

		int64 end_time = cv::getTickCount();
		double elapsed_time_ms = (end_time - start_time) * 1000 / cv::getTickFrequency();
		cout << "FPS: " << 1000 / elapsed_time_ms << endl;

		int delay = std::max(1, target_frame_time - static_cast<int>(elapsed_time_ms));
		if (cv::waitKey(delay) == 27) // 27=esc key
		{
			break;
		}
		// debug++;
		// if(debug == 2){
		// 	break;
		// }
	}
}
int main(const int argc, const char **argv)
{
	enum Mode mode;
	bool is_video = false;
#pragma region Arguments Parsing
	if (argc < 3)
	{
		fprintf(stderr, "Not enough arguments, at least 3 are required. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
		return -1;
	}
	if (strcmp(argv[1], "-H") == 0)
	{
		mode = HARRIS;
	}
	else if (strcmp(argv[1], "-C") == 0)
	{
		mode = CANNY;
	}
	else if (strcmp(argv[1], "-O") == 0)
	{
		mode = OTSU_BIN;
	}
	else if (strcmp(argv[1], "-S") == 0)
	{
		mode = SHI_TOMASI;
	}
	else if (strcmp(argv[1], "-A") == 0)
	{
		mode = ALL;
	}
	else
	{
		fprintf(stderr, "No execution mode specified. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
		return -1;
	}

	std::string filename = "";
	std::string arg = argv[2];
	if (arg.substr(0, 3) == "-f=")
	{
		filename = arg.substr(3);
		if (filename == "")
		{
			fprintf(stderr, "Empty filename. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
			return -1;
		}

		std::string ext = filename.substr(filename.find_last_of(".") + 1);
		if (ext != "jpg" && ext != "png" && ext != "mp4")
		{
			fprintf(stderr, "Invalid file extension. Only jpg, png and mp4 are supported. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
			return -1;
		}
		if (ext == "mp4")
		{
			is_video = true;
		}
	}
	else
	{
		fprintf(stderr, "No file specified. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
		return -1;
	}

	int low_threshold = 0;
	int high_threshold = 0;
	if (argc > 3)
	{
		if (mode == CANNY)
		{
			std::string arg = argv[3];
			if (arg == "-g")
			{
				if (is_video)
				{
					fprintf(stderr, "Cannot use GUI thresholding with videos. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
					return -1;
				}
				mode = CANNY_GUI;
			}
			else
			{
				if (argc < 5)
				{
					fprintf(stderr, "No -G specified, implying manual thresholding. Not enough parameters. -l and -h required. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
					return -1;
				}
				if (arg.substr(0, 3) == "-l=")
				{
					if (arg.substr(3) == "")
					{
						fprintf(stderr, "Invalid low threshold. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
						return -1;
					}
					try
					{
						low_threshold = std::stoi(arg.substr(3));
					}
					catch (const std::exception &e)
					{
						fprintf(stderr, "Invalid low threshold. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
						return -1;
					}
					mode = CANNY_MANUAL;
				}
				else
				{
					fprintf(stderr, "You need to specify a low threshold with \"-l=<int>\". Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
					return -1;
				}
				arg = argv[4];
				if (arg.substr(0, 3) == "-h=")
				{
					if (arg.substr(3) == "")
					{
						fprintf(stderr, "Invalid high threshold. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
						return -1;
					}
					try
					{
						high_threshold = std::stoi(arg.substr(3));
					}
					catch (const std::exception &e)
					{
						fprintf(stderr, "Invalid high threshold. Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
						return -1;
					}
				}
				else
				{
					fprintf(stderr, "You need to specify a high threshold with \"-l=<int>\". Usage: %s -C -f=filename [-G | [-l=low_threshold -h=high_threshold]\n", argv[0]);
					return -1;
				}
				if (argc > 5)
				{
					fprintf(stderr, "Too many arguments for the specified mode. Ignoring extra arguments.\n");
				}
			}
		}
		else
		{
			fprintf(stderr, "Too many arguments for the specified mode. Ignoring extra arguments.\n");
		}
	}
#pragma endregion
#pragma region Driver Code
	if (mode == ALL)
	{
		if (is_video)
		{
			handle_video(HARRIS, filename, low_threshold, high_threshold, true);
		}
		else
		{
			fprintf(stderr, "All mode is only supported for videos. Usage: %s -A -f=filename\n", argv[0]);
			return -1;
		}
	}
	else
	{
		if (is_video)
		{
			handle_video(mode, filename, low_threshold, high_threshold);
		}
		else
		{
			handle_image(mode, filename, low_threshold, high_threshold);
		}
	}
#pragma endregion
	return 0;
}
// g++ -std=c++11 -IC:C:\opencv\opencv\build\include  -LC:C:\opencv\opencv\build\x64\vc15\lib -lopencv_core470 -lopencv_highgui470 -lopencv_imgcodecs470 -lopencv_imgproc470 -o my_program.exe main.cpp
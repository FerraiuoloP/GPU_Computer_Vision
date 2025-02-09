#pragma once
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
cv::Mat harrisCornerDetectorCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH);
cv::Mat otsuBinarization(cv::Mat *img);
cv::Mat cannyEdgeDetectionCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH);
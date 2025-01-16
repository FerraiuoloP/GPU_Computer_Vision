#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
void harrisCornerDetector(cv::Mat *img, float *sobel_x, float *sobel_y, int width, int height, float K, float ALPHA, float *gaussian_kernel_d, int FILTER_WIDTH, bool shi_tomasi = false);
void cannyEdgeDetector(cv::Mat *img, float *sobel_x, float *sobel_y, int width, int height, float low_threshold, float high_threshold, float *gaussian_kernel_d, int FILTER_WIDTH);
cv::Mat harrisCornerDetectorCPU(cv::Mat *img, float *gaussian_kernel, float *sobel_x_kernel, float *sobel_y_kernel, int FILTER_WIDTH);
cv::Mat cannyEdgeDetectionCPU(cv::Mat *img, float *sobel_x_kernel, float *sobel_y_kernel, int FILTER_WIDTH);
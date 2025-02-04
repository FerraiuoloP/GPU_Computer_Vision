// #include "utils.h"
#include <string>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// include PI
#include <cmath>

using namespace std;
using namespace cv;
float *computeGaussianKernel(int filterWidth, float filterSigma)
{
    float *host_filter;

    // host_filter = new float[filterWidth * filterWidth];
    host_filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));

    float filterSum = 0.f;

    for (int r = -filterWidth / 2; r <= filterWidth / 2; ++r)
    {
        for (int c = -filterWidth / 2; c <= filterWidth / 2; ++c)
        {
            float filterValue = expf(-(float)(c * c + r * r) / (2.f * filterSigma * filterSigma));
            filterValue = filterValue / (2.f * M_PI * filterSigma * filterSigma);
            (host_filter)[(r + filterWidth / 2) * filterWidth + c + filterWidth / 2] = filterValue;
            filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.f / filterSum;

    for (int r = -filterWidth / 2; r <= filterWidth / 2; ++r)
    {
        for (int c = -filterWidth / 2; c <= filterWidth / 2; ++c)
        {
            (host_filter)[(r + filterWidth / 2) * filterWidth + c + filterWidth / 2] *= normalizationFactor;
        }
    }
    return host_filter;
}
void saveImage(int height, int width, float *img, string name)
{
    cv::Mat printImage(height, width, CV_32F, img);
    cv::Mat displayImage1;
    printImage.convertTo(displayImage1, CV_8UC1, 1.0);
    cv::imwrite(name, displayImage1);
}
void showImage(int height, int width, float *img, string name)
{
    cv::Mat printImage(height, width, CV_32F, img);
    cv::Mat displayImage1;
    printImage.convertTo(displayImage1, CV_8UC1, 1.0);
    cv::imshow(name, displayImage1);
    cv::waitKey(0);
}
void showImage2(int height, int width, float *img, string name)
{
    // pixels are in the range 0-1
    cv::Mat printImage(height, width, CV_32F, img);
    cv::Mat displayImage1;
    printImage.convertTo(displayImage1, CV_8UC1, 255);
    cv::imshow(name, displayImage1);
    cv::waitKey(0);
}

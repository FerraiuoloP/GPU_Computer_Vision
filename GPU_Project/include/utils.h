#pragma once
#include <string>
float *computeGaussianKernel(int filterWidth, float filterSigma);
void saveImage(int height, int width, float *img, std::string name);
void showImage(int height, int width, float *img, std::string name);
void showImage2(int height, int width, float *img, std::string name);
void showImageCPU(cv::Mat img);
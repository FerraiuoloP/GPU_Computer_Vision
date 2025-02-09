#pragma once
const int FILTER_WIDTH = 5;
const int FILTER_RADIUS = FILTER_WIDTH / 2;
// const float FILTER_SIGMA = 1.75f;
const float FILTER_SIGMA = 1.5f;
// const float FILTER_SIGMA = 3;
const float ALPHA = 0.05;
const float K = 0.05;

void rgbToGrayKernelWrap(uchar4 *img_d, float *gray_d, int N, int M);
void harrisCornerKernelWrap(float *img_sobel_x, float *img_sobel_y, float *img_harris, int width, int height, float k);
float harrisMainKernelWrap(uchar4 *img_data_h, uchar4 *img_data_d, float *sobel_x, float *sobel_y, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size, bool shi_tomasi, float *harris_map);
void cannyMainKernelWrap(uchar4 *img_data_h, uchar4 *img_data_d, float *sobel_x, float *sobel_y, int width, int height, float low_threshold, float high_threshold, float *gaussian_kernel, int g_kernel_size, bool is_video);
void convolutionGPUWrap(float *d_Result, float *d_Data, int data_w, int data_h, float *d_kernel, int kernel_size);
void separableConvolutionKernelWrap(float *img_d, float *img_out_d, int width, int height, float *kernel_x, float *kernel_y, int kernel_size);
int otsuThreshold(float *image, int width, int height);
void binarizeImgWrapper(unsigned char *img_h, float *img_d, int width, int height, int threshold);
int mapCommonKernelWrap(const float *harris1, const float *harris2, int width, int height, float threshold, float tollerance, int window, int *d_idx1Mapping, int *d_idx2Mapping);

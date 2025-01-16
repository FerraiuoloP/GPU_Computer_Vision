// #include "utils.h"
#include <string>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include "../include/cuda_kernel.cuh"
#include "../include/utils.h"
using namespace std;
using namespace cv;

void harrisCornerDetector(cv::Mat *img, float *sobel_x, float *sobel_y, int width, int height, float K, float ALPHA, float *gaussian_kernel_d, int FILTER_WIDTH, bool shi_tomasi)
{
    float *img_harris_d, *img_harris_h;
    size_t img_gray_size_h = width * height * sizeof(float);

    img_harris_h = (float *)malloc(img_gray_size_h);
    cudaMalloc(&img_harris_d, img_gray_size_h);

    harrisMainKernelWrap(sobel_x, sobel_y, img_harris_d, width, height, K, ALPHA, gaussian_kernel_d, FILTER_WIDTH, shi_tomasi);

    // harris to host
    cudaMemcpy(img_harris_h, img_harris_d, img_gray_size_h, cudaMemcpyDeviceToHost);

    // Measure time
    auto start = chrono::high_resolution_clock::now();
    // finding max and min in the Harris response map
    float max = -100000000;
    float min = 1000000;
    for (int i = 0; i < width * height; i++)
    {
        if (img_harris_h[i] > max)
        {
            max = img_harris_h[i];
        }
        if (img_harris_h[i] < 0.)
        {
            img_harris_h[i] = 0.;
        }

        if (img_harris_h[i] < min)
        {
            min = img_harris_h[i];
        }
    }
    auto end = chrono::high_resolution_clock::now();

    // Nanoseconds in millis
    auto nanosec = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    float millisec = (float)nanosec / 1000000;
    printf("Time taken to find max and min: %f ms\n", millisec);
    cout << "Max value: " << max << endl;
    cout << "Min value: " << min << endl;

    // corner thresholding
    for (int i = 0; i < width * height; i++)
    {
        if (img_harris_h[i] > ALPHA * max && i % width > 1 && i % width < width - 1 && i / width > 1 && i / width < height - 1)
        {
            // coloring 2x2 pixels area around the corner
            for (int j = -2; j < 2; j++)
            {
                for (int k = -2; k < 2; k++)
                {
                    int x = (i % width) + k;
                    int y = (i / width) + j;
                    if (x >= 0 && x < width && y >= 0 && y < height)
                    {
                        img->at<cv::Vec3b>(y, x) = cv::Vec3b(240, 0, 0);
                    }
                }
            }
        }
    }
    cudaFree(img_harris_d);
    free(img_harris_h);
}

void cannyEdgeDetector(cv::Mat *img, float *sobel_x, float *sobel_y, int width, int height, float low_threshold, float high_threshold, float *gaussian_kernel_d, int FILTER_WIDTH)
{
    // float *img_sobel_x, *img_sobel_y, *img_sobel, *sobel_directions, *img_canny_d;
    // size_t img_gray_size_h = width * height * sizeof(float);
    // float *img_debug_h = (float *)malloc(img_gray_size_h);

    // cudaMalloc(&img_sobel_x, img_gray_size_h);
    // cudaMalloc(&img_sobel_y, img_gray_size_h);
    // cudaMalloc(&img_sobel, img_gray_size_h);
    // cudaMalloc(&sobel_directions, img_gray_size_h);
    // cudaMalloc(&img_canny_d, img_gray_size_h);

    // // cannyMainKernelWrap(sobel_x, sobel_y, img_canny_d, width, height, low_threshold, high_threshold, gaussian_kernel_d, FILTER_WIDTH);

    // // // copy the sobel image to the host
    // cudaMemcpy(img_debug_h, img_canny_d, img_gray_size_h, cudaMemcpyDeviceToHost);
    // // showImage(height, width, img_debug_h, "Sobel Combined");

    // // Copy img_debug_h to img.data using cv methods
    // for (int i = 0; i < width * height; i++)
    // {
    //     img->at<cv::Vec3b>(i / width, i % width) = cv::Vec3b(img_debug_h[i], img_debug_h[i], img_debug_h[i]);
    // }

    // cudaFree(img_sobel_x);
    // cudaFree(img_sobel_y);
    // cudaFree(img_sobel);
    // cudaFree(sobel_directions);
    // cudaFree(img_canny_d);

    // free(img_debug_h);
}

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

// #define MEASURE_TIME

/**
 * @brief Shows an image using OpenCV
 *
 * @param img Input image
 */
void showImageCPU(cv::Mat img)
{
    cv::Mat displayImage1;
    img.convertTo(displayImage1, CV_8UC1, 1.0);
    cv::imshow("Image", displayImage1);
    cv::waitKey(0);
}
/**
 * @brief Trivial CPU Convolution implementation
 *
 * @param inputImage Input image
 * @param kernel Filter kernel
 * @param kernelSize Kernel size
 * @return cv::Mat Output image
 */
cv::Mat applyConvolutionCPU(const cv::Mat &inputImage, const float *kernel, int kernelSize);
cv::Mat applyConvolutionCPU(const cv::Mat &inputImage, const float *kernel, int kernelSize)
{
    if (kernelSize % 2 == 0)
    {
        std::cerr << "Error: Kernel size must be odd." << std::endl;
        return cv::Mat();
    }
    int pad = kernelSize / 2;

    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
#ifdef MEASURE_TIME
    double start = cv::getTickCount();
#endif
    for (int y = pad; y < inputImage.rows - pad; y++)
    {
        for (int x = pad; x < inputImage.cols - pad; x++)
        {
            float pixelValue = 0.0;

            // convolution
            for (int ky = -pad; ky <= pad; ky++)
            {
                for (int kx = -pad; kx <= pad; kx++)
                {
                    int kernelIndex = (ky + pad) * kernelSize + (kx + pad);
                    float kernelValue = kernel[kernelIndex];

                    int imageY = y + ky;
                    int imageX = x + kx;

                    pixelValue += inputImage.at<float>(imageY, imageX) * kernelValue;
                }
            }
            outputImage.at<float>(y, x) = float(pixelValue);
        }
    }
#ifdef MEASURE_TIME
    double end = cv::getTickCount();
    double time = (end - start) / cv::getTickFrequency();
    cout << "Convolution CPU time: " << time * 1000 << "ms" << endl;
#endif

    return outputImage;
}

/**
 * @brief Computes the optimal otsu threshold of a given image
 *
 * @param image  Input image
 * @return int Optimal Otsu threshold
 */
int otsuThreshold(cv::Mat &image);
int otsuThreshold(cv::Mat &image)
{
    int hist[256] = {0};
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            hist[(int)image.at<uchar>(i, j)]++;
        }
    }
    int total = image.rows * image.cols;
    float sum = 0;
    for (int i = 0; i < 256; i++)
    {
        sum += i * hist[i];
    }
    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float varMax = 0;
    int threshold = 0;
    for (int i = 0; i < 256; i++)
    {
        wB += hist[i];
        if (wB == 0)
            continue;
        wF = total - wB;
        if (wF == 0)
            break;
        sumB += (float)(i * hist[i]);
        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;
        float varBetween = (float)wB * (float)wF * (mB - mF) * (mB - mF);
        if (varBetween > varMax)
        {
            varMax = varBetween;
            threshold = i;
        }
    }
    return threshold;
}

/**
 * @brief Applies Harris Corner Detection on an image
 *
 * @param img Input image
 * @param gaussian_kernel Gaussian kernel
 * @param sobel_x_kernel  Sobel x kernel
 * @param sobel_y_kernel  Sobel y kernel
 * @param FILTER_WIDTH Filter width of the gaussian kernel
 * @return cv::Mat Image with Harris corners marked in red
 */
cv::Mat harrisCornerDetectorCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH)

{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img_gray(img->rows, img->cols, CV_32F);
    // rgb to grayscale
    for (int i = 0; i < img->rows; i++)
    {
        for (int j = 0; j < img->cols; j++)
        {
            cv::Vec3b pixel = img->at<cv::Vec3b>(i, j);
            img_gray.at<float>(i, j) = 0.299 * float(pixel[0]) + 0.587 * float(pixel[1]) + 0.114 * float(pixel[2]);
        }
    }
    // cv::imwrite("debug/gray_cpu.jpg", img_gray);
    // showImage(img_gray);

    // apply Gaussian Blur
    cv::Mat img_blurred(img_gray.rows, img_gray.cols, CV_32F);
    img_blurred = applyConvolutionCPU(img_gray, gaussian_kernel, FILTER_WIDTH);
    // cv::imwrite("debug/blurred_cpu.jpg", img_blurred);

    // showImage(img_blurred);

    // computing the sobel x and y gradients
    cv::Mat sobel_x, sobel_y;
    sobel_x = applyConvolutionCPU(img_blurred, sobel_x_kernel, 3);
    // cv::imwrite("debug/sobel_x_cpu.jpg", sobel_x);
    sobel_y = applyConvolutionCPU(img_blurred, sobel_y_kernel, 3);
    // cv::imwrite("debug/sobel_y_cpu.jpg", sobel_y);

    // Computing harris response map
    cv::Mat img_harris = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    for (int i = 1; i < img_blurred.rows - 1; i++)
    {
        for (int j = 1; j < img_blurred.cols - 1; j++)
        {
            float Ix2 = sobel_x.at<float>(i, j) * sobel_x.at<float>(i, j);
            float Iy2 = sobel_y.at<float>(i, j) * sobel_y.at<float>(i, j);
            float Ixy = sobel_x.at<float>(i, j) * sobel_y.at<float>(i, j);
            float det = Ix2 * Iy2 - Ixy * Ixy;
            float trace = Ix2 + Iy2;
            // img_harris.at<float>(i, j) = det - 0.05 * trace * trace;
            if (trace != 0)
            {

                img_harris.at<float>(i, j) = det / (trace);
            }
            else
            {
                img_harris.at<float>(i, j) = 0;
            }
            // cout << img_harris.at<float>(i, j) << endl;
        }
    }

    // save harris response map
    // cv::imwrite("debug/harris_cpu.jpg", img_harris);

    // normalize the image
    // cv::normalize(img_harris, img_harris, 0, 1, cv::NORM_MINMAX);

    // finding max in harris response map
    float max = -100000000;
    float min = 1000000;
    // find max
    for (int i = 0; i < img_harris.rows; i++)
    {
        for (int j = 0; j < img_harris.cols; j++)
        {
            if (img_harris.at<float>(i, j) > max)
            {
                max = img_harris.at<float>(i, j);
            }
            if (img_harris.at<float>(i, j) < min)
            {
                min = img_harris.at<float>(i, j);
            }
        }
    }

    // showImage(img_harris);
    // print max value

    // NMS
    for (int i = 1; i < img_harris.rows - 1; i++)
    {
        for (int j = 1; j < img_harris.cols - 1; j++)
        {
            float local_max = -10000;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (img_harris.rows > i + k && img_harris.cols > j + l && i + k >= 0 && j + l >= 0 && img_harris.at<float>(i + k, j + l) > local_max)
                    {
                        local_max = img_harris.at<float>(i + k, j + l);
                    }
                }
            }
            if (img_harris.at<float>(i, j) < local_max)
            {
                img_harris.at<float>(i, j) = 0;
            }
        }
    }

    // corner thresholding
    for (int i = 0; i < img_harris.rows; i++)
    {
        for (int j = 0; j < img_harris.cols; j++)
        {
            // if the pixel is a corner
            if (img_harris.at<float>(i, j) > 0.03 * max)
            {
                // color 2x2 pixels
                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        int x = j + k;
                        int y = i + l;
                        if (x >= 0 && x < img_harris.cols && y >= 0 && y < img_harris.rows)
                        {
                            img->at<cv::Vec3b>(y, x) = cv::Vec3b(240, 0, 0);
                        }
                    }
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Harris CPU time: " << duration.count() << "ms" << endl;

    return *img;
}

/**
 * @brief Binirizes an image using Otsu's method
 *
 * @param img Input image
 * @return cv::Mat  Binarized image
 */
cv::Mat otsuBinarization(cv::Mat *img)
{
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img_gray(img->rows, img->cols, CV_32F);
    // rgb to grayscale
    for (int i = 0; i < img->rows; i++)
    {
        for (int j = 0; j < img->cols; j++)
        {
            cv::Vec3b pixel = img->at<cv::Vec3b>(i, j);
            img_gray.at<float>(i, j) = 0.299 * float(pixel[0]) + 0.587 * float(pixel[1]) + 0.114 * float(pixel[2]);
        }
    }

    // otsu thresholding
    int threshold = otsuThreshold(img_gray);
    // cout << "Threshold: " << threshold << endl;

    // binarize the image
    for (int i = 0; i < img_gray.rows; i++)
    {
        for (int j = 0; j < img_gray.cols; j++)
        {
            if (img_gray.at<float>(i, j) > threshold)
            {
                img_gray.at<float>(i, j) = 255;
            }
            else
            {
                img_gray.at<float>(i, j) = 0;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Otsu CPU time: " << duration.count() << "ms" << endl;
    return img_gray;
}

/**
 * @brief Applies Canny Edge Detection on an image
 *
 * @param img Input image
 * @param gaussian_kernel Gaussian kernel
 * @param sobel_x_kernel  Sobel x kernel
 * @param sobel_y_kernel Sobel y kernel
 * @param FILTER_WIDTH Filter width of the gaussian kernel
 * @return cv::Mat Canny edge detected image
 */
cv::Mat cannyEdgeDetectionCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH)
{
    // rgb to grayscale
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat img_gray(img->rows, img->cols, CV_32F);
    for (int i = 0; i < img->rows; i++)
    {
        for (int j = 0; j < img->cols; j++)
        {
            cv::Vec3b pixel = img->at<cv::Vec3b>(i, j);
            img_gray.at<float>(i, j) = 0.299 * float(pixel[0]) + 0.587 * float(pixel[1]) + 0.114 * float(pixel[2]);
        }
    }
    // cv::imwrite("debug/gray_cpu.jpg", img_gray);

    // apply Gaussian Blur
    cv::Mat img_blurred(img_gray.rows, img_gray.cols, CV_32F);
    img_blurred = applyConvolutionCPU(img_gray, gaussian_kernel, FILTER_WIDTH);
    // cv::imwrite("debug/blurred_cpu.jpg", img_blurred);

    // computing the sobel x and y gradients
    cv::Mat sobel_x, sobel_y;
    sobel_x = applyConvolutionCPU(img_blurred, sobel_x_kernel, 3);
    // cv::imwrite("debug/sobel_x_cpu.jpg", sobel_x);
    sobel_y = applyConvolutionCPU(img_blurred, sobel_y_kernel, 3);
    // cv::imwrite("debug/sobel_y_cpu.jpg", sobel_y);

    // computing the magnitude and direction of the gradient
    cv::Mat magnitude = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    cv::Mat direction = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    for (int i = 0; i < img_blurred.rows; i++)
    {
        for (int j = 0; j < img_blurred.cols; j++)
        {
            magnitude.at<float>(i, j) = sqrt(sobel_x.at<float>(i, j) * sobel_x.at<float>(i, j) + sobel_y.at<float>(i, j) * sobel_y.at<float>(i, j));
            direction.at<float>(i, j) = atan2(sobel_y.at<float>(i, j), sobel_x.at<float>(i, j));
        }
    }
    // cv::imwrite("debug/combined_gradients_cpu.jpg", magnitude);

    float highThreshold = float(otsuThreshold(img_blurred));
    float lowThreshold = highThreshold / 2;
    // NMS(lowerboud+double thresholding)
    cv::Mat nonMaxSuppressed = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    for (int i = 1; i < img_blurred.rows - 1; i++)
    {
        for (int j = 1; j < img_blurred.cols - 1; j++)
        {
            float angle = direction.at<float>(i, j) * 180 / M_PI;
            if ((angle >= -22.5 && angle < 22.5) || (angle >= 157.5 && angle <= 180) || (angle >= -180 && angle < -157.5))
            {
                if (magnitude.at<float>(i, j) > magnitude.at<float>(i, j + 1) && magnitude.at<float>(i, j) > magnitude.at<float>(i, j - 1))
                {
                    nonMaxSuppressed.at<float>(i, j) = magnitude.at<float>(i, j);
                }
            }
            else if ((angle >= 22.5 && angle < 67.5) || (angle >= -157.5 && angle < -112.5))
            {
                if (magnitude.at<float>(i, j) > magnitude.at<float>(i - 1, j + 1) && magnitude.at<float>(i, j) > magnitude.at<float>(i + 1, j - 1))
                {
                    nonMaxSuppressed.at<float>(i, j) = magnitude.at<float>(i, j);
                }
            }
            else if ((angle >= 67.5 && angle < 112.5) || (angle >= -112.5 && angle < -67.5))
            {
                if (magnitude.at<float>(i, j) > magnitude.at<float>(i - 1, j) && magnitude.at<float>(i, j) > magnitude.at<float>(i + 1, j))
                {
                    nonMaxSuppressed.at<float>(i, j) = magnitude.at<float>(i, j);
                }
            }
            else if ((angle >= 112.5 && angle < 157.5) || (angle >= -67.5 && angle < -22.5))
            {
                if (magnitude.at<float>(i, j) > magnitude.at<float>(i - 1, j - 1) && magnitude.at<float>(i, j) > magnitude.at<float>(i + 1, j + 1))
                {
                    nonMaxSuppressed.at<float>(i, j) = magnitude.at<float>(i, j);
                }
            }
            // if greater than high_threshold, set to 255
            // if greater than low_threshold, set to 128
            // else set to 0
            if (nonMaxSuppressed.at<float>(i, j) > highThreshold)
            {
                nonMaxSuppressed.at<float>(i, j) = 255;
            }
            else if (nonMaxSuppressed.at<float>(i, j) > lowThreshold)
            {
                nonMaxSuppressed.at<float>(i, j) = 128;
            }
            else
            {
                nonMaxSuppressed.at<float>(i, j) = 0;
            }
        }
    }

    // cv::imwrite("debug/non_max_suppressed_cpu.jpg", nonMaxSuppressed);

    cv::Mat img_canny = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    // float highThreshold = 40;

    // cout << "Threshold: " << highThreshold << endl;

    // hysteresis
    for (int i = 1; i < img_blurred.rows - 1; i++)
    {
        for (int j = 1; j < img_blurred.cols - 1; j++)
        {
            if (nonMaxSuppressed.at<float>(i, j) >= 255)
            {
                img_canny.at<float>(i, j) = 255;
            }
            else if (nonMaxSuppressed.at<float>(i, j) >= 128)
            {
                bool is_connected_to_strong = false;
                for (int k = -1; k <= 1 && !is_connected_to_strong; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        if (nonMaxSuppressed.at<float>(i + k, j + l) >= 255)
                        {
                            // img_canny.at<float>(i, j) = 255;
                            is_connected_to_strong = true;
                            break;
                        }
                    }
                }
                if (is_connected_to_strong)
                {
                    img_canny.at<float>(i, j) = 1;
                }
                else
                {
                    img_canny.at<float>(i, j) = 0;
                }
            }
            else
            {
                img_canny.at<float>(i, j) = 0;
            }
        }
    }
    // save it
    // cv::imwrite("debug/2_cpu.jpg", img_canny);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "Canny CPU time: " << duration.count() << "ms" << endl;

    return img_canny;
}

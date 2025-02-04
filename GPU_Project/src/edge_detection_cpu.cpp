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

void showImageCPU(cv::Mat img)
{
    cv::Mat displayImage1;
    img.convertTo(displayImage1, CV_8UC1, 1.0);
    cv::imshow("Image", displayImage1);
    cv::waitKey(0);
}

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

    return outputImage;
}
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
 * Computes Harris corner detector on the cpu given an image
 *
 */
cv::Mat harrisCornerDetectorCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH)

{
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
    // showImage(img_gray);

    // apply Gaussian Blur
    cv::Mat img_blurred(img_gray.rows, img_gray.cols, CV_32F);
    img_blurred = applyConvolutionCPU(img_gray, gaussian_kernel, FILTER_WIDTH);

    // showImage(img_blurred);

    // computing the sobel x and y gradients
    cv::Mat sobel_x, sobel_y;
    sobel_x = applyConvolutionCPU(img_blurred, sobel_x_kernel, 3);
    sobel_y = applyConvolutionCPU(img_blurred, sobel_y_kernel, 3);

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
            img_harris.at<float>(i, j) = det / (trace + 0.00000001);
            // cout << img_harris.at<float>(i, j) << endl;
        }
    }

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
    cout << "Max value: " << max << endl;
    cout << "Min value: " << min << endl;

    // NMS
    for (int i = 1; i < img_harris.rows - 1; i++)
    {
        for (int j = 1; j < img_harris.cols - 1; j++)
        {
            float max = 0;
            for (int k = -1; k <= 1; k++)
            {
                for (int l = -1; l <= 1; l++)
                {
                    if (img_harris.rows > i + k && img_harris.cols > j + l && i + k >= 0 && j + l >= 0 && img_harris.at<float>(i + k, j + l) > max)
                    {
                        max = img_harris.at<float>(i + k, j + l);
                    }
                }
            }
            if (img_harris.at<float>(i, j) < max)
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
            if (img_harris.at<float>(i, j) > 0.035 * max)
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

    return *img;
}

cv::Mat otsuBinarization(cv::Mat *img)
{
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

    return img_gray;
}

cv::Mat cannyEdgeDetectionCPU(cv::Mat *img, const float *gaussian_kernel, const float *sobel_x_kernel, const float *sobel_y_kernel, int FILTER_WIDTH)
{
    // rgb to grayscale
    cv::Mat img_gray(img->rows, img->cols, CV_32F);
    for (int i = 0; i < img->rows; i++)
    {
        for (int j = 0; j < img->cols; j++)
        {
            cv::Vec3b pixel = img->at<cv::Vec3b>(i, j);
            img_gray.at<float>(i, j) = 0.299 * float(pixel[0]) + 0.587 * float(pixel[1]) + 0.114 * float(pixel[2]);
        }
    }

    // apply Gaussian Blur
    cv::Mat img_blurred(img_gray.rows, img_gray.cols, CV_32F);
    img_blurred = applyConvolutionCPU(img_gray, gaussian_kernel, FILTER_WIDTH);

    // computing the sobel x and y gradients
    cv::Mat sobel_x, sobel_y;
    sobel_x = applyConvolutionCPU(img_blurred, sobel_x_kernel, 3);
    sobel_y = applyConvolutionCPU(img_blurred, sobel_y_kernel, 3);

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
        }
    }

    cv::Mat img_canny = cv::Mat::zeros(img_blurred.rows, img_blurred.cols, CV_32F);
    // float highThreshold = 40;
    float highThreshold = float(otsuThreshold(img_blurred));
    // cout << "Threshold: " << highThreshold << endl;

    // hysteresis
    float lowThreshold = highThreshold / 2;
    for (int i = 1; i < img_blurred.rows - 1; i++)
    {
        for (int j = 1; j < img_blurred.cols - 1; j++)
        {
            if (nonMaxSuppressed.at<float>(i, j) > highThreshold)
            {
                img_canny.at<float>(i, j) = 1;
            }
            else if (nonMaxSuppressed.at<float>(i, j) > lowThreshold)
            {
                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        if (nonMaxSuppressed.at<float>(i + k, j + l) > highThreshold)
                        {
                            img_canny.at<float>(i, j) = 1;
                        }
                    }
                }
            }
        }
    }

    return img_canny;
}

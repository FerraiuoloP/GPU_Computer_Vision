// nvcc -std=c++14 -ccbin g++ main.cpp src/cuda_kernel.cu -o main `pkg-config --cflags --libs opencv4`
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include "include/utils.h"
#include "include/edge_detection.h"

using namespace cv;
using namespace std;
const int FILTER_WIDTH = 3;
const float FILTER_SIGMA = 1.75f;
const float ALPHA = 0.05;
const float K = 0.05;

enum Mode
{
    // -H. Normal Harris Corner Detection
    HARRIS,
    // -C. Canny Edge Detection with Otsu Thresholding
    CANNY,
    // -O. Otsu thresholding method for image binarization
    OTSU_BIN,

};
const float sobel_x_kernel[9] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
const float sobel_y_kernel[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
void handle_image(enum Mode mode, std::string filename, float *gaussian_kernel, bool from_video = false, cv::Mat img_v = cv::Mat())
{
    cv::Mat img;
    if (from_video)
    {
        img = img_v;
    }
    else
    {
        img = cv::imread(filename, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Error: Unable to load image." << std::endl;
            return;
        }
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // variable declarations
    switch (mode)
    {
    case HARRIS:
        cout << "Harris Corner Detection" << endl;
        img = harrisCornerDetectorCPU(&img, gaussian_kernel, sobel_x_kernel, sobel_y_kernel, FILTER_WIDTH);
        break;
    case CANNY:
        cout << "Canny Edge Detection with Otsu Thresholding" << endl;
        img = cannyEdgeDetectionCPU(&img, gaussian_kernel, sobel_x_kernel, sobel_y_kernel, FILTER_WIDTH);
        // save it to debug/2_cpu.jpg
        cv::imwrite("debug/2_cpu.jpg", img);
        break;
    case OTSU_BIN:
        cout << "Otsu Binarization" << endl;
        img = otsuBinarization(&img);
        break;
    default:
        cout << "Invalid mode" << endl;
        break;
    }

    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imshow("Image", img);
    if (!from_video)
    {
        cv::waitKey(0);
    }
    img.release();
}
void handle_video(enum Mode mode, std::string filename, float *gaussian_kernel)
{
    cv::VideoCapture cap(filename);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to load video." << std::endl;
        return;
    }
    cv::Mat img;
    while (true)
    {
        cap >> img;
        if (img.empty())
        {
            break;
        }
        handle_image(mode, filename, gaussian_kernel, true, img);

        if (cv::waitKey(1) == 27)
        {
            break;
        }
    }
}
int main(const int argc, const char **argv)
{
    enum Mode mode;
    bool is_video = false;
    cv::Mat img;
#pragma region Arguments Parsing
    if (argc < 3)
    {
        fprintf(stderr, "Not enough arguments, at least 3 are required. Usage: %s [-H | -C | -O ] -f=filename\n", argv[0]);
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
    else
    {
        fprintf(stderr, "No execution mode specified. Usage: %s [-H | -C | -O ] -f=filename\n", argv[0]);
        return -1;
    }

    std::string filename = "";
    std::string arg = argv[2];
    if (arg.substr(0, 3) == "-f=")
    {
        filename = arg.substr(3);
        if (filename == "")
        {
            fprintf(stderr, "Empty filename. Usage: %s [-H | -C | -O ] -f=filename\n", argv[0]);
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
        fprintf(stderr, "No file specified. Usage: %s [-H | -C | -O ] -f=filename\n", argv[0]);
        return -1;
    }

    if (argc > 3)
    {
        fprintf(stderr, "Too many arguments. Extra arguments will be ignored. Usage: %s [-H | -C | -O | -S] -f=filename\n", argv[0]);
    }
#pragma endregion

#pragma region driver code
    float *gaussian_kernel = computeGaussianKernel(FILTER_WIDTH, FILTER_SIGMA);
    if (is_video)
    {
        handle_video(mode, filename, gaussian_kernel);
    }
    else
    {
        // measure time
        handle_image(mode, filename, gaussian_kernel);
    }
    free(gaussian_kernel);

    return 0;
}
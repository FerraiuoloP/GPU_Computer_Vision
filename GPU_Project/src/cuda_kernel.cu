#define DEBUG

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <assert.h>
#include "../include/cuda_kernel.cuh"
using namespace std;
#define TILE_WIDTH 16 // 16 X 16 TILE
#define KERNEL_RADIUS 1
inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

__global__ void rgbToGrayKernel(unsigned char *img_d, float *gray_d, int N, int M)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < M)
    {
        int idx = i * M + j;
        int idx2 = 3 * idx;
        float r = (float)(img_d[idx2]) * 0.299;
        float g = (float)(img_d[idx2 + 1]) * 0.587;
        float b = (float)(img_d[idx2 + 2]) * 0.114;

        gray_d[idx] = (r + g + b);
    }
}

__global__ void nonMaximumSuppression_(float *harris_map, float *corners_output, int N, int M, int window_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = i * N + j;
        int k = window_size / 2;
        float max_val = 0.0f;
        for (int u = -k; u <= k; u++)
        {
            for (int v = -k; v <= k; v++)
            {
                int x = i + u;
                int y = j + v;
                if (x >= 0 && x < M && y >= 0 && y < N)
                {
                    max_val = fmaxf(max_val, harris_map[x * N + y]);
                }
            }
        }
        corners_output[idx] = (harris_map[idx] == max_val) ? harris_map[idx] : 0.0f;
    }
}

__global__ void nonMaximumSuppression(float *harris_map, float *corners_output, int N, int M, int window_size)
{
    // Shared memory for the tile and its border
    // 4 because the kernel size is 5, thus 5/2 = 2 and 2*2 = 4. In this way we can load the corners of the tile
    __shared__ float data[TILE_WIDTH + FILTER_WIDTH / 2 * 2][TILE_WIDTH + FILTER_WIDTH / 2 * 2];

    const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int gLoc = x0 + y0 * N;

    // Load corners into shared memory
    if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
    {
        // Top-left corner
        int x = x0 - 1;
        int y = y0 - 1;
        data[threadIdx.y][threadIdx.x] = (x >= 0 && y >= 0) ? harris_map[x + y * N] : 0.0f;

        // Top-right corner
        x = x0 + 1;
        y = y0 - 1;
        data[threadIdx.y][threadIdx.x + 2] = (x < N && y >= 0) ? harris_map[x + y * N] : 0.0f;

        // Bottom-left corner
        x = x0 - 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x] = (x >= 0 && y < M) ? harris_map[x + y * N] : 0.0f;

        // Bottom-right corner
        x = x0 + 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x + 2] = (x < N && y < M) ? harris_map[x + y * N] : 0.0f;
    }

    __syncthreads();

    // Perform convolution
    if (x0 < N && y0 < M)
    {
        float max_val = 0.0f;
        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                // max_val += data[threadIdx.y + 1 + i][threadIdx.x + 1 + j] *
                //            d_Kernel[(i + 1) * 3 + (j + 1)];
                max_val = fmaxf(max_val, data[threadIdx.y + 1 + i][threadIdx.x + 1 + j]);
            }
        }
        corners_output[gLoc] = (data[threadIdx.y + 1][threadIdx.x + 1] == max_val) ? data[threadIdx.y + 1][threadIdx.x + 1] : 0.0f;
    }
}

__global__ void convolutionGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    float *d_Kernel)
{
    // Shared memory for the tile and its border
    __shared__ float data[TILE_WIDTH + FILTER_WIDTH / 2 * 2][TILE_WIDTH + FILTER_WIDTH / 2 * 2];

    // Calculate global memory location of this thread
    const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int gLoc = x0 + y0 * dataW;

    // Load corners into shared memory
    if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
    {
        // Top-left corner
        int x = x0 - 1;
        int y = y0 - 1;
        data[threadIdx.y][threadIdx.x] = (x >= 0 && y >= 0) ? d_Data[x + y * dataW] : 0.0f;

        // Top-right corner
        x = x0 + 1;
        y = y0 - 1;
        data[threadIdx.y][threadIdx.x + 2] = (x < dataW && y >= 0) ? d_Data[x + y * dataW] : 0.0f;

        // Bottom-left corner
        x = x0 - 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x] = (x >= 0 && y < dataH) ? d_Data[x + y * dataW] : 0.0f;

        // Bottom-right corner
        x = x0 + 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x + 2] = (x < dataW && y < dataH) ? d_Data[x + y * dataW] : 0.0f;
    }

    __syncthreads();

    // Perform convolution
    if (x0 < dataW && y0 < dataH)
    {
        float sum = 0.0f;
        for (int i = -1; i <= 1; ++i)
        {
            for (int j = -1; j <= 1; ++j)
            {
                sum += data[threadIdx.y + 1 + i][threadIdx.x + 1 + j] *
                       d_Kernel[(i + 1) * 3 + (j + 1)];
            }
        }
        d_Result[gLoc] = sum;
    }
}

__global__ void vecAdd(float *A, float *B, float *C, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void vecMul(float *A, float *B, float *C, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        C[idx] = A[idx] * B[idx];
    }
}

__global__ void vecSub(float *A, float *B, float *C, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void vecDiv(float *A, float *B, float *C, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        if (B[idx] != 0)
        {
            C[idx] = A[idx] / B[idx];
        }
        else
        {
            C[idx] = 0; // division by zero error handling
        }
    }
}

// N = width, M = height
__global__ void computeShiTommasiResponse(const float *Ixx, const float *Iyy, const float *Ixy, float *corners_output, int M, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        float A = Ixx[idx];
        float B = Ixy[idx];
        float C = Iyy[idx];

        // Compute eigenvalues using determinant and trace
        float trace = A + C;
        float determinant = A * C - B * B;

        float lambda1 = trace / 2 + sqrtf(trace * trace / 4 - determinant);
        float lambda2 = trace / 2 - sqrtf(trace * trace / 4 - determinant);

        corners_output[idx] = fminf(lambda1, lambda2);
    }
}

__global__ void applyThreshold(const float *harris_map, float *corners_output, int N, int M, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = j * M + i;
        int val = harris_map[idx];
        corners_output[idx] = (val > threshold) ? val : 0.0f;
    }
}
__global__ void applyConvolution(float *img_d, float *img_out_d, int N, int M, float *kernel, int kernel_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N)
    {
        int idx = i * N + j;
        int k = kernel_size / 2;
        float sum = 0.0f;
        for (int u = -k; u <= k; u++)
        {
            for (int v = -k; v <= k; v++)
            {
                int x = i + u;
                int y = j + v;
                if (x >= 0 && x < M && y >= 0 && y < N)
                {
                    sum += img_d[x * N + y] * kernel[(u + k) * kernel_size + (v + k)];
                }
            }
        }
        img_out_d[idx] = sum;
    }
}

/**
 * @brief Combine the gradient in x and y directions and compute the magnitude of the gradient along with the direction.
 *
 * @param img_sobel_x Gradient in the x direction.
 * @param img_sobel_y Gradient in the y direction.
 * @param img_sobel Output image containing the magnitude of the gradient for each pixel.
 * @param sobel_directions Output image containing the direction of the gradient for each pixel.
 * @param width for the image.
 * @param height for the image.
 */
__global__ void combineGradientsKernel(float *img_sobel_x, float *img_sobel_y, float *img_sobel, float *sobel_directions, int N, int M)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N)
    {
        int idx = i * N + j;
        float gx = img_sobel_x[idx];
        float gy = img_sobel_y[idx];

        img_sobel[idx] = hypotf(gx, gy);
        // img_sobel[idx] = sqrt((float)(gx * gx + gy * gy));
        sobel_directions[idx] = atan2f(gy, gx);
    }
}

__global__ void lowerBoundCutoffSuppression_sh(float *img_magn_gradient, float *img_dir_gradient, float *img_output, int width, int height)
{
    int x0 = blockIdx.x * blockDim.x + threadIdx.x;
    int y0 = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float img_magn_gradient_shared[TILE_WIDTH + FILTER_WIDTH / 2 * 2][TILE_WIDTH + FILTER_WIDTH / 2 * 2];

    // Load corners into shared memory
    if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
    {

        // Top-left corner
        int x = x0 - 1;
        int y = y0 - 1;
        img_magn_gradient_shared[threadIdx.y][threadIdx.x] = (x >= 0 && y >= 0) ? img_magn_gradient[x + y * width] : 0.0f;

        // Top-right corner
        x = x0 + 1;
        y = y0 - 1;
        img_magn_gradient_shared[threadIdx.y][threadIdx.x + 2] = (x < width && y >= 0) ? img_magn_gradient[x + y * width] : 0.0f;

        // Bottom-left corner
        x = x0 - 1;
        y = y0 + 1;
        img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x] = (x >= 0 && y < height) ? img_magn_gradient[x + y * width] : 0.0f;

        // Bottom-right corner
        x = x0 + 1;
        y = y0 + 1;
        img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x + 2] = (x < width && y < height) ? img_magn_gradient[x + y * width] : 0.0f;
    }
    if (x0 < width && y0 < height)
    {
        int idx = y0 * width + x0;
        // neighbours magnitude in the N/W and S/E directions
        float neigh_1 = 0.0f;
        float neigh_2 = 0.0f;
        // Gradient direction expressed in degrees
        float angle = img_dir_gradient[idx] * 180.0 / M_PI; // Converti in gradi
        if (angle < 0)
            angle += 180.0;
        // If the angle rounded down is 0 or 180 degrees, the edge is vertical thus the neighbours are to the left and right
        if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.))
        // if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
        {
            // neigh_1 = img_magn_gradient[idx + 1];
            neigh_1 = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 2];
            // neigh_2 = img_magn_gradient[idx - 1];
            neigh_2 = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x];
        }
        // If the angle rounded down is 45 degrees, the edge is diagonal thus the neighbours are to the top right and bottom left
        if (angle >= 22.5f && angle < 67.5f)
        {
            // neigh_1 = img_magn_gradient[idx + width - 1];
            neigh_1 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x];
            // neigh_2 = img_magn_gradient[idx - width + 1];
            neigh_2 = img_magn_gradient_shared[threadIdx.y][threadIdx.x + 2];
        }
        // If the angle rounded down is 90 degrees the edge is horizontal thus the neighbours are above and below
        if (angle >= 67.5f && angle < 112.5f)
        {
            // neigh_1 = img_magn_gradient[idx + width];
            neigh_1 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x + 1];
            // neigh_2 = img_magn_gradient[idx - width];
            neigh_2 = img_magn_gradient_shared[threadIdx.y][threadIdx.x + 1];
        }
        // If the angle rounded down is 135 degrees the edge is diagonal thus the neighbours are to the top left and bottom right
        if (angle >= 112.5f && angle < 157.5f)
        {
            // neigh_1 = img_magn_gradient[idx - width - 1];
            neigh_1 = img_magn_gradient_shared[threadIdx.y][threadIdx.x];
            // neigh_2 = img_magn_gradient[idx + width + 1];
            neigh_2 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x + 2];
        }
        // If the pixel is not the maximum, set it to 0
        // if (img_magn_gradient[idx] > neigh_1 && img_magn_gradient[idx] > neigh_2)
        if (img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 1] > neigh_1 && img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 1] > neigh_2)
        {
            // img_output[idx] = img_magn_gradient[idx];
            img_output[idx] = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 1];
        }
        else
        {
            img_output[idx] = 0.0f;
        }
    }
}
/**
 * @brief Compute a non maximum suppression on the gradient magnitude image. Each pixel is compared with its neighbors in the direction of the gradient. If the pixel is not the maximum, it is set to 0.
 * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Gradient_magnitude_thresholding_or_lower_bound_cut-off_suppression for further details.
 * @param img_magn_gradient Image containing the magnitude of the gradient for each pixel.
 * @param img_dir_gradient Image containing the direction of the gradient for each pixel.
 * @param img_output Image containing the non maximum suppressed gradient magnitude.
 * @param width width of the image.
 * @param height height of the image.
 * @param low_th Lower threshold for the double thresholding.
 * @param high_th Higher threshold for the double thresholding.
 * @return __global__
 * TODO: sharedify
 */
__global__ void lowerBoundCutoffSuppression(float *img_magn_gradient, float *img_dir_gradient, float *img_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        // neighbours magnitude in the N/W and S/E directions
        float neigh_1 = 0.0f;
        float neigh_2 = 0.0f;
        // Gradient direction expressed in degrees
        float angle = img_dir_gradient[idx] * 180.0 / M_PI; // Converti in gradi
        if (angle < 0)
            angle += 180.0;
        // If the angle rounded down is 0 or 180 degrees, the edge is vertical thus the neighbours are to the left and right
        if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.))
        // if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
        {
            neigh_1 = img_magn_gradient[idx + 1];
            neigh_2 = img_magn_gradient[idx - 1];
        }
        // If the angle rounded down is 45 degrees, the edge is diagonal thus the neighbours are to the top right and bottom left
        if (angle >= 22.5f && angle < 67.5f)
        {
            neigh_1 = img_magn_gradient[idx + width - 1];
            neigh_2 = img_magn_gradient[idx - width + 1];
        }
        // If the angle rounded down is 90 degrees the edge is horizontal thus the neighbours are above and below
        if (angle >= 67.5f && angle < 112.5f)
        {
            neigh_1 = img_magn_gradient[idx + width];
            neigh_2 = img_magn_gradient[idx - width];
        }
        // If the angle rounded down is 135 degrees the edge is diagonal thus the neighbours are to the top left and bottom right
        if (angle >= 112.5f && angle < 157.5f)
        {
            neigh_1 = img_magn_gradient[idx - width - 1];
            neigh_2 = img_magn_gradient[idx + width + 1];
        }
        // If the pixel is not the maximum, set it to 0
        if (img_magn_gradient[idx] > neigh_1 && img_magn_gradient[idx] > neigh_2)
        {
            img_output[idx] = img_magn_gradient[idx];
        }
        else
        {
            img_output[idx] = 0.0f;
        }
    }
}
/**
 * @brief Compute the double thresholding suppression. If the pixel is greater than the high threshold, it is marked as a strong edge.
 * If the pixel is less than the low threshold, it is marked as a non-edge.
 * If the pixel is between the two thresholds, it is marked as a weak edge.
 *
 * @param LBCOS_image Low-bound cut-off suppressed image
 * @param two_th_supr_img  Double threshold suppressed output image
 * @param width width of the image
 * @param height height of the image
 * @param lowThreshold low threshold
 * @param highThreshold high threshold
 */
__global__ void doubleThresholdSuppression(float *LBCOS_img, float *two_th_supr_img, int width, int height, float low_threshold, float high_threshold)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        int idx = y * width + x;
        int val = LBCOS_img[idx];
        if (val >= high_threshold)
        {
            two_th_supr_img[idx] = 255.0f; // Strong edge
        }
        else if (val < low_threshold)
        {
            two_th_supr_img[idx] = 0.0f; // Non-edge
        }
        else // between the two thresholds
        {
            two_th_supr_img[idx] = 128.0f; // Weak edge
        }
    }
}
// /**
//  * @brief Compute the hysteresis thresholding. If the pixel is a weak edge and has a strong edge neighbour, it is marked as a strong edge.
//  * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Edge_tracking_by_hysteresis for further details.
//  * @param TTS_img Two-threshold suppressed image
//  * @param output Canny edge detected output image
//  * @param width width of the image
//  * @param height height of the image
//  * NON SHARED
//  */
// __global__ void hysteresis(float *TTS_img, int width, int height)
// {
//     int x = threadIdx.x + blockIdx.x * blockDim.x;
//     int y = threadIdx.y + blockIdx.y * blockDim.y;

//     if (x < width - 1 && y < height - 1)
//     {
//         int idx = y * width + x;
//         // int idx = x + y * width;
//         int size = width * height;
//         if (TTS_img[idx] >= 255.0f)
//         {
//             TTS_img[idx] = 255.0f; // strong edge is kept
//         }
//         else if (TTS_img[idx] >= 128.0f)
//         {
//             bool is_connected_to_strong = false;
//             // printf("weak edge\n");
//             // blob analysis
//             for (int i = -1; i <= 1 && !is_connected_to_strong; i++)
//             {
//                 for (int j = -1; j <= 1; j++)
//                 {
//                     int neighbour_idx = (y + j) * width + (x + i);
//                     if (neighbour_idx >= 0 && neighbour_idx < size && TTS_img[neighbour_idx] >= 255.0f)
//                     {
//                         is_connected_to_strong = true;
//                         break;
//                     }
//                 }
//             }
//             if (is_connected_to_strong)
//             {
//                 TTS_img[idx] = 255.0f;
//             }
//             else
//             {
//                 TTS_img[idx] = 0.0f;
//             }
//         }
//         else
//         {
//             TTS_img[idx] = 0.0f;
//         }
//     }
// }
/**
 * @brief Compute the hysteresis thresholding. If the pixel is a weak edge and has a strong edge neighbour, it is marked as a strong edge.
 * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Edge_tracking_by_hysteresis for further details.
 * @param TTS_img Two-threshold suppressed image
 * @param output Canny edge detected output image
 * @param width width of the image
 * @param height height of the image
 * SHARED
 */
__global__ void hysteresis(float *TTS_img, int width, int height)
{
    __shared__ float data[TILE_WIDTH + FILTER_WIDTH / 2 * 2][TILE_WIDTH + FILTER_WIDTH / 2 * 2];

    const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
    const int gLoc = x0 + y0 * width;

    // Load corners into shared memory
    if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
    {

        // Top-left corner
        int x = x0 - 1;
        int y = y0 - 1;
        data[threadIdx.y][threadIdx.x] = (x >= 0 && y >= 0) ? TTS_img[x + y * width] : 0.0f;

        // Top-right corner
        x = x0 + 1;
        y = y0 - 1;
        data[threadIdx.y][threadIdx.x + 2] = (x < width && y >= 0) ? TTS_img[x + y * width] : 0.0f;

        // Bottom-left corner
        x = x0 - 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x] = (x >= 0 && y < height) ? TTS_img[x + y * width] : 0.0f;

        // Bottom-right corner
        x = x0 + 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x + 2] = (x < width && y < height) ? TTS_img[x + y * width] : 0.0f;
    }

    __syncthreads();
    float tts_val = 0.0f;
    if (x0 < width - 1 && y0 < height - 1)
    {
        if (data[threadIdx.y + 1][threadIdx.x + 1] >= 255.0f)
        {

            tts_val = 255.0f;
        }
        else if (data[threadIdx.y + 1][threadIdx.x + 1] >= 128.0f)
        {
            bool is_connected_to_strong = false;

            // Perform convolution
            if (x0 < width && y0 < height)
            {
                bool is_connected_to_strong = false;
                for (int i = -1; i <= 1 && !is_connected_to_strong; ++i)
                {
                    for (int j = -1; j <= 1; ++j)
                    {
                        int neighbour_idx = (threadIdx.y + 1 + i) * width + (threadIdx.x + 1 + j);
                        if (neighbour_idx >= 0 && neighbour_idx < width * height && data[threadIdx.y + 1 + i][threadIdx.x + 1 + j] >= 255.0f)
                        {
                            is_connected_to_strong = true;
                            break;
                        }
                    }
                }
            }
            if (is_connected_to_strong)
            {
                tts_val = 255.0f;
            }
            else
            {
                tts_val = 0.0f;
            }
        }
        else
        {
            tts_val = 0.0f;
        }
        TTS_img[gLoc] = tts_val;
    }
}

// __global__ void hysteresis(float *TTS_img, int width, int height)
// {
//     __shared__ float data[TILE_WIDTH + FILTER_WIDTH/2 *2][TILE_WIDTH + FILTER_WIDTH/2 *2];

//     const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
//     const int y0 = threadIdx.y + blockIdx.y * blockDim.y;
//     const int gLoc = x0 + y0 * width;

//     // Load corners into shared memory
//     if (threadIdx.x < TILE_WIDTH && threadIdx.y < TILE_WIDTH)
//     {
//         //load itself
//         data[threadIdx.y+1][threadIdx.x+1] = TTS_img[gLoc];

//         if(threadIdx.x ==0){
//             //load left halo
//             data[threadIdx.y][0] = (x0-1 >= 0) ? TTS_img[gLoc-1] : 0.0f;
//         }
//         if(threadIdx.x == TILE_WIDTH-1){
//             //load right halo
//             data[threadIdx.y][TILE_WIDTH+1] = (x0+1 < width) ? TTS_img[gLoc+1] : 0.0f;
//         }
//         if(threadIdx.y == 0){
//             //load top halo
//             data[0][threadIdx.x] = (y0-1 >= 0) ? TTS_img[gLoc-width] : 0.0f;
//         }
//         if(threadIdx.y == TILE_WIDTH-1){
//             //load bottom halo
//             data[TILE_WIDTH+1][threadIdx.x] = (y0+1 < height) ? TTS_img[gLoc+width] : 0.0f;
//         }
//         // Top-left corner
//         if(threadIdx.x == 0 && threadIdx.y == 0){
//             data[0][0] = (x0-1 >= 0 && y0-1 >= 0) ? TTS_img[gLoc-width-1] : 0.0f;
//         }
//         // Top-right corner
//         if(threadIdx.x == TILE_WIDTH-1 && threadIdx.y == 0){
//             data[0][TILE_WIDTH+1] = (x0+1 < width && y0-1 >= 0) ? TTS_img[gLoc-width+1] : 0.0f;
//         }
//         // Bottom-left corner
//         if(threadIdx.x == 0 && threadIdx.y == TILE_WIDTH-1){
//             data[TILE_WIDTH+1][0] = (x0-1 >= 0 && y0+1 < height) ? TTS_img[gLoc+width-1] : 0.0f;
//         }
//         // Bottom-right corner
//         if(threadIdx.x == TILE_WIDTH-1 && threadIdx.y == TILE_WIDTH-1){
//             data[TILE_WIDTH+1][TILE_WIDTH+1] = (x0+1 < width && y0+1 < height) ? TTS_img[gLoc+width+1] : 0.0f;
//         }

//     }

//     __syncthreads();
//     float tts_val = 0.0f;
//     if (x0 < width - 1 && y0 < height - 1)
//     {
//         if (data[threadIdx.y + 1][threadIdx.x + 1] >= 255.0f)
//         {

//             tts_val=255.0f;
//         }
//         else if (data[threadIdx.y + 1][threadIdx.x + 1] >= 128.0f)
//         {
//             bool is_connected_to_strong = false;

//             // Perform convolution
//             if (x0 < width && y0 < height)
//             {
//                 bool is_connected_to_strong = false;
//                 for (int i = -1; i <= 1 && !is_connected_to_strong; ++i)
//                 {
//                     for (int j = -1; j <= 1; ++j)
//                     {
//                         int neighbour_idx = (threadIdx.y + 1 + i) * width + (threadIdx.x + 1 + j);
//                         if (neighbour_idx >= 0 && neighbour_idx < width * height && data[threadIdx.y + 1 + i][threadIdx.x + 1 + j] >= 255.0f)
//                         {
//                             is_connected_to_strong = true;
//                             break;
//                         }
//                     }
//                 }
//             }
//             if (is_connected_to_strong)
//             {
//                 tts_val = 255.0f;
//             }
//             else
//             {
//                 tts_val = 0.0f;
//             }
//         }
//         else
//         {
//            tts_val = 0.0f;
//         }
//         TTS_img[gLoc] = tts_val;
// }
//     }

void printDebugArr(float *arr_d, int n, char *stringa)
{
    float *arr_h = (float *)malloc(n * sizeof(float));
    cudaMemcpy(arr_h, arr_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
    {
        printf("%s[%d] = %f\n", stringa, i, arr_h[i]);
    }
    free(arr_h);
}

void rgbToGrayKernelWrap(unsigned char *img_d, float *gray_d, int N, int M)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    rgbToGrayKernel<<<grid, block>>>(img_d, gray_d, N, M);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for RGB to Gray: %f ms\n", milliseconds);

    // printDebugArr(gray_d, 10, (char *)"gray_d");
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel RGB: %s\n", cudaGetErrorString(err));
    }
}

void gaussianBlurKernelWrap(float *img_d, float *img_out_d, int N, int M, float *kernel, int kernel_size)
{
    const dim3 blockSize(16, 16, 1);
    const dim3 gridSize(N / blockSize.x + 1, M / blockSize.y + 1, 1);
    applyConvolution<<<gridSize, blockSize>>>(img_d, img_out_d, N, M, kernel, kernel_size);
    // printDebugArr(img_out_d, 10, (char *)"blur");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel GAUS: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void convolutionGPUWrap(float *d_Result, float *d_Data, int data_w, int data_h, float *d_kernel)
{
    const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((float)data_w / TILE_WIDTH), ceil((float)data_h / TILE_WIDTH));
    cudaEvent_t start, stop;

    // const dim3 gridDim((data_w + 16 - 1) / 16, (data_h + 16 - 1) / 16);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    convolutionGPU<<<dimGrid, blockSize>>>(d_Result, d_Data, data_w, data_h, d_kernel);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for convolution: %f ms\n", milliseconds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel CONVGPU: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

void cannyMainKernelWrap(float *sobel_x, float *sobel_y, float *output, int width, int height, float low_th, float high_th, float *gauss_kernel, int g_kernel_size)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    float *img_sobel, *sobel_directions, *lbcs_img, *tts_img, *img_debug;
    img_debug = (float *)malloc(width * height * sizeof(float));

    // cudamallocs
    cudaMalloc(&img_sobel, width * height * sizeof(float));
    cudaMalloc(&sobel_directions, width * height * sizeof(float));
    cudaMalloc(&lbcs_img, width * height * sizeof(float));
    cudaMalloc(&tts_img, width * height * sizeof(float));

    combineGradientsKernel<<<grid, block>>>(sobel_x, sobel_y, img_sobel, sobel_directions, width, height);

    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    lowerBoundCutoffSuppression_sh<<<grid, block>>>(img_sobel, sobel_directions, lbcs_img, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for LBCS: %f ms\n", milliseconds);

    doubleThresholdSuppression<<<grid, block>>>(lbcs_img, output, width, height, low_th, high_th);

    // dim3 block2 = dim3(TILE_WIDTH, TILE_WIDTH);
    // dim3 grid2((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // time the hysteresis kernel
    cudaEvent_t start2, stop2;
    float milliseconds2 = 0;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);
    hysteresis<<<grid, block>>>(output, width, height);
    cudaDeviceSynchronize();
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);
    cudaEventElapsedTime(&milliseconds2, start2, stop2);
    // printf("Elapsed time for Hysteresis: %f ms\n", milliseconds2);

    cudaFree(img_sobel);
    cudaFree(sobel_directions);
    cudaFree(lbcs_img);
    cudaFree(tts_img);
    free(img_debug);
}
void harrisMainKernelWrap(float *sobel_x, float *sobel_y, float *output, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size)
{
    int n = width * height;
    float *Ix2_d, *Iy2_d, *IxIy_d, *IxIy_d2, *detM_d, *traceM_d;
    // cudaStream_t stream1, stream2, stream3;
    cudaStream_t streams[3];
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y,
                        1);

    for (int i = 0; i < 3; i++)
    {
        checkCuda(cudaStreamCreate(&streams[i]));
    }
    // cuda malloc
    cudaMalloc(&Ix2_d, n * sizeof(float));
    cudaMalloc(&Iy2_d, n * sizeof(float));
    cudaMalloc(&IxIy_d, n * sizeof(float));
    cudaMalloc(&IxIy_d2, n * sizeof(float));
    cudaMalloc(&detM_d, n * sizeof(float));
    cudaMalloc(&traceM_d, n * sizeof(float));

    // Create CUDA streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);

    // 1. Compute Ix^2, Iy^2, and Ix*Iy in parallel using streams
    vecMul<<<gridSize, blockSize, 0, streams[0]>>>(sobel_x, sobel_x, Ix2_d, width, height);  // Ix^2
    vecMul<<<gridSize, blockSize, 0, streams[1]>>>(sobel_y, sobel_y, Iy2_d, width, height);  // Iy^2
    vecMul<<<gridSize, blockSize, 0, streams[2]>>>(sobel_x, sobel_y, IxIy_d, width, height); // Ix * Iy

    // Synchronize streams
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
    cudaDeviceSynchronize();

    // 2. Apply Gaussian blur to Ix2, Iy2, and IxIy
    // applyConvolution<<<gridSize, blockSize, 0, streams[0]>>>(Ix2_d, Ix2_d, width, height, gaussian_kernel, g_kernel_size);
    // applyConvolution<<<gridSize, blockSize, 0, streams[1]>>>(Iy2_d, Iy2_d, width, height, gaussian_kernel, g_kernel_size);
    // applyConvolution<<<gridSize, blockSize, 0, streams[0]>>>(IxIy_d, IxIy_d, width, height, gaussian_kernel, g_kernel_size);

    cudaEventRecord(start);
    // applyConvolution<<<gridSize, blockSize>>>(Ix2_d, Ix2_d, width, height, gaussian_kernel, g_kernel_size);
    convolutionGPUWrap(Ix2_d, Ix2_d, width, height, gaussian_kernel);
    // applyConvolution<<<gridSize, blockSize>>>(Iy2_d, Iy2_d, width, height, gaussian_kernel, g_kernel_size);
    convolutionGPUWrap(Iy2_d, Iy2_d, width, height, gaussian_kernel);
    // applyConvolution<<<gridSize, blockSize>>>(IxIy_d, IxIy_d, width, height, gaussian_kernel, g_kernel_size);
    convolutionGPUWrap(IxIy_d, IxIy_d, width, height, gaussian_kernel);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for Gaussian blur: %f ms\n", milliseconds);

    // Synchronize streams
    cudaDeviceSynchronize();

    // 3. Compute det(M) = Ix2 * Iy2 - (IxIy)^2
    vecMul<<<gridSize, blockSize, 0, streams[0]>>>(IxIy_d, IxIy_d, IxIy_d2, width, height); // (IxIy)^2
    vecMul<<<gridSize, blockSize, 0, streams[1]>>>(Ix2_d, Iy2_d, detM_d, width, height);    // Ix2 * Iy2
    vecSub<<<gridSize, blockSize>>>(detM_d, IxIy_d2, detM_d, width, height);                // det(M)

    // 4. Compute trace(M) = Ix2 + Iy2
    vecAdd<<<gridSize, blockSize>>>(Ix2_d, Iy2_d, traceM_d, width, height); // trace(M)

    // 5. Compute response R = det(M) / trace(M)
    vecDiv<<<gridSize, blockSize>>>(detM_d, traceM_d, output, width, height);

    // 5b. Compute the Shi-Tomasi response
    // computeShiTommasiResponse<<<gridSize, blockSize>>>(Ix2_d, Iy2_d, IxIy_d, output, width, height);

    // 6. Non-maximum suppression on the Harris response
    cudaEventRecord(start);
    nonMaximumSuppression<<<gridSize, blockSize>>>(output, output, width, height, g_kernel_size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Elapsed time for NMS: %f ms\n", milliseconds);

    // Cleanup
    cudaFree(Ix2_d);
    cudaFree(Iy2_d);
    cudaFree(IxIy_d);
    cudaFree(IxIy_d2);
    cudaFree(detM_d);
    cudaFree(traceM_d);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);

    // Error check (optional, for debugging purposes)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

#define DEBUG

#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <assert.h>
#include "../include/cuda_kernel.cuh"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
#define TILE_WIDTH 16 // 16 X 16 TILE

inline cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
    // printf("CUda result: %d\n", result);
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }
#endif
    return result;
}

/*************************
 *
 * GENERIC KERNELS
 *
 *************************/

/**
 * @brief Vector addition kernel. It computes the element-wise addition of two vectors.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 * @param M Number of rows
 * @param N Number of columns
 */
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

/**
 * @brief Vector multiplication kernel. It computes the element-wise multiplication of two vectors.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 * @param M Number of rows
 * @param N Number of columns
 */
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
/**
 * @brief Vector subtraction kernel. It computes the element-wise subtraction of two vectors.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 * @param M Number of rows
 * @param N Number of columns
 */
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
/**
 * @brief Vector division kernel. It computes the element-wise division of two vectors.
 * If the denominator is 0, the result is set to 0.
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C
 * @param M Number of rows
 * @param N Number of columns
 */
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

/**
 * @brief Utility kernel to copy a 1 channel image to a 3 channel image.
 *
 * @param src_img_d The source 1 channel image
 * @param dst_img_d The destination 3 channel image
 * @param width Width of the image
 * @param height Width of the image
 */

__global__ void copy1ChannelTo4(float *src_img_d, uchar4 *dst_img_d, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        int idx = j * width + i;

        unsigned char val = (unsigned char)src_img_d[idx];

        dst_img_d[idx] = make_uchar4(val, val, val, 255);
    }
}
/**
 * @brief Given an Harris Map and a threshold, it colors the corners in red of the destination image.
 *
 * @param harris_map Harris response map
 * @param dst_img_d Destination image
 * @param width Width of the image
 * @param height Height of the image
 * @param threshold Threshold value on which to consider a pixel as a corner
 */
__global__ void cornerColoring(float *harris_map, uchar4 *dst_img_d, int width, int height, float threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        int idx = j * width + i; // Corrected indexing order (row-major)

        if (harris_map[idx] > threshold && i > 2 && j > 2 && i < width - 2 && j < height - 2)
        {
            // Coloring a 3x3 window around the detected corner
            for (int k = -1; k <= 1; k++)
            {
                for (int z = -1; z <= 1; z++)
                {
                    int new_i = i + k;
                    int new_j = j + z;

                    if (new_i >= 0 && new_j >= 0 && new_i < width && new_j < height)
                    {
                        int idx_neigh = new_j * width + new_i;
                        dst_img_d[idx_neigh] = make_uchar4(255, 0, 0, 255); // Set to red (RGBA)
                    }
                }
            }
        }
    }
}

/***********************
 *
 * Convolution Kernels
 *
 **********************/

// __constant__ float d_Kernel[9];
/**
 * @brief Fast Convolution kernel with shared memory. It computes the convolution of an image with a given kernel
 *
 * @param result_d Output image
 * @param data_d Input image
 * @param width Width of the image
 * @param height Height of the image
 * @param kernel_d Convolution kernel
 */
__global__ void convolutionGPU(float *result_d, float *data_d, int width, int height, const float *kernel_d, int filter_size, int shared_size)
{
    extern __shared__ float data_flat[];
    // shared_size = (TILE_WIDTH + 2 * (kernel_size / 2));
    const int filter_radius = filter_size / 2;

    // Calculate base coordinates for shared memory block
    int xBase = blockIdx.x * TILE_WIDTH - filter_radius;
    int yBase = blockIdx.y * TILE_WIDTH - filter_radius;

    // Collaborative loading of shared memory
    for (int i = threadIdx.y; i < shared_size; i += blockDim.y)
    {
        for (int j = threadIdx.x; j < shared_size; j += blockDim.x)
        {
            int x = xBase + j;
            int y = yBase + i;
            data_flat[i * shared_size + j] =
                (x >= 0 && y >= 0 && x < width && y < height) ? data_d[y * width + x] : 0.0f;
        }
    }
    __syncthreads();

    // Calculate global coordinates
    const int x0 = threadIdx.x + blockIdx.x * blockDim.x;
    const int y0 = threadIdx.y + blockIdx.y * blockDim.y;

    if (x0 < width && y0 < height)
    {
        float sum = 0.0f;
#pragma unroll 3
        for (int i = -filter_radius; i <= filter_radius; ++i)
        {
#pragma unroll 3
            for (int j = -filter_radius; j <= filter_radius; ++j)
            {
                int shX = threadIdx.x + filter_radius + j;
                int shY = threadIdx.y + filter_radius + i;
                sum += data_flat[shY * shared_size + shX] *
                       kernel_d[(i + filter_radius) * filter_size + (j + filter_radius)];
            }
        }
        result_d[y0 * width + x0] = sum;
    }
}
// How many results a thread computes per convolution
#define ROW_THREAD_STEPS 8
// Border around the shared memory
#define ROW_HALO_STEPS 1
__global__ void rowConvolution(float *data_d, float *result_d, int width, int height, float *kernel_d)
{
    // Added +1 to shared memory width to prevent out-of-bounds access
    __shared__ float data[TILE_WIDTH][(ROW_THREAD_STEPS + 2 * ROW_HALO_STEPS) * TILE_WIDTH + 1];

    const int x0 = (blockIdx.x * ROW_THREAD_STEPS - ROW_HALO_STEPS) * TILE_WIDTH + threadIdx.x;
    const int y0 = blockIdx.y * blockDim.y + threadIdx.y;

    result_d += y0 * width + x0;
    data_d += y0 * width + x0;

    // Loading main data with boundary checks
#pragma unroll
    for (int i = ROW_HALO_STEPS; i < ROW_HALO_STEPS + ROW_THREAD_STEPS; i++)
    {
        int x = x0 + i * TILE_WIDTH;
        int y = y0;
        bool in_bounds = (x >= 0) && (x < width) && (y >= 0) && (y < height);
        data[threadIdx.y][threadIdx.x + i * TILE_WIDTH] =
            in_bounds ? data_d[i * TILE_WIDTH] : 0.0f;
    }

    // Loading left halo with full boundary checks
#pragma unroll
    for (int i = 0; i < ROW_HALO_STEPS; i++)
    {
        int x = x0 + i * TILE_WIDTH;
        int y = y0;
        bool in_bounds = (x >= 0) && (x < width) && (y >= 0) && (y < height);
        data[threadIdx.y][threadIdx.x + i * TILE_WIDTH] =
            in_bounds ? data_d[i * TILE_WIDTH] : 0.0f;
    }

    // Loading right halo with full boundary checks
#pragma unroll
    for (int i = ROW_HALO_STEPS + ROW_THREAD_STEPS;
         i < ROW_HALO_STEPS + ROW_THREAD_STEPS + ROW_HALO_STEPS;
         i++)
    {
        int x = x0 + i * TILE_WIDTH;
        int y = y0;
        bool in_bounds = (x >= 0) && (x < width) && (y >= 0) && (y < height);
        data[threadIdx.y][threadIdx.x + i * TILE_WIDTH] =
            in_bounds ? data_d[i * TILE_WIDTH] : 0.0f;
    }

    __syncthreads();

    // Compute convolution with write boundary checks
#pragma unroll
    for (int i = ROW_HALO_STEPS; i < ROW_HALO_STEPS + ROW_THREAD_STEPS; i++)
    {
        float sum = 0.0f;
#pragma unroll
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++)
        {
            sum += data[threadIdx.y][threadIdx.x + i * TILE_WIDTH + j] * kernel_d[FILTER_RADIUS + j];
        }
        int x = x0 + i * TILE_WIDTH;
        int y = y0;
        if (x >= 0 && x < width && y >= 0 && y < height)
        {
            result_d[i * TILE_WIDTH] = sum;
        }
    }
}
// How many results a thread computes per convolution
#define COL_THREAD_STEPS 8
// Border around the shared memory
#define COL_HALO_STEPS 1
__global__ void columnConvolution(float *data_d, float *result_d, int width, int height, float *kernel_d)
{
    __shared__ float data[TILE_WIDTH][(COL_THREAD_STEPS + 2 * COL_HALO_STEPS) * TILE_WIDTH + 1];

    const int x0 = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int y0 = (blockIdx.y * COL_THREAD_STEPS - COL_HALO_STEPS) * TILE_WIDTH + threadIdx.y;

    result_d += y0 * width + x0;
    data_d += y0 * width + x0;

    // Loading main data
#pragma unroll
    for (int i = COL_HALO_STEPS; i < COL_HALO_STEPS + COL_THREAD_STEPS; i++)
    {
        int y_access = y0 + i * TILE_WIDTH;
        int x_access = x0;
        data[threadIdx.x][threadIdx.y + i * TILE_WIDTH] =
            (y_access >= 0 && y_access < height && x_access < width)
                ? data_d[i * TILE_WIDTH * width]
                : 0.0f;
    }

    // Loading top halo
#pragma unroll
    for (int i = 0; i < COL_HALO_STEPS; i++)
    {
        int y_access = y0 + i * TILE_WIDTH;
        int x_access = x0;
        data[threadIdx.x][threadIdx.y + i * TILE_WIDTH] =
            (y_access >= 0 && y_access < height && x_access < width)
                ? data_d[i * TILE_WIDTH * width]
                : 0.0f;
    }

    // Loading bottom halo
#pragma unroll
    for (int i = COL_HALO_STEPS + COL_THREAD_STEPS; i < COL_HALO_STEPS + COL_THREAD_STEPS + COL_HALO_STEPS; i++)
    {
        int y_access = y0 + i * TILE_WIDTH;
        int x_access = x0;
        data[threadIdx.x][threadIdx.y + i * TILE_WIDTH] =
            (y_access >= 0 && y_access < height && x_access < width)
                ? data_d[i * TILE_WIDTH * width]
                : 0.0f;
    }

    __syncthreads();

    // Computing convolution
#pragma unroll
    for (int i = COL_HALO_STEPS; i < COL_HALO_STEPS + COL_THREAD_STEPS; i++)
    {
        float sum = 0.0f;
#pragma unroll
        for (int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++)
        {
            sum += data[threadIdx.x][threadIdx.y + i * TILE_WIDTH + j] * kernel_d[FILTER_RADIUS + j];
        }
        int y = y0 + i * TILE_WIDTH;
        if (y >= 0 && y < height && x0 < width)
        {
            result_d[i * TILE_WIDTH * width] = sum;
        }
    }
}
// Simple not-optimized convolution kernel (no longer used)
__global__ void applyConvolution(float *img_d, float *img_out_d, int width, int height, float *kernel, int kernel_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < height && j < width)
    {
        int idx = i * width + j;
        int k = kernel_size / 2;
        float sum = 0.0f;
        for (int u = -k; u <= k; u++)
        {
            for (int v = -k; v <= k; v++)
            {
                int x = i + u;
                int y = j + v;
                if (x >= 0 && x < height && y >= 0 && y < width)
                {
                    sum += img_d[x * width + y] * kernel[(u + k) * kernel_size + (v + k)];
                }
            }
        }
        img_out_d[idx] = sum;
    }
}

/**************
 *
 * Main kernels
 *
 **************/

/**
 * @brief Kernel that convers a RGBA image to a Grayscale image
 *
 * @param img_d Source RGBA image
 * @param gray_d  Destination Grayscale image
 * @param width Width of the image
 * @param height Height of the image
 */
__global__ void rgbToGrayKernel(const uchar4 *img_d, float *gray_d, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        uchar4 pixel = img_d[idx];
        gray_d[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    }
}

/**
 * @brief Converts a RGB image to RGBA, padding the alpha channel with 255
 * This is so that we can exploit memory transactions and improve coalescing.
 * @param rgb_data Input RGB image
 * @param padded_data Output RGBA image
 * @param width Width of the image
 * @param height Height of the image
 */
void padRGBToUchar4(const unsigned char *rgb_data, uchar4 *padded_data, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = y * width + x;
            int rgb_idx = 3 * idx;
            padded_data[idx] = make_uchar4(
                rgb_data[rgb_idx],     // R
                rgb_data[rgb_idx + 1], // G
                rgb_data[rgb_idx + 2], // B
                0                      // Padding (alpha)
            );
        }
    }
}

/**
 * @brief Warp-level max reduction exploiting shfl intrinsic. At the end, the thread in the lane 0 will have the maximum value of the warp.
 * @param val value to be reduced that is passed between threads withing same warp
 */
__device__ float warpReduceMax(float val)
{
    // max-reduction using shfl_down
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

#define THREADS_PER_BLOCK 1024

/**
 * @brief Block-level reduction to find the maximum value
 * At the end, the maximum value of the block is stored in the first thread of the block
 * @param val value to be max-reduced
 */
__device__ float blockReduceMax(float val)
{
    //
    __shared__ float shared[THREADS_PER_BLOCK / 32]; // Shared memory for warp-level results
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    int lane = tid % warpSize;   // Lane(aka thread) index in the warp
    int warpId = tid / warpSize; // Warp index within the block

    val = warpReduceMax(val);

    // writing max value of the warp to shared memory
    if (lane == 0)
    {
        shared[warpId] = val;
    }

    __syncthreads(); // making sure every warps computed their max value and wrote it to shared memory

    // Only the first warp performs the final reduction
    if (warpId == 0)
    {
        val = (tid < blockDim.x * blockDim.y / 32) ? shared[lane] : -FLT_MAX;
        val = warpReduceMax(val);
    }

    return val;
}
/**
 * @brief custom atomicMax implementation since there isn't an overloaded one with float values
 *
 * @see https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda
 * @param address Address of the value to be updated
 * @param val Value to be compared with the current value
 */
__device__ static float atomicMax(float *address, float val)
{
    int *address_as_i = (int *)address;
    int old = *address_as_i, assumed;
    do
    {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
                          __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * @brief Fast max reduction exploiting cuda shfl intrinsics.
 * It computes the maximum value of an image using a block-level reduction.
 * Then it uses an atomic operation to find the final maximum value among all blocks.
 *
 * @param img Input image
 * @param max_value Final max value
 * @param width Width of the image
 * @param height Height of the image
 */
__global__ void findMaxReductionSHFL(float *img, float *max_value, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute linear thread index for stride
    int stride_x = gridDim.x * blockDim.x;
    int stride_y = gridDim.y * blockDim.y;

    float local_max = -FLT_MAX;

    // Process all elements assigned to this thread with strided access
    for (int j = y; j < height; j += stride_y)
    {
        for (int i = x; i < width; i += stride_x)
        {
            int idx = j * width + i;
            local_max = max(local_max, img[idx]);
        }
    }

    // Block-level reduction to find max value
    float block_max = blockReduceMax(local_max);

    // Max value found by using custom atomic max operation
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        atomicMax(max_value, block_max);
        // max_value[0] = atomicMax(max_value, block_max);
    }
}

/**
 * @brief Max reduction using shared memory
 *
 * @param img Input image
 * @param max_value Final max value
 * @param width Width of the image
 * @param height Height of the image
 */
__global__ void findMaxReductionSHRD(float *img, float *max_value, int width, int height)
{
    extern __shared__ float shared_max[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int globalIdx = y * width + x;

    shared_max[tid] = img[globalIdx];
    __syncthreads();
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Max value found by using custom atomic max operation
    if (tid == 0)
    {
        atomicMax(max_value, shared_max[0]);
        // max_value[0] = atomicMax(max_value, block_max);
    }
}

/**
 * @brief Non maximum suppression kernel. It compares the current pixel with its neighbours in a 3x3 window.
 * If the current pixel is not the maximum, it is set to 0.
 *
 * @param harris_map Harris response map
 * @param corners_output Output image
 * @param width Width of the image
 * @param height Height of the image
 */
__global__ void nonMaximumSuppression(float *harris_map, float *corners_output, int width, int height)
{
    // Shared memory for the tile and its border
    // 4 because the kernel size is 5, thus 5/2 = 2 and 2*2 = 4. In this way we can load the corners of the tile
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
        data[threadIdx.y][threadIdx.x] = (x >= 0 && y >= 0) ? harris_map[x + y * width] : 0.0f;

        // Top-right corner
        x = x0 + 1;
        y = y0 - 1;
        data[threadIdx.y][threadIdx.x + 2] = (x < width && y >= 0) ? harris_map[x + y * width] : 0.0f;

        // Bottom-left corner
        x = x0 - 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x] = (x >= 0 && y < height) ? harris_map[x + y * width] : 0.0f;

        // Bottom-right corner
        x = x0 + 1;
        y = y0 + 1;
        data[threadIdx.y + 2][threadIdx.x + 2] = (x < width && y < height) ? harris_map[x + y * width] : 0.0f;
    }

    __syncthreads();

    // Perform convolution
    if (x0 < width && y0 < height)
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

// non optimized NMS(non shared memory version)
__global__ void nonMaximumSuppression_(float *harris_map, float *corners_output, int width, int height, int window_size)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < height && j < width)
    {
        int idx = i * width + j;
        int k = window_size / 2;
        float max_val = 0.0f;
        for (int u = -k; u <= k; u++)
        {
            for (int v = -k; v <= k; v++)
            {
                int x = i + u;
                int y = j + v;
                if (x >= 0 && x < height && y >= 0 && y < width)
                {
                    max_val = fmaxf(max_val, harris_map[x * width + y]);
                }
            }
        }
        corners_output[idx] = (harris_map[idx] == max_val) ? harris_map[idx] : 0.0f;
    }
}

/**
 * @brief Kernel that computes Shi-Tommasi response for each pixel in the image: response=min(lambda1, lambda2) where lambda1 and lambda2 are the eigenvalues of the structure tensor.
 *
 * @param Ixx Squared gradient in x direction
 * @param Iyy Squared gradient in y direction
 * @param Ixy Product of the gradients(x and y direction)
 * @param corners_output Output image containing the Shi-Tommasi response for each pixel
 * @param width Width of the image
 * @param height Height of the image
 */
__global__ void computeShiTommasiResponse(const float *Ixx, const float *Iyy, const float *Ixy, float *corners_output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
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
__global__ void combineGradientsKernel(float *img_sobel_x, float *img_sobel_y, float *img_sobel, float *sobel_directions, int width, int height)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < width && j < height)
    {
        int idx = j * width + i;
        float gx = img_sobel_x[idx];
        float gy = img_sobel_y[idx];

        img_sobel[idx] = hypotf(gx, gy);
        // img_sobel[idx] = sqrt((float)(gx * gx + gy * gy));
        sobel_directions[idx] = atan2f(gy, gx);
    }
}

/**
 * @brief Compute a non maximum suppression on the gradient magnitude image. Each pixel is compared with its neighbors in the direction of the gradient. If the pixel is not the maximum, it is set to 0.
 * Shared memory version
 * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Gradient_magnitude_thresholding_or_lower_bound_cut-off_suppression for further details.
 * @param img_magn_gradient Image containing the magnitude of the gradient for each pixel.
 * @param img_dir_gradient Image containing the direction of the gradient for each pixel.
 * @param img_output Image containing the non maximum suppressed gradient magnitude.
 * @param width width of the image.
 * @param height height of the image.
 * @param low_th Lower threshold for the double thresholding.
 * @param high_th Higher threshold for the double thresholding.
 */
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
    __syncthreads();
    if (x0 < width && y0 < height)
    {
        int idx = y0 * width + x0;

        /**
         * NMS without Interpolation - Faster but less accurate(jagged edges)
         *
         */

        // // neighbours magnitude in the N/W and S/E directions
        // float neigh_1 = 0.0f;
        // float neigh_2 = 0.0f;
        // // Gradient direction expressed in degrees
        // float angle = img_dir_gradient[idx] * 180.0 / M_PI; // Converti in gradi
        // if (angle < 0)
        //     angle += 180.0;
        // // If the angle rounded down is 0 or 180 degrees, the edge is vertical thus the neighbours are to the left and right
        // if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.))
        // // if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f && angle <= 180.0f))
        // {
        //     // neigh_1 = img_magn_gradient[idx + 1];
        //     neigh_1 = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 2];
        //     // neigh_2 = img_magn_gradient[idx - 1];
        //     neigh_2 = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x];
        // }
        // // If the angle rounded down is 45 degrees, the edge is diagonal thus the neighbours are to the top right and bottom left
        // if (angle >= 22.5f && angle < 67.5f)
        // {
        //     // neigh_1 = img_magn_gradient[idx + width - 1];
        //     neigh_1 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x];
        //     // neigh_2 = img_magn_gradient[idx - width + 1];
        //     neigh_2 = img_magn_gradient_shared[threadIdx.y][threadIdx.x + 2];
        // }
        // // If the angle rounded down is 90 degrees the edge is horizontal thus the neighbours are above and below
        // if (angle >= 67.5f && angle < 112.5f)
        // {
        //     // neigh_1 = img_magn_gradient[idx + width];
        //     neigh_1 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x + 1];
        //     // neigh_2 = img_magn_gradient[idx - width];
        //     neigh_2 = img_magn_gradient_shared[threadIdx.y][threadIdx.x + 1];
        // }
        // // If the angle rounded down is 135 degrees the edge is diagonal thus the neighbours are to the top left and bottom right
        // if (angle >= 112.5f && angle < 157.5f)
        // {
        //     // neigh_1 = img_magn_gradient[idx - width - 1];
        //     neigh_1 = img_magn_gradient_shared[threadIdx.y][threadIdx.x];
        //     // neigh_2 = img_magn_gradient[idx + width + 1];
        //     neigh_2 = img_magn_gradient_shared[threadIdx.y + 2][threadIdx.x + 2];
        // }
        // // If the pixel is not the maximum, set it to 0
        // // if (img_magn_gradient[idx] > neigh_1 && img_magn_gradient[idx] > neigh_2)
        // float tmp = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 1];
        // if (tmp > neigh_1 && tmp > neigh_2)
        // {
        //     // img_output[idx] = img_magn_gradient[idx];
        //     img_output[idx] = tmp;
        // }
        // else
        // {
        //     img_output[idx] = 0.0f;
        // }

        /**
         * NMS with Bilinear Interpolation - Slower but smoother edges
         *
         */
        float curMag = img_magn_gradient_shared[threadIdx.y + 1][threadIdx.x + 1];
        float angle = img_dir_gradient[idx];
        if (angle < 0)
            angle += M_PI;

        float dx = cosf(angle);
        float dy = sinf(angle);

        float pos_x = x0 + dx;
        float pos_y = y0 + dy;
        float neg_x = x0 - dx;
        float neg_y = y0 - dy;

        // Bilinear interpolation for positive neighbour
        int pos_x0 = floorf(pos_x);
        int pos_y0 = floorf(pos_y);
        int pos_x1 = pos_x0 + 1;
        int pos_y1 = pos_y0 + 1;
        float wx = pos_x - pos_x0;
        float wy = pos_y - pos_y0;
        float pos_neigh = (1.0f - wx) * (1.0f - wy) * img_magn_gradient[pos_y0 * width + pos_x0] +
                          wx * (1.0f - wy) * img_magn_gradient[pos_y0 * width + pos_x1] +
                          (1.0f - wx) * wy * img_magn_gradient[pos_y1 * width + pos_x0] +
                          wx * wy * img_magn_gradient[pos_y1 * width + pos_x1];

        // Bilinear interpolation for negative neighbour
        int neg_x0 = floorf(neg_x);
        int neg_y0 = floorf(neg_y);
        int neg_x1 = neg_x0 + 1;
        int neg_y1 = neg_y0 + 1;
        float wx_neg = neg_x - neg_x0;
        float wy_neg = neg_y - neg_y0;
        float neg_neigh = (1.0f - wx_neg) * (1.0f - wy_neg) * img_magn_gradient[neg_y0 * width + neg_x0] +
                          wx_neg * (1.0f - wy_neg) * img_magn_gradient[neg_y0 * width + neg_x1] +
                          (1.0f - wx_neg) * wy_neg * img_magn_gradient[neg_y1 * width + neg_x0] +
                          wx_neg * wy_neg * img_magn_gradient[neg_y1 * width + neg_x1];

        if (curMag >= pos_neigh && curMag >= neg_neigh)
        {
            img_output[idx] = curMag;
        }
        else
        {
            img_output[idx] = 0.0f;
        }
    }
}

/**
 * @brief Compute a non maximum suppression on the gradient magnitude image. Each pixel is compared with its neighbors in the direction of the gradient. If the pixel is not the maximum, it is set to 0.
 * Non shared memory version
 * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Gradient_magnitude_thresholding_or_lower_bound_cut-off_suppression for further details.
 * @param img_magn_gradient Image containing the magnitude of the gradient for each pixel.
 * @param img_dir_gradient Image containing the direction of the gradient for each pixel.
 * @param img_output Image containing the non maximum suppressed gradient magnitude.
 * @param width width of the image.
 * @param height height of the image.
 * @param low_th Lower threshold for the double thresholding.
 * @param high_th Higher threshold for the double thresholding.
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
/**
//  * @brief Compute the hysteresis thresholding. If the pixel is a weak edge and has a strong edge neighbour, it is marked as a strong edge.
//  * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Edge_tracking_by_hysteresis for further details.
//  * @param TTS_img Two-threshold suppressed image
//  * @param output Canny edge detected output image
//  * @param width width of the image
//  * @param height height of the image
//  * NON SHARED
//  */
__global__ void hysteresis_(float *TTS_img, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width - 1 && y < height - 1)
    {
        int idx = y * width + x;
        // int idx = x + y * width;
        int size = width * height;
        if (TTS_img[idx] >= 255.0f)
            return;
        if (TTS_img[idx] >= 128.0f)
        {
            bool is_connected_to_strong = false;
            // printf("weak edge\n");
            // blob analysis
            for (int i = -1; i <= 1 && !is_connected_to_strong; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int neighbour_idx = (y + j) * width + (x + i);
                    if (neighbour_idx >= 0 && neighbour_idx < size && TTS_img[neighbour_idx] >= 255.0f)
                    {
                        is_connected_to_strong = true;
                        break;
                    }
                }
            }
            if (is_connected_to_strong)
            {
                TTS_img[idx] = 255.0f;
            }
            else
            {
                TTS_img[idx] = 0.0f;
            }
        }
        else
        {
            TTS_img[idx] = 0.0f;
        }
    }
}

/**
 * @brief Compute the hysteresis thresholding. If the pixel is a weak edge and has a strong edge neighbour, it is marked as a strong edge.
 * @cite https://en.wikipedia.org/wiki/Canny_edge_detector#Edge_tracking_by_hysteresis for further details.
 * @param TTS_img Two-threshold suppressed image
 * @param output Canny edge detected output image
 * @param width width of the image
 * @param height height of the image
 * SHARED
 */
#define WEAK_EDGE 128.0f
#define STRONG_EDGE 255.0f
__global__ void hysteresis(float *TTS_img, int width, int height, bool iterative)
{
    // __shared__ float data[TILE_WIDTH + FILTER_WIDTH / 2 * 2][TILE_WIDTH + FILTER_WIDTH / 2 * 2];
    __shared__ float data[TILE_WIDTH + 2][TILE_WIDTH + 2];

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
        float val = data[threadIdx.y + 1][threadIdx.x + 1];
        if (val >= STRONG_EDGE)
        {
            tts_val = 255.0f;
        }
        else if (val >= WEAK_EDGE)
        {
            bool is_connected_to_strong = false;

            // Perform local blob analysis
            // if (x0 < width && y0 < height)
            // {
            // bool is_connected_to_strong = false;
            for (int i = -1; i <= 1 && !is_connected_to_strong; i++)
            {
                for (int j = -1; j <= 1; j++)
                {
                    int neighbour_idx = (threadIdx.y + 1 + i) * width + (threadIdx.x + 1 + j);
                    if (neighbour_idx >= 0 && neighbour_idx < width * height && data[threadIdx.y + 1 + i][threadIdx.x + 1 + j] >= 255.0f)
                    {
                        is_connected_to_strong = true;
                        break;
                    }
                }
            }
            // }
            if (iterative)
                tts_val = is_connected_to_strong ? STRONG_EDGE : WEAK_EDGE;
            else
                tts_val = is_connected_to_strong ? STRONG_EDGE : 0.0f;
        }
        else
        {
            tts_val = 0.0f;
        }
        TTS_img[gLoc] = tts_val;
    }
}

/**
 * @brief Utility function to print the content of a float (device)array.
 * It copies the content of the device array to the host and prints it.
 *
 * @param arr_d device array
 * @param n number of elements to print
 * @param stringa string to print before the array
 */
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

/**
 * @brief Apply a separable convolution on the input image using the given kernel.
 * It first applies a row convolution and then a column convolution. This is equivalent to applying a 2D convolution.
 *
 * @param img_d Input image
 * @param img_out_d Output image
 * @param width Width of the image
 * @param height Height of the image
 * @param kernel_x Kernel in the x direction
 * @param kernel_y Kernel in the y direction
 * @param kernel_size Size of the kernel
 */
void separableConvolutionKernelWrap(float *img_d, float *img_out_d, int width, int height,
                                    float *kernel_x, float *kernel_y, int kernel_size)
{
    const dim3 blockSize(TILE_WIDTH, TILE_WIDTH);


    const dim3 gridSize_row(
        (width + TILE_WIDTH * ROW_THREAD_STEPS - 1) / (TILE_WIDTH * ROW_THREAD_STEPS),
        (height + TILE_WIDTH - 1) / TILE_WIDTH 
    );

   
    const dim3 gridSize_col(
        (width + TILE_WIDTH - 1) / TILE_WIDTH, 
        (height + TILE_WIDTH * COL_THREAD_STEPS - 1) / (TILE_WIDTH * COL_THREAD_STEPS));
    

    float *img_temp_d;
    cudaMalloc(&img_temp_d, width * height * sizeof(float));

    rowConvolution<<<gridSize_row, blockSize>>>(img_d, img_temp_d, width, height, kernel_x);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel ROW: %s\n", cudaGetErrorString(err));
    }

    columnConvolution<<<gridSize_col, blockSize>>>(img_temp_d, img_out_d, width, height, kernel_y);
    cudaDeviceSynchronize();


    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel COL: %s\n", cudaGetErrorString(err));
    }

    cudaFree(img_temp_d);
}
/**
 * @brief Wrapper for the RGB to Gray kernel.
 *
 * @param img_d Input image
 * @param gray_d Output image
 * @param N Width of the image
 * @param M Height of the image
 */
void rgbToGrayKernelWrap(uchar4 *img_d, float *gray_d, int width, int height)
{

    // Launch kernel
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    // debug
    if (img_d == nullptr || gray_d == nullptr)
    {
        fprintf(stderr, "Error: NULL pointer before kernel launch!\n");
    }

    rgbToGrayKernel<<<grid, block>>>(img_d, gray_d, width, height);
    cudaDeviceSynchronize();

    // Error checking
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel RGB: %s\n", cudaGetErrorName(err));
    }
}

/**
 * @brief Wrapper for the convolution kernel.
 *
 * @param d_Result Output image
 * @param d_Data Input image
 * @param data_w Width of the image
 * @param data_h Height of the image
 * @param d_kernel Kernel
 */
void convolutionGPUWrap(float *result_d, float *data_d, int data_w, int data_h, float *__restrict__ d_kernel, int kernel_size)
{
    const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(ceil((float)data_w / TILE_WIDTH), ceil((float)data_h / TILE_WIDTH));
    // cudaEvent_t start, stop;

    // const dim3 gridDim((data_w + 16 - 1) / 16, (data_h + 16 - 1) / 16);
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaMemcpyToSymbol(d_Kernel, d_kernel, 9 * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // cudaEventRecord(start);
    int sharedWidth = TILE_WIDTH + 2 * (kernel_size / 2);
    // sharedMemSize *= sharedMemSize * sharedMemSize;

    
    convolutionGPU<<<dimGrid, blockSize, (sharedWidth * sharedWidth) * sizeof(float)>>>(result_d, data_d, data_w, data_h, d_kernel, kernel_size, sharedWidth);
    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Elapsed time for convolution: %f ms\n", milliseconds);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in kernel CONVGPU: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
}
/**
 * @brief Driver function for the Canny edge detection algorithm.
 *
 * @param img_data_h Host image data
 * @param img_data_d Device image data
 * @param sobel_x Sobel kernel in the x direction
 * @param sobel_y Sobel kernel in the y direction
 * @param width Width of the image
 * @param height Height of the image
 * @param low_th Lower threshold for the double thresholding
 * @param high_th Higher threshold for the double thresholding
 * @param gauss_kernel Gaussian kernel
 * @param g_kernel_size Size of the Gaussian kernel
 */
void cannyMainKernelWrap(uchar4 *img_data_h, uchar4 *img_data_d, float *sobel_x, float *sobel_y, int width, int height, float low_th, float high_th, float *gauss_kernel, int g_kernel_size, bool is_video)
{
    size_t img_size = width * height * sizeof(float);
    // float milliseconds = 0;
    // float milliseconds2 = 0;
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    float *img_sobel, *sobel_directions, *lbcs_img, *tts_img, *img_debug_h;
    float *output_d;

    // cudaEvent_t start, stop;
    // cudaEvent_t start2, stop2;
    img_debug_h = (float *)malloc(img_size);

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudamallocs
    cudaMalloc(&img_sobel, img_size);
    cudaMalloc(&sobel_directions, img_size);
    cudaMalloc(&lbcs_img, img_size);
    cudaMalloc(&tts_img, img_size);
    cudaMalloc(&output_d, img_size);
    // cudaMalloc(&img_data_d, width * height * 3 * sizeof(unsigned char));

    // 1. Combining gradients to get magnitude and direction of each pixel
    combineGradientsKernel<<<grid, block>>>(sobel_x, sobel_y, img_sobel, sobel_directions, width, height);
    // // save image
    // cudaMemcpy(img_debug_h, img_sobel, img_size, cudaMemcpyDeviceToHost);
    // cv::Mat printImage(height, width, CV_32F, img_debug_h);
    // cv::Mat displayImage1;
    // printImage.convertTo(displayImage1, CV_8UC1, 1.0);
    // cv::imwrite("debug/combined_gradients_cuda.jpg", displayImage1);
    cudaDeviceSynchronize();
    // 2. Lower bound cutoff suppression to suppress non-maximum pixels with respect to the gradient direction
    // cudaEventRecord(start);
    lowerBoundCutoffSuppression_sh<<<grid, block>>>(img_sobel, sobel_directions, lbcs_img, width, height);
    // cudaMemcpy(img_debug_h, lbcs_img, img_size, cudaMemcpyDeviceToHost);
    // cv::Mat printImage2(height, width, CV_32F, img_debug_h);
    // cv::Mat displayImage2;
    // printImage2.convertTo(displayImage2, CV_8UC1, 1.0);
    // cv::imwrite("debug/lbcs_cuda.jpg", displayImage2);
    cudaDeviceSynchronize();
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Elapsed time for LBCS: %f ms\n", milliseconds);

    // 3. Double thresholding suppression to mark edge pixels as strong, weak or non-edge
    doubleThresholdSuppression<<<grid, block>>>(lbcs_img, output_d, width, height, low_th, high_th);
    // cudaMemcpy(img_debug_h, output_d, img_size, cudaMemcpyDeviceToHost);
    // cv::Mat printImage3(height, width, CV_32F, img_debug_h);
    // cv::Mat displayImage3;
    // printImage3.convertTo(displayImage3, CV_8UC1, 1.0);
    // cv::imwrite("debug/double_threshold_cuda.jpg", displayImage3);
    cudaDeviceSynchronize();
    // cudaEventCreate(&start2);
    // cudaEventCreate(&stop2);

    // 4. Hysteresis to mark weak edge pixels as strong if they are connected to strong edge pixels
    // cudaEventRecord(start2);
    if (!is_video)
    {
        for (int i = 0; i < 30; i++)
        {
            hysteresis<<<grid, block>>>(output_d, width, height, true);
            cudaDeviceSynchronize();
        }
    }
    hysteresis<<<grid, block>>>(output_d, width, height, false);
    cudaDeviceSynchronize();
    // cudaEventRecord(stop2);
    // cudaEventSynchronize(stop2);
    // cudaEventElapsedTime(&milliseconds2, start2, stop2);
    // printf("Elapsed time for Hysteresis: %f ms\n", milliseconds2);

    // Copying img data from cv object to device, then overwriting it with the output of the kernel and copying it back to the host
    // cudaMemcpy(img_data_d, img_data, width * height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    copy1ChannelTo4<<<grid, block>>>(output_d, img_data_d, width, height);
    cudaMemcpy(img_data_h, img_data_d, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);

    cudaFree(output_d);
    cudaFree(img_sobel);
    cudaFree(sobel_directions);
    cudaFree(lbcs_img);
    cudaFree(tts_img);
    // cudaFree(img_data_d);
    free(img_debug_h);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in canny kernel wrap: %s\n", cudaGetErrorString(err));
    }
}
/**
 * @brief Driver function for the Harris corner detection algorithm.
 *
 * @param img_data_h Host image data
 * @param img_data_d Device image data
 * @param sobel_x Sobel kernel in the x direction
 * @param sobel_y Sobel kernel in the y direction
 * @param width Width of the image
 * @param height Height of the image
 * @param k K parameter for the Harris corner detection
 * @param alpha Alpha parameter for the Harris corner detection
 * @param gaussian_kernel Gaussian kernel
 * @param g_kernel_size Gaussian kernel size
 * @param shi_tomasi Flag to enable Shi-Tomasi corner detection
 * @param harris_map_d Harris map output
 * @return treshold value
 */
float harrisMainKernelWrap(uchar4 *img_data_h, uchar4 *img_data_d, float *sobel_x, float *sobel_y, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size, bool shi_tomasi, float *harris_map_d)
{
    int n = width * height;
    // float milliseconds = 0;
    float max_value_f = -FLT_MAX;
    float *Ix2_d, *Iy2_d, *IxIy_d, *IxIy_d2, *detM_d, *traceM_d;
    float *output_d;
    float *max_value_d;

    cudaStream_t streams[3];
    // cudaEvent_t start, stop;
    const dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y,
                        1);
    size_t sh_mem_size = blockSize.x * blockSize.y * sizeof(float);
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

#pragma region Memory Allocation
    // cuda malloc
    cudaMalloc(&Ix2_d, n * sizeof(float));
    cudaMalloc(&Iy2_d, n * sizeof(float));
    cudaMalloc(&IxIy_d, n * sizeof(float));
    cudaMalloc(&IxIy_d2, n * sizeof(float));
    cudaMalloc(&detM_d, n * sizeof(float));
    cudaMalloc(&traceM_d, n * sizeof(float));
    cudaMalloc(&output_d, n * sizeof(float));
    cudaMalloc(&max_value_d, sizeof(float));

    cudaMemcpy(max_value_d, &max_value_f, sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA streams
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
#pragma endregion

#pragma region 1-2. Preparing Gradients
    // 1. Compute Ix^2, Iy^2, and Ix*Iy in parallel using streams
    vecMul<<<gridSize, blockSize, 0, streams[0]>>>(sobel_x, sobel_x, Ix2_d, width, height);  // Ix^2
    vecMul<<<gridSize, blockSize, 0, streams[1]>>>(sobel_y, sobel_y, Iy2_d, width, height);  // Iy^2
    vecMul<<<gridSize, blockSize, 0, streams[2]>>>(sobel_x, sobel_y, IxIy_d, width, height); // Ix * Iy

    // Synchronize streams
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);

    // 2. Apply Gaussian blur to Ix2, Iy2, and IxIy


    // cudaEventRecord(start);
    convolutionGPUWrap(Ix2_d, Ix2_d, width, height, gaussian_kernel, g_kernel_size);
    convolutionGPUWrap(Iy2_d, Iy2_d, width, height, gaussian_kernel, g_kernel_size);
    convolutionGPUWrap(IxIy_d, IxIy_d, width, height, gaussian_kernel, g_kernel_size);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Elapsed time for Gaussian blur: %f ms\n", milliseconds);
#pragma endregion

#pragma region 3-5. Computing Harris/ShiTomasi Response
    if (!shi_tomasi)
    {

        // 3. Compute det(M) = Ix2 * Iy2 - (IxIy)^2
        vecMul<<<gridSize, blockSize, 0, streams[0]>>>(IxIy_d, IxIy_d, IxIy_d2, width, height); // (IxIy)^2
        vecMul<<<gridSize, blockSize, 0, streams[1]>>>(Ix2_d, Iy2_d, detM_d, width, height);    // Ix2 * Iy2
        vecSub<<<gridSize, blockSize>>>(detM_d, IxIy_d2, detM_d, width, height);                // det(M)

        // 4. Compute trace(M) = Ix2 + Iy2
        vecAdd<<<gridSize, blockSize>>>(Ix2_d, Iy2_d, traceM_d, width, height); // trace(M)

        // 5. Compute response R = det(M) / trace(M)
        vecDiv<<<gridSize, blockSize>>>(detM_d, traceM_d, output_d, width, height);
    }
    else
    {
        // 5b. Compute the Shi-Tomasi response. R =  min(trace / 2 + sqrtf(trace * trace / 4 - determinant), trace / 2 . sqrtf(trace * trace / 4 - determinant))
        computeShiTommasiResponse<<<gridSize, blockSize>>>(Ix2_d, Iy2_d, IxIy_d, output_d, width, height);
    }
    cudaDeviceSynchronize();
#pragma endregion

#pragma region 6. Non-maximum suppression on the Harris response
    // cudaEventRecord(start);
    nonMaximumSuppression<<<gridSize, blockSize>>>(output_d, output_d, width, height);

    if (harris_map_d != nullptr)
        cudaMemcpy(harris_map_d, output_d, width * height * sizeof(float), cudaMemcpyDeviceToDevice);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Elapsed time for NMS: %f ms\n", milliseconds);
#pragma endregion

#pragma region 7. Finding Max value in Harris response

    findMaxReductionSHFL<<<gridSize, blockSize, sh_mem_size>>>(output_d, max_value_d, width, height);

    // moving max to host so that we can use it for thresholding
    cudaMemcpy(&max_value_f, max_value_d, sizeof(float), cudaMemcpyDeviceToHost);

#pragma endregion

#pragma region 8. Corner thresholding
    float treshold = max_value_f * alpha;
    if (harris_map_d == nullptr) // Naive optical flow, no need to color corners
        cornerColoring<<<gridSize, blockSize>>>(output_d, img_data_d, width, height, treshold);
    cudaMemcpy(img_data_h, img_data_d, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
#pragma endregion

#pragma region Cleanup
    cudaFree(Ix2_d);
    cudaFree(Iy2_d);
    cudaFree(IxIy_d);
    cudaFree(IxIy_d2);
    cudaFree(detM_d);
    cudaFree(traceM_d);
    cudaFree(output_d);
    cudaFree(max_value_d);
    // cudaFree(img_data_d);

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaStreamDestroy(streams[2]);

    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
#pragma endregion

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in harris kernel wrap: %s\n", cudaGetErrorString(err));
    }

    return treshold;
}


/**
 * @brief  Kernel for naive demo of motion detection
 * 
 * @param harris1 Harris map of first frame
 * @param harris2 Harris map of second frame
 * @param width width of the images
 * @param height height of the images
 * @param threshold threshold to consider a corner
 * @param tollerance tollerance (to lower the threshold)
 * @param window window size to search for matching corners
 * @param idx1Mapping_d mapping of corners from first frame
 * @param idx2Mapping_d mapping of corners from second frame
 * @param mappingCount_d number of common corners
 */

__global__ void mapCommonCornersKernel(const float *__restrict__ harris1,
                                       const float *__restrict__ harris2,
                                       int width, int height,
                                       float threshold, float tolerance,
                                       int window,
                                       int *idx1Mapping_d, int *idx2Mapping_d,
                                       int *mappingCount_d)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int idx = y * width + x;

    __shared__ float sharedHarris1[TILE_WIDTH][TILE_WIDTH];

    int localX = threadIdx.x;
    int localY = threadIdx.y;

    sharedHarris1[localY][localX] = harris1[idx];

    __syncthreads();

    // Only process if the corner response is significant
    if (sharedHarris1[localY][localX] > threshold)
    {
        float bestMatchDiff = threshold * tolerance;
        int bestMatchIdx = -1;

        for (int j = -window; j <= window; j++)
        {
            int y2 = y + j;
            if (y2 < 0 || y2 >= height)
                continue;

            for (int i = -window; i <= window; i++)
            {
                int x2 = x + i;
                if (x2 < 0 || x2 >= width)
                    continue;

                int idx2 = y2 * width + x2;
                float diff = fabsf(sharedHarris1[localY][localX] - harris2[idx2]);

                if (harris2[idx2] > threshold && diff < bestMatchDiff)
                {
                    bestMatchDiff = diff;
                    bestMatchIdx = idx2;
                }
            }
        }

        if (bestMatchIdx != -1)
        {
            int pos = atomicAdd(mappingCount_d, 1);
            idx1Mapping_d[pos] = idx;
            idx2Mapping_d[pos] = bestMatchIdx;
        }
    }
}

/**
 * @brief Wrapper for the naive demo of motion detection
 * 
 * @param harris1 Harris map of first frame
 * @param harris2 Harris map of second frame
 * @param width width of the images
 * @param height height of the images
 * @param threshold threshold to consider a corner
 * @param tollerance tollerance (to lower the threshold)
 * @param window window size to search for matching corners
 * @param idx1Mapping_d mapping of corners from first frame
 * @param idx2Mapping_d mapping of corners from second frame
 * @return number of common corners
 */
int mapCommonKernelWrap(const float *harris1,
                        const float *harris2,
                        int width,
                        int height,
                        float threshold,
                        const float tollerance,
                        int window,
                        int *idx1Mapping_d,
                        int *idx2Mapping_d)
{
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    int *mappingCount_d;
    cudaMalloc(&mappingCount_d, sizeof(int));
    cudaMemset(mappingCount_d, 0, sizeof(int));

    mapCommonCornersKernel<<<grid, block>>>(harris1, harris2, width, height, threshold, tollerance, window, idx1Mapping_d, idx2Mapping_d, mappingCount_d);
    cudaDeviceSynchronize();

    int mappingCount_h;
    cudaMemcpy(&mappingCount_h, mappingCount_d, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(mappingCount_d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error in mapCommon kernel wrap: %s\n", cudaGetErrorString(err));
    }
    return mappingCount_h;
}

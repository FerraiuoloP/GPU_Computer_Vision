#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <assert.h>
// to profile ncu -o otsu2 ./build/main -O -f="input/traffic.jpg"

/**
 * @brief Compute the histogram of the image. For each pixel value, increment the count of the histogram.
 * Optimized version, approx 4x faster than the non-optimized version.
 * @param image The image to compute the histogram of.
 * @param histogram The histogram to store the pixel values.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void computeHistogram(float *image, int *histogram, int width, int height)
{
    __shared__ int shared_histogram[256];

    shared_histogram[threadIdx.x] = 0;
    __syncthreads();

    const int total_pixels = width * height;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // each thread processes multiple pixels to reduce total blocks
    for (int i = tid; i < total_pixels; i += blockDim.x * gridDim.x)
    {
        int x = i % width;
        int y = i / width;
        float pixel = image[y * width + x];
        // casting pixel to bin number
        int bin = (int)pixel;
        atomicAdd(&shared_histogram[bin], 1);
    }
    __syncthreads();

    // merging shared histogram to global histogram
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
    {
        atomicAdd(&histogram[i], shared_histogram[i]); // Global atomic (fewer collisions)
    }
}
// Non optimized version
// __global__ void computeHistogram(float *image, int *histogram, int width, int height)
// {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int index = y * width + x;

//     if (x < width && y < height)
//     {
//         atomicAdd(&histogram[(int)image[index]], 1);
//     }
// }
/**
 * @brief Compute the probability for each pixel value(0-255) in the histogram, by dividing the histogram value by the total number of pixels.
 *
 * @param image
 * @param histogram
 * @param probabilities
 * @param length
 */
__global__ void computeProbabilitiesHistogram(float *image, int *histogram, float *probabilities, int length)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < length)
    {
        probabilities[x] = (float)histogram[x] / (float)length;
    }
}

/**
 * @brief Compute the mean of the histogram.
 *
 * @param histogram The histogram to compute the mean of.
 * @param start From which level to compute the mean.
 * @param threshold At which level to compute the mean.
 * @param isMean The flag to compute the mean or the weights.
 * @param weights The weights of the class. Used only if isMean is True.
 * @return The mean of the histogram.
 */
__device__ float computeMeanOrWeights(float *histogram, int start, int threshold, int isMean, float weights)
{
    float sum = 0;
    for (int i = start; i < threshold; i++)
    {
        if (isMean)
            sum += i * histogram[i];
        else
            sum += histogram[i];
    }
    if (isMean)
        return sum / weights;
    return sum;
}

/**
 * @brief Compute the Otsu threshold of the image.
 *
 * @param image The image to compute the Otsu threshold of.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return The Otsu threshold of the image.
 */
/**
 *
 * 1. Calcola le probabilità di ogni pixel value
 * 1.a mu_t = sum_i_to_l(i*pi) -> media totale
 * 1.b mu(k)
*  2. Per ogni threshold k(0-255)
    a Calcola w0 -> somma delle probabilità dell'hist da hist[0] a hist[k]
    b Calcola w1 -> 1-w0
        - w0 e w1 sono le probabilità delle due classi(background e foreground)
    c Calcola mu0 -> sum_i_to_k(i*pi/w0)
    d Calcola mu1 -> sum_k+1_to_l(i*pi/w1)
        - mu0 e mu1 sono le medie delle due classi(background e foreground)
    e Calcola sigma^2_B -> w0*w1(mu1-mu0)^2

    Il threshold migliore è quello con Sigma^2_B massima
 */
__global__ void otsuThresholdKernel(float *image, float *probabilities, int width, int height, int *sigma2_b)
{
    // int x = blockIdx.x * blockDim.x + threadIdx.x;
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // int index = y * width + x;
    int index = threadIdx.x;

    __shared__ float shared_probabilities[256];
    shared_probabilities[threadIdx.x] = probabilities[threadIdx.x];
    __syncthreads();

    // it' s useless this check since we are always working with 256 threads for a 256x1 histogram
    // if (x < width && y < height && index < 255)
    // {

    float w0 = computeMeanOrWeights(shared_probabilities, 0, index, 0, 0);
    float w1 = 1 - w0;

    float mu0 = computeMeanOrWeights(shared_probabilities, 0, index, 1, w0);
    float mu1 = computeMeanOrWeights(shared_probabilities, index + 1, 255, 1, w1);

    // float w0 = computeMeanOrWeights(probabilities, 0, index, 0, 0);
    // float mu0 = computeMeanOrWeights(probabilities, 0, index, 1, w0);
    // float mu1 = computeMeanOrWeights(probabilities, index + 1, 255, 1, w1);

    float var = w0 * w1 * ((mu0 - mu1) * (mu0 - mu1));

    // Compute the Sigma^2_B for that threshold.
    sigma2_b[index] = (int)var;
    // }
}
/**
 * @brief Find the maximum value in the array of sigma2_b. Shared-memory based.
 *
 * @param sigma2_b The array of sigma2_b.
 * @param max_threshold The maximum threshold for which the sigma2_b is maximum.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void findMaxReductionSHRD(int *sigma2_b, int *max_threshold, int width, int height)
{
    __shared__ int sigma2b_shared[256];
    __shared__ int sigma2b_shared_idx[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sigma2b_shared[tid] = sigma2_b[gid];
    sigma2b_shared_idx[tid] = gid;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2)
    {
        if (tid < offset)
        {
            sigma2b_shared[tid] = sigma2b_shared[tid] > sigma2b_shared[tid + offset] ? sigma2b_shared[tid] : sigma2b_shared[tid + offset];
            sigma2b_shared_idx[tid] = sigma2b_shared[tid] > sigma2b_shared[tid + offset] ? sigma2b_shared_idx[tid] : sigma2b_shared_idx[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        max_threshold[0] = sigma2b_shared_idx[0];
    }
}

/**
 * @brief Warp-level max reduction exploiting shfl intrinsic. At the end, the thread in the lane 0 will have the maximum value of the warp.
 * @param val value to be reduced that is passed between threads withing same warp
 */
__device__ int warpReduceMax(int val)
{
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}
/**
 * @brief Block-level max reduction. At the end, the thread in the lane 0 will have the maximum value of the whole block.
 *
 * @param val  value to be reduced that is passed between threads withing same warp in a given block.
 */
__device__ int blockReduceMax(int val)
{
    __shared__ int shared[256 / 32];
    int tid = threadIdx.x;

    int lane = tid % warpSize;
    int warpId = tid / warpSize;

    val = warpReduceMax(val);

    if (lane == 0)
    {
        shared[warpId] = val;
    }

    __syncthreads();

    if (warpId == 0)
    {
        val = (tid < blockDim.x / warpSize) ? shared[lane] : 0;
        val = warpReduceMax(val);
    }
    return val;
}

/**
 * @brief Finds the maximum value inside a given integer array of fixed size 256. Shuffle-based.
 *
 * @param sigma2_b It will contain the sigma2_b values for each threshold.
 * @param max_threshold It will contain the maximum threshold for which the sigma2_b is maximum.
 */
__global__ void findMaxReductionSHFL(int *sigma2_b, int *max_threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    // stride
    int stride = gridDim.x * blockDim.x;

    int local_max = -INT32_MAX;
    int local_max_idx = -1;

#pragma unroll
    for (int i = x; i < 256; i += stride)
    {
        if (sigma2_b[i] > local_max)
        {
            local_max = sigma2_b[i];
            local_max_idx = i;
        }
    }
    int val = local_max_idx | local_max << 16;
    // int val = local_max;
    int block_max = blockReduceMax(val);

    if (threadIdx.x == 0)
    {
        atomicMax(max_threshold, block_max);
    }
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        max_threshold[0] = max_threshold[0] & 0xFFFF;
        // printf("Max threshold: %d\n", *max_threshold);
    }
}

/**
 * @brief Binarize an image using a given threshold.
 *
 * @param output_d Output image.
 * @param img_d Input image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param threshold Threshold to binarize the image.
 */
__global__ void binarizeImgKernel(unsigned char *output_d, float *img_d, int width, int height, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        output_d[idx] = (unsigned char)(img_d[idx] > threshold ? 255 : 0);
    }
}

/**
 * @brief Binarizes an image using a given threshold.
 *
 * @param img_gray_h Host image.
 * @param img_d Device image.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param threshold Threshold to binarize the image.
 */
void binarizeImgWrapper(unsigned char *img_gray_h, float *img_d, int width, int height, int threshold)
{
    unsigned char *output_d;
    size_t img_size = width * height;
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaMalloc(&output_d, img_size * sizeof(unsigned char));

    binarizeImgKernel<<<gridSize, blockSize>>>(output_d, img_d, width, height, threshold);

    cudaMemcpy(img_gray_h, output_d, img_size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(output_d);
}

/**
 * @brief Compute the Otsu threshold of the image.
 *
 * @param image The image to compute the Otsu threshold of.
 * @param width The width of the image.
 * @param height The height of the image.
 * @return The Otsu threshold of the image.
 */
int otsuThreshold(float *image, int width, int height)
{
    // const dim3 blockSize(16, 16);
    // const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    const dim3 blockSize(256, 1);
    const dim3 gridSize((width * height + blockSize.x - 1) / blockSize.x, 1);
    // 256x1 thread since we only work with a 256x1 histogram
    const dim3 gridSize2(1, 1);
    const dim3 blockSize2(256, 1);
    // float milliseconds = 0;
    int *histogram;
    float *probabilities;
    int *sigma2_b;
    int *max_threshold_d;
    int *max_threshold_h = (int *)malloc(1 * sizeof(int));
    // cudaEvent_t start, stop;

    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    cudaMalloc(&histogram, 256 * sizeof(int));
    cudaMemset(histogram, 0, 256 * sizeof(int));
    cudaMalloc(&probabilities, 256 * sizeof(float));
    cudaMalloc(&sigma2_b, 256 * sizeof(int));

    // histogram

    computeHistogram<<<gridSize, blockSize>>>(image, histogram, width, height);

    // probabilities
    computeProbabilitiesHistogram<<<gridSize2, blockSize2>>>(image, histogram, probabilities, width * height);

    // sigma2b for each threshold
    // cudaEventRecord(start);
    otsuThresholdKernel<<<gridSize2, blockSize2>>>(image, probabilities, width, height, sigma2_b);
    // cudaEventRecord(stop);

    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Otsu threshold elapsed time: %f ms\n", milliseconds);

    // Second part of otsu where we find the effective max threshold

    cudaMalloc(&max_threshold_d, 1 * sizeof(int));

    // cudaEventRecord(start);
    // findMaxReductionSHRD<<<gridSize2, blockSize2>>>(sigma2_b, max_threshold_d, width, height);
    findMaxReductionSHFL<<<gridSize2, blockSize2>>>(sigma2_b, max_threshold_d);

    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // printf("Find max threshold CUDA elapsed time: %f ms\n", milliseconds);

    cudaMemcpy(max_threshold_h, max_threshold_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Max threshold Cuda: %d\n", max_threshold_h[0]);
    cudaDeviceSynchronize();

    cudaFree(histogram);
    cudaFree(probabilities);
    cudaFree(sigma2_b);
    cudaFree(max_threshold_d);
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    return max_threshold_h[0];
}

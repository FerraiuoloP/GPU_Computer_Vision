#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <assert.h>

/**
 * @brief Compute the histogram of the image. For each pixel value, increment the count of the histogram.
 *
 * @param image The image to compute the histogram of.
 * @param histogram The histogram to store the pixel values.
 * @param width The width of the image.
 * @param height The height of the image.
 */
__global__ void computeHistogram(float *image, int *histogram, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height)
    {
        atomicAdd(&histogram[(int)image[index]], 1);
    }
}
/**
 * @brief Compute the probability for each pixel value(0-255) in the histogram, by dividing the histogram value by the total number of pixels.
 *
 * @param image
 * @param histogram
 * @param probabilities
 * @param length
 * @return __global__
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
__global__ void otsu_threshold_kernel(float *image, int *histogram, float *probabilities, int width, int height, int *sigma2_b)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height && index < 255)
    {

        float w0 = computeMeanOrWeights(probabilities, 0, index, 0, 0);
        float w1 = 1 - w0;

        float mu0 = computeMeanOrWeights(probabilities, 0, index, 1, w0);
        float mu1 = computeMeanOrWeights(probabilities, index + 1, 255, 1, w1);

        float var = w0 * w1 * ((mu0 - mu1) * (mu0 - mu1));

        // Compute the Sigma^2_B for that threshold.
        sigma2_b[index] = (int)var;
    }
}

__global__ void find_max_reduction(int *sigma2_b, int *max_threshold, int width, int height)
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

__global__ void binarize_img_kernel(unsigned char *output_d, float *img_d, int width, int height, int threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = x * height + y;
        output_d[idx] = (unsigned char)(img_d[idx] > threshold ? 255 : 0);
    }
}

void binarize_img_wrapper(unsigned char *img_gray_h, float *img_d, int width, int height, int threshold)
{
    unsigned char *output_d;
    size_t img_size = width * height;
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaMalloc(&output_d, img_size * sizeof(unsigned char));

    binarize_img_kernel<<<gridSize, blockSize>>>(output_d, img_d, width, height, threshold);

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
int otsu_threshold(float *image, int width, int height)
{
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
    const dim3 gridSize2(1, 1);
    const dim3 blockSize2(256, 1);
    float milliseconds = 0;
    int *histogram;
    float *probabilities;
    int *sigma2_b;
    int *max_threshold_d;
    int *max_threshold_h = (int *)malloc(1 * sizeof(int));
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&histogram, 256 * sizeof(int));
    cudaMemset(histogram, 0, 256 * sizeof(int));
    cudaMalloc(&probabilities, 256 * sizeof(float));
    cudaMalloc(&sigma2_b, 256 * sizeof(int));

    // histogram
    computeHistogram<<<gridSize, blockSize>>>(image, histogram, width, height);

    // probabilities
    computeProbabilitiesHistogram<<<gridSize2, blockSize2>>>(image, histogram, probabilities, width * height);

    // sigma2b for each threshold
    cudaEventRecord(start);
    otsu_threshold_kernel<<<gridSize2, blockSize2>>>(image, histogram, probabilities, width, height, sigma2_b);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Otsu threshold elapsed time: %f ms\n", milliseconds);

    // Second part of otsu where we find the effective max threshold

    // Find Max Parallel
    cudaEventRecord(start);

    cudaMalloc(&max_threshold_d, 1 * sizeof(int));
    find_max_reduction<<<gridSize2, blockSize2>>>(sigma2_b, max_threshold_d, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Find max threshold CUDA elapsed time: %f ms\n", milliseconds);

    cudaMemcpy(max_threshold_h, max_threshold_d, 1 * sizeof(int), cudaMemcpyDeviceToHost);
    printf("Max threshold Cuda: %d\n", max_threshold_h[0]);
    cudaDeviceSynchronize();

    cudaFree(histogram);
    cudaFree(probabilities);
    cudaFree(sigma2_b);
    cudaFree(max_threshold_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return max_threshold_h[0];
}

void binarize_img(float *img_d, float *img_h, int width, int height)
{
    int threshold = otsu_threshold(img_d, width, height);
    for (int i = 0; i < width * height; i++)
    {
        img_h[i] = img_h[i] > threshold ? 255 : 0;
    }
}

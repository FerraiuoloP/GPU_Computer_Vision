// // List wrapper function callable by .cpp file.
// // TODO: define the wrapper funtions to be used wherever it is required by other CPP files
// void rgbToGrayKernelWrap(unsigned char *img_d, float *gray_d, int N, int M);
// void gaussianBlurKernelWrap(float *img_d, float *img_out_d, int N, int M, float *kernel, int kernel_size);
// void combineGradientsKernelWrap(float *img_sobel_x, float *img_sobel_y, float *img_sobel, float *sobel_directions, int width, int height);
// void harrisCornerKernelWrap(float *img_sobel_x, float *img_sobel_y, float *img_harris, int width, int height, float k);
// void mapKernelToRGBWrap(float *img_harris, unsigned char *img_harris_rgb, int width, int height);
// void harrisMainKernelWrap(float *sobel_x, float *sobel_y, float *output, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size);
// void cannyMainKernelWrap(float *sobel_x, float *sobel_y, float *output, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size);

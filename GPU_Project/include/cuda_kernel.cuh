const int FILTER_WIDTH = 3;
const float FILTER_SIGMA = 1.75f;
// const float FILTER_SIGMA = 3;
const float ALPHA = 0.05;
const float K = 0.05;

void rgbToGrayKernelWrap(unsigned char *img_d, float *gray_d, int N, int M);
void gaussianBlurKernelWrap(float *img_d, float *img_out_d, int N, int M, float *kernel, int kernel_size);
void harrisCornerKernelWrap(float *img_sobel_x, float *img_sobel_y, float *img_harris, int width, int height, float k);
void mapKernelToRGBWrap(float *img_harris, unsigned char *img_harris_rgb, int width, int height);
void harrisMainKernelWrap(unsigned char *img_data, float *sobel_x, float *sobel_y, int width, int height, float k, float alpha, float *gaussian_kernel, int g_kernel_size, bool shi_tomasi);
void cannyMainKernelWrap(unsigned char *img_data, float *sobel_x, float *sobel_y, int width, int height, float low_threshold, float high_threshold, float *gaussian_kernel, int g_kernel_size);
void convolutionGPUWrap(float *d_Result, float *d_Data, int data_w, int data_h, float *d_kernel);
int otsu_threshold(float *image, int width, int height);
void binarize_img(float *img_d, float *img_h, int width, int height);
void binarize_img_wrapper(unsigned char *img_h, float *img_d, int width, int height, int threshold);
void convolutionGPU2Wrap(float *img_input_d, float *mask, float *P, int channels, int width, int height);
void findMaxWrap(int *arr_d, int *arr_d_o, int size);

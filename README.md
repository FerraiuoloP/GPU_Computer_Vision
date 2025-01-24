# GPU_Project_Final
## Description
A project for the GPU Programming course in the Computer Engineering Master Degree at Politecnico di Torino.
The project consinsts in a CUDA implementation of various computer visiion algorithms, such as:
- **Canny Edge Detector:** t detects edges in an image, that is, areas where the intensity of the image changes rapidly.
- **Shi-Tomasi/Harris Corner Detector:** it detects corners in an image,that is, points where two edges meet.
- **Otsu's Thresholding:** it is a method used to automatically pick the threshold value that best discriminates between the background and foreground of an image. It can be used for both image binarization and in Canny edge detection for the thresholding step.

The project is able to process both images and videos.
So far, only jpg,png and mp4 files are supported.
Technically, it can support all OpenCV formats, so a pull request is welcome to add more formats.

## How to use
First, move to the project directory:
```bash
cd GPU_Project
```
To compile the project, run the following command:
```bash
make all
```
An example of how to run the project is the following:
```bash
make run ARGS="-H -f=input/traffic.jpg"
```
### Arguments
- **Operating mode:**
  - **-H:** Harris Corner Detector
  - **-S:** Shi-Tomasi Corner Detector
  - **-C:** Canny Edge Detector
  - **-O:** Otsu's thresholding for image binarization
  - **-A:** Process all the algorithms at once using multi-threading.
- **-f:** path to the input image or video
- **Additional arguments:**
    - **Canny** can be operated also in two different modes:
        - **-g:** GUI Mode. This allows to selects the thresholds interactively. Only available for images.
        - **Manual thresholds:** You can also specify the thresholds manually by adding the following arguments:
            - **-l:** lower threshold
            - **-h:** upper threshold
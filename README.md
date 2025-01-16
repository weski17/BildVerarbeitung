# Bildverarbeitung und Mustererkennung (Image Processing and Pattern Recognition)

This repository contains solutions to image processing tasks as part of a practical course for the summer semester 2024. The tasks include histogram analysis, edge detection, circle detection, and hand skeletonization.

## Table of Contents

1. [Overview](#overview)
2. [Tasks](#tasks)
   - [Task 1: Histogram Analysis](#task-1-histogram-analysis)
   - [Task 2: Edge Detection and Circle Detection](#task-2-edge-detection-and-circle-detection)
   - [Task 3: Hand Skeletonization](#task-3-hand-skeletonization)


---

## Overview

This project implements various computer vision techniques from scratch using Python and OpenCV. The main focus areas include histogram computation, image segmentation, edge detection, circle detection, and hand skeletonization in video sequences or images.

---

## Tasks

### Task 1: Histogram Analysis

- **1D Histogram**:
  - Compute and visualize the histogram of a grayscale image.
- **2D Histogram**:
  - Create a 2D histogram based on the green and blue channels of a color image.
- **Segmentation**:
  - Select a region in the 2D histogram and visualize all pixels in the image that fall within this range.

### Task 2: Edge Detection and Circle Detection

- **Canny Edge Detector**:
  - **Binomial Filtering**: Smooth the image with a 5x5 binomial filter.
  - **Sobel Filtering**: Compute gradient magnitudes and directions.
  - **Non-Maximum Suppression**: Thin edges by suppressing non-maximum values.
  - **Hysteresis Thresholding**: Connect strong and weak edges to create the final edge map.
- **Hough Circle Detection**:
  - Implement a Hough transformation to detect circles in an image or video, allowing flexible parameterization for radius detection.
- **Eye Detection**:
  - Detect eyes in a video using the Hough transformation. The detected circles are overlaid on each video frame.

### Task 3: Hand Skeletonization

- **Segmentation**:
  - Segment a hand in each video frame or a single image using the Region Growing Algorithm. The user selects the seed pixel, and segmentation is based on grayscale homogeneity and pixel count constraints.
- **Hole Filling**:
  - Fill holes in the segmented hand mask using a closing operation with customizable structural elements.
- **Skeletonization**:
  - Reduce the segmented hand mask to a 1-pixel-wide skeleton and visualize the result.

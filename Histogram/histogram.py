"""
 * Author: Wael Eskeif
 * Date: 01/05/2024
 *
 * This script provides functionality to:
 * (a) Compute and visualize a 1D histogram for grayscale images.
 * (b) Compute and visualize a 2D histogram for color images (Blue-Green channels).
 * (c) Segment a selected region of the 2D histogram.
 """

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def computeHistogram(image_gray):
    """
    Compute the 1D histogram of a grayscale image.

    Args:
        image_gray (np.ndarray): Grayscale image with two dimensions (rows, cols).

    Returns:
        np.ndarray: Histogram array of size 256 containing pixel counts.

    Raises:
        ValueError: If the provided image is not grayscale.
    """
    if len(image_gray.shape) != 2:
        raise ValueError("The provided image is not a grayscale image with two dimensions.")

    rows, cols = image_gray.shape
    histogram = np.zeros(256, dtype=int)

    for row in range(rows):
        for col in range(cols):
            pixel_value = image_gray[row, col]
            histogram[pixel_value] += 1

    return histogram

def visualizeHistogram(histogram):
    """
    Visualize a 1D histogram of a grayscale image.

    Args:
        histogram (np.ndarray): 1D histogram array of size 256.
    """
    plt.figure()
    plt.title('1D Histogram of the Grayscale Image')
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    plt.bar(range(256), histogram, color='black')
    plt.xlim([0, 256])
    plt.show()

def computeHistogram2D(image_color):
    """
    Compute a 2D histogram for the Blue and Green channels of a color image.

    Args:
        image_color (np.ndarray): Color image in BGR format.

    Returns:
        np.ndarray: 2D histogram array of size 256x256.

    Raises:
        ValueError: If the input image is None.
    """
    if image_color is None:
        raise ValueError("Image could not be loaded")

    image = cv.cvtColor(image_color, cv.COLOR_BGR2RGB)
    blue_channel = image[:, :, 2]
    green_channel = image[:, :, 1]

    image_height, image_width = blue_channel.shape
    histo = np.zeros((256, 256), dtype=int)

    for i in range(image_height):
        for j in range(image_width):
            blue_value = blue_channel[i, j]
            green_value = green_channel[i, j]
            histo[blue_value, green_value] += 1

    return histo

def visualizeHistogram2D(histogram2D):
    """
    Visualize a 2D histogram of the Blue and Green channels.

    Args:
        histogram2D (np.ndarray): 2D histogram array of size 256x256.
    """
    plt.imshow(histogram2D, interpolation='nearest', extent=[0, 256, 0, 256], aspect='auto', origin='lower')
    plt.title('2D Histogram of Blue and Green Channels')
    plt.xlabel('Blue Channel')
    plt.ylabel('Green Channel')
    plt.colorbar()
    plt.show()

def segment_histogram(histogram):
    """
    Segment a selected region of the provided 2D histogram.

    Args:
        histogram (np.ndarray): 2D histogram array to segment.

    Raises:
        ValueError: If the provided histogram is None or invalid.
    """
    if histogram is None:
        raise ValueError("The provided histogram is invalid.")

    max_value = np.max(histogram)
    if max_value == 0:
        print("Histogram maximum value is zero, unable to normalize.")
        return

    image_for_display = np.uint8(255 * histogram / max_value)

    cv.namedWindow('Histogram Display', cv.WINDOW_NORMAL)
    cv.imshow('Histogram Display', image_for_display)
    cv.waitKey(1)
    roi = cv.selectROI('Histogram Display', image_for_display, showCrosshair=True)
    cv.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No region was selected.")
        return

    x, y, width, height = map(int, roi)
    extracted_region = histogram[y:y+height, x:x+width]
    rows, cols = extracted_region.shape
    segmented_region = np.zeros((rows, cols), dtype=np.uint8)
    threshold_value = 100

    for i in range(rows):
        for j in range(cols):
            segmented_region[i, j] = 255 if extracted_region[i, j] > threshold_value else 0

    cv.namedWindow('Segmentation Result', cv.WINDOW_NORMAL)
    cv.imshow('Segmentation Result', cv.resize(segmented_region, (200, 200)))
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # Load the image
    image_color = cv.imread('strand.jpg')
    if image_color is None:
        raise ValueError("No image provided. Please provide a valid grayscale image.")

    # Convert to a grayscale image
    image_gray = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)

    # (a) 1D Histogram
    histogram = computeHistogram(image_gray)
    visualizeHistogram(histogram)

    # (b) 2D Histogram
    histogram2D = computeHistogram2D(image_color)
    visualizeHistogram2D(histogram2D)

    # (c) Segmentation
    segment_histogram(histogram2D)

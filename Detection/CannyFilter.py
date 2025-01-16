"""
 * Author: Wael Eskeif
 * Date: 01/05/2024
 *
 * This script implements edge detection using a series of steps:
 * (1) Binomial filtering to smooth the image.
 * (2) Sobel filtering to compute gradient magnitudes and directions.
 * (3) Non-maximum suppression to thin edges.
 * (4) Hysteresis thresholding to connect strong and weak edges.
 """

import cv2 as cv
import numpy as np
import math

def binomial_filter():
    """
    Generate a 5x5 binomial filter kernel.

    Returns:
        np.ndarray: A 5x5 binomial filter kernel normalized to sum to 1.
    """
    coefficients = np.array([1, 4, 6, 4, 1])
    kernel = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            kernel[i, j] = (coefficients[i] * coefficients[j]) / 256  # Normalization
    return kernel

def apply_binomial_filter(image, kernel):
    """
    Apply a binomial filter to a grayscale image.

    Args:
        image (np.ndarray): Input grayscale image.
        kernel (np.ndarray): 5x5 binomial filter kernel.

    Returns:
        np.ndarray: Filtered grayscale image.
    """
    kernel_size = kernel.shape[0]
    edge = kernel_size // 2
    rows, cols = image.shape
    filtered_image = np.zeros_like(image)

    for i in range(edge, rows - edge):
        for j in range(edge, cols - edge):
            region = image[i - edge:i + edge + 1, j - edge:j + edge + 1]
            filtered_image[i, j] = (kernel * region).sum()

    return filtered_image

def apply_sobel_filter(image):
    """
    Apply Sobel filters to compute gradient magnitude and direction.

    Args:
        image (np.ndarray): Input grayscale image.

    Returns:
        tuple: Gradient magnitude and direction images.
    """
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    sobel_x = np.zeros_like(image)
    sobel_y = np.zeros_like(image)

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            region = image[i - 1:i + 2, j - 1:j + 2]
            sobel_x[i, j] = np.sum(region * Gx)
            sobel_y[i, j] = np.sum(region * Gy)

    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.float32)
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    return gradient_magnitude, gradient_direction

def convert_to_degrees_and_limit(angle_rad):
    """
    Convert an angle in radians to degrees and restrict it to [0, 180).

    Args:
        angle_rad (float): Angle in radians.

    Returns:
        float: Angle in degrees restricted to [0, 180).
    """
    angle_deg = angle_rad * (180 / math.pi)
    return angle_deg % 180

def non_max_suppression(gradient_magnitude, gradient_direction):
    """
    Apply non-maximum suppression to thin edges.

    Args:
        gradient_magnitude (np.ndarray): Gradient magnitude image.
        gradient_direction (np.ndarray): Gradient direction image in radians.

    Returns:
        np.ndarray: Thinned edge image after suppression.
    """
    rows, cols = gradient_magnitude.shape
    output = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_angle = convert_to_degrees_and_limit(gradient_direction[i, j])
            q = r = 255

            if (0 <= current_angle < 22.5) or (157.5 <= current_angle < 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif (22.5 <= current_angle < 67.5):
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif (67.5 <= current_angle < 112.5):
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif (112.5 <= current_angle < 157.5):
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if (gradient_magnitude[i, j] >= q) and (gradient_magnitude[i, j] >= r):
                output[i, j] = gradient_magnitude[i, j]

    return output

def hysteresis(image, low_threshold, high_threshold):
    """
    Perform hysteresis thresholding to connect strong and weak edges.

    Args:
        image (np.ndarray): Input image after non-maximum suppression.
        low_threshold (float): Low threshold for edge detection.
        high_threshold (float): High threshold for edge detection.

    Returns:
        np.ndarray: Image with connected edges.
    """
    rows, cols = image.shape
    output = np.zeros_like(image)

    strong = 255
    weak = 0

    for i in range(rows):
        for j in range(cols):
            if image[i, j] >= high_threshold:
                output[i, j] = strong
            elif image[i, j] >= low_threshold:
                output[i, j] = weak

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if output[i, j] == weak:
                if any(output[i + di, j + dj] == strong for di in (-1, 0, 1) for dj in (-1, 0, 1)):
                    output[i, j] = strong

    return output

def normalize_image(image):
    """
    Normalize an image to the range [0, 255].

    Args:
        image (np.ndarray): Input image to normalize.

    Returns:
        np.ndarray: Normalized image.
    """
    normalized_image = (image - image.min()) / (image.max() - image.min()) * 255
    return normalized_image.astype(np.uint8)

if __name__ == "__main__":
    """
    Main script for edge detection using the implemented functions.
    """
    bild_url = "loewe.jpeg"
    image_gray = cv.imread(bild_url, cv.IMREAD_GRAYSCALE)

    if image_gray is None:
        raise FileNotFoundError(f"Image '{bild_url}' not found.")

    binomial_filter1 = binomial_filter()
    filtered_image = apply_binomial_filter(image_gray, binomial_filter1)
    gradient_magnitude, gradient_direction = apply_sobel_filter(filtered_image)

    non_max_suppressed = non_max_suppression(gradient_magnitude, gradient_direction)
    hysteresis_image = hysteresis(non_max_suppressed, 100, 200)

    cv.imshow('Hysteresis Result', hysteresis_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

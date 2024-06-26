# Autor: Wael Eskeif
# Datum: 12.06.2024

import cv2 as cv
import numpy as np

image_gray = None
def regionGrowing(image_gray, seedPoint, localThreshold, maxNumOfPixelThreshold):
    """
    Performs region growing segmentation on a grayscale image.

    Args:
        image_gray (numpy.ndarray): The grayscale image on which the algorithm will be applied.
        seedPoint (tuple): The starting point (x, y) for region growing.
        localThreshold (int): The threshold for pixel similarity (homogeneity). Neighboring pixels are included
                              in the region if their intensity difference with the seed pixel is less than this value.
        maxNumOfPixelThreshold (int): The maximum number of pixels to include in the segmented region.

    Returns:
        segmented: A binary image where segmented pixels are marked with 255 (white) and others with 0 (black).
    """
    rows, cols = image_gray.shape
    segmented = np.zeros_like(image_gray)
    seed_grey_value = image_gray[seedPoint[0], seedPoint[1]]

    queue = [seedPoint]
    numSegmentedPixels = 1

    while queue and numSegmentedPixels < maxNumOfPixelThreshold:
        x, y = queue.pop(0)
        segmented[x, y] = 255  # Mark the pixel as segmented

        # Überprüfe benachbarte Pixel
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy  # Position of the neighboring pixel

            # Check if the neighboring pixel is within the image boundaries and not yet segmented
            if 0 <= nx < rows and 0 <= ny < cols and segmented[nx, ny] == 0:
                # Check the homogeneity criterion
                if abs(int(image_gray[nx, ny]) - int(seed_grey_value)) < localThreshold:
                    segmented[nx, ny] = 255  # Pixel als segmentiert markieren
                    queue.append((nx, ny))
                    numSegmentedPixels += 1

    return segmented

def mouseEvent(event, x, y, flags, param):
    """
    Handles mouse events for the OpenCV window.

    Args:
        event: The type of the mouse event (e.g., cv.EVENT_LBUTTONDOWN).
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags: Any relevant flags passed by OpenCV (not used here).
        param: Additional parameters passed to the callback function (not used here).

    If the left mouse button is clicked, the function starts the region growing process from the clicked point.
    It applies a Gaussian blur to the grayscale image, detects edges using the Canny method, and then
    performs region growing segmentation. The result is displayed in a new window.
    """
    global image_gray
    if event == cv.EVENT_LBUTTONDOWN:
        seedPoint = [y, x]

        localThreshold = 70
        maxNumOfPixelThreshold = 100000
        segmented_image = regionGrowing(image_gray, seedPoint, localThreshold, maxNumOfPixelThreshold)

        struct_element = np.ones((3, 3), np.uint8)

        closed_image = close_image(segmented_image, struct_element)

        eroded_image = erode(closed_image, struct_element)
        edges_image = subtract(closed_image, eroded_image)

        cv.imshow('Segmented Image', segmented_image)
        cv.imshow('Closed Image', closed_image)
        cv.imshow('Edges Image', edges_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

def dilate(binary_image, struct_element):
    """
    Perform dilation on a binary image using the given structuring element.

    Args:
        binary_image: The input binary image (values 0 or 255).
        struct_element: The structuring element for dilation.

    Returns:
        output_image: The dilated binary image.
    """
    rows, cols = binary_image.shape
    struct_rows, struct_cols = struct_element.shape
    struct_x = struct_rows // 2
    struct_y = struct_cols // 2
    output_image = np.zeros_like(binary_image)

    for i in range(rows):
        for j in range(cols):
            if binary_image[i, j] == 255:
                for m in range(struct_rows):
                    for n in range(struct_cols):
                        if struct_element[m, n] == 1:
                            # Calculate the corresponding position in the output image
                            x = i + m - struct_x
                            y = j + n - struct_y
                            if 0 <= x < rows and 0 <= y < cols:
                                # Set the corresponding pixel in the output image to foreground (white)
                                output_image[x, y] = 255

    return output_image

def erode(binary_image, struct_element):
    """
    Perform erosion on a binary image using the given structuring element.

    Args:
        binary_image (numpy.ndarray): The input binary image (values 0 or 255).
        struct_element (numpy.ndarray): The structuring element for erosion.

    Returns:
        output_image: The eroded binary image.
    """
    rows, cols = binary_image.shape
    struct_rows, struct_cols = struct_element.shape
    struct_x = struct_rows // 2
    struct_y = struct_cols // 2

    output_image = np.zeros_like(binary_image)

    for i in range(rows):
        for j in range(cols):
            match = True
            for m in range(struct_rows):
                for n in range(struct_cols):
                    # Check if the structuring element at this position is 1
                    if struct_element[m, n] == 1:
                        x = i + m - struct_x
                        y = j + n - struct_y
                        # Check if the resulting coordinate is within the image bounds
                        if x < 0 or x >= rows or y < 0 or y >= cols or binary_image[x, y] != 255:
                            match = False
                            break
                if not match:
                    break
            # If all pixels in the structuring element match and are within bounds, set the output pixel to 255
            if match:
                output_image[i, j] = 255

    return output_image

def close_image(binary_image, struct_element):
    """
    Perform morphological closing on a binary image.

    Args:
        binary_image: The input binary image (values 0 or 255).
        struct_element: The structuring element for closing.

    Returns:
        closed: The binary image after closing.
    """
    dilated = dilate(binary_image, struct_element)
    closed = erode(dilated, struct_element)
    return closed

def subtract(image1, image2):
    """
    subtracts image2 from image1.

    Args:
        image1 (numpy.ndarray): The first input image.
        image2 (numpy.ndarray): The second input image to be subtracted from the first.

    Returns:
        output_image: The result of the subtraction.
    """
    output_image = np.zeros_like(image1)
    rows, cols = image1.shape

    for i in range(rows):
        for j in range(cols):
            output_image[i, j] = max(image1[i, j] - image2[i, j], 0)

    return output_image

if __name__ == "__main__":
    image = cv.imread('hand.jpeg')
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.namedWindow('image')
    cv.setMouseCallback('image', mouseEvent)

    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

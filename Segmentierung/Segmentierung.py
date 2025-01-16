import cv2 as cv
import numpy as np

def regionGrowing(image_gray, seedPoint, localThreshold, maxNumOfPixelThreshold):
    """
    Performs region growing segmentation on a grayscale image.
    """
    rows, cols = image_gray.shape
    segmented = np.zeros_like(image_gray, dtype=np.uint8)
    seed_grey_value = image_gray[seedPoint[0], seedPoint[1]]
    queue = [seedPoint]
    numSegmentedPixels = 0

    while queue and numSegmentedPixels < maxNumOfPixelThreshold:
        x, y = queue.pop(0)
        if segmented[x, y] == 0:  # Check if the pixel has not already been segmented
            segmented[x, y] = 255
            numSegmentedPixels += 1

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and segmented[nx, ny] == 0:
                    if abs(int(image_gray[nx, ny]) - seed_grey_value) < localThreshold:
                        queue.append((nx, ny))

    return segmented

def dilate(binary_image, struct_element):
    """
    Perform dilation on a binary image using the given structuring element.
    """
    rows, cols = binary_image.shape
    output_image = np.zeros_like(binary_image)
    k, l = struct_element.shape
    k2, l2 = k // 2, l // 2

    for i in range(rows):
        for j in range(cols):
            max_value = 0
            for di in range(-k2, k2 + 1):
                for dj in range(-l2, l2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if struct_element[di + k2, dj + l2]:
                            max_value = max(max_value, binary_image[ni, nj])
            output_image[i, j] = max_value

    return output_image

def erode(binary_image, struct_element):
    """
    Perform erosion on a binary image using the given structuring element.
    """
    rows, cols = binary_image.shape
    output_image = np.zeros_like(binary_image)
    k, l = struct_element.shape
    k2, l2 = k // 2, l // 2

    for i in range(rows):
        for j in range(cols):
            min_value = 255
            all_ones = True
            for di in range(-k2, k2 + 1):
                for dj in range(-l2, l2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        if struct_element[di + k2, dj + l2] and binary_image[ni, nj] == 0:
                            all_ones = False
                            break
                if not all_ones:
                    break
            if all_ones:
                output_image[i, j] = 255

    return output_image

def close_image(binary_image, struct_element):
    """
    Perform morphological closing on a binary image.
    """
    dilated = dilate(binary_image, struct_element)
    closed = erode(dilated, struct_element)
    return closed

def skeletonize(binary_image):
    """
    Skeletonize the binary image to a one-pixel wide structure using morphological operations.
    """
    skel = np.zeros_like(binary_image)
    img = binary_image.copy()
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = img - temp
        skel = skel | temp
        img = eroded
        if cv.countNonZero(img) == 0:
            break

    return skel

def mouseEvent(event, x, y, flags, param):
    global image_gray
    if event == cv.EVENT_LBUTTONDOWN:
        segmented_image = regionGrowing(image_gray, (y, x), 70, 100000)
        struct_element = np.ones((3, 3), np.uint8)
        closed_image = close_image(segmented_image, struct_element)
        skeletonized_image = skeletonize(closed_image)

        cv.imshow('Segmented Image', segmented_image)
        cv.imshow('Closed Image', closed_image)
        cv.imshow('Skeletonized Image', skeletonized_image)

if __name__ == "__main__":
    image = cv.imread('hand.jpeg')
    if image is None:
        print("Error loading image.")
        exit()
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    cv.namedWindow('image')
    cv.setMouseCallback('image', mouseEvent)
    cv.imshow('image', image_gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

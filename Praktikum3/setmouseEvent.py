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
        numpy.ndarray: A binary image where segmented pixels are marked with 255 (white) and others with 0 (black).
    """

    rows, cols = image_gray.shape
    segmented = np.zeros_like(image_gray)
    seed_grey_value = image_gray[seedPoint[0], seedPoint[1]]

    queue = [seedPoint]
    numSegmentedPixels = 1

    while queue and numSegmentedPixels < maxNumOfPixelThreshold:
        x, y = queue.pop(0)
        segmented[x, y] = 255  # Pixel als segmentiert markieren

        # Überprüfe benachbarte Pixel
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy # Position des Nachbarpixels

            # Überprüfe, ob das Nachbarpixel innerhalb der Bildgrenzen liegt und noch nicht segmentiert wurde.
            if 0 <= nx < rows and 0 <= ny < cols and segmented[nx, ny] == 0:
                # Überprüfe Homogenitätskriterium
                if abs(int(image_gray[nx, ny]) - int(seed_grey_value)) < localThreshold:
                    segmented[nx, ny] = 255  # Pixel als segmentiert markieren
                    queue.append((nx, ny))
                    numSegmentedPixels += 1

    return segmented


def mouseEvent(event, x, y, flags, param):
    global image_gray
    if event == cv.EVENT_LBUTTONDOWN:
        seedPoint = [y, x]

        filterd_image = cv.GaussianBlur(image_gray, (5, 5), 0)
        canny_image = cv.Canny(filterd_image, 100, 250)

        localThreshold = 70
        maxNumOfPixelThreshold = 90000
        segmented_image = regionGrowing(image_gray, seedPoint, localThreshold, maxNumOfPixelThreshold)

        cv.imshow('image', segmented_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


image = cv.imread('hand.jpeg')
image_gray = cv.imread('hand.jpeg', cv.IMREAD_GRAYSCALE)

# Fenster erstellen und Mausereignisse verknüpfen
cv.namedWindow('image')
cv.setMouseCallback('image', mouseEvent)

# Bild anzeigen
cv.imshow('image', image)
cv.waitKey(0)
cv.destroyAllWindows()

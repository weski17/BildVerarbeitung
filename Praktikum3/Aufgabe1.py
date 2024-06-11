import cv2 as cv
import numpy as np

def mouseEvent(event, x, y, flags, param):
    global image_gray

    if event == cv2EVENT_LBUTTONDOWN:
        seedPoint = [y,x]

        # ------------------------------------------------------------------------
        # (a) Segmentierung
        # ------------------------------------------------------------------------
        # 1. Implementieren Sie die Methode
        mask = regionGrowing(image_gray, seedPoint, localThreshold, maxNumOfPixelThreshold)
        # 2. Visualisieren Sie Ihr Ergebnis


def prepare_image(image):
    gray_image = cv.imread(image,cv.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise ValueError("1")
    filterd_image = cv.GaussianBlur(gray_image, ())



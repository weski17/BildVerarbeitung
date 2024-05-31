import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def binomial_filter():
    coefficients = np.array([1, 4, 6, 4, 1])
    kernel = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            kernel[i, j] = (coefficients[i] * coefficients[j]) / 256  # zur Normalisierung
    return kernel


def apply_binomial_filter(image, kernel):

    kernel_size = kernel.shape[0]
    edge = kernel_size // 2  # 5 / 2 = 2 (//)
    rows, cols = image.shape

    filtered_image = np.zeros_like(image)

    # Die Faltung nur im inneren Bereich des Bildes anwenden, wo volle Nachbarschaft existiert
    for i in range(edge, rows - edge):
        for j in range(edge, cols - edge):
            # Die Region, die der Größe des Kernels entspricht, auswählen
            region = image[i - edge:i + edge + 1, j - edge:j + edge + 1]
            # Das Produkt von Kernel und Bildregion summieren
            filtered_image[i, j] = (kernel * region).sum()

    return filtered_image


def apply_sobel_filter_simple(image):
    # Definition der Sobel-Kernmatrizen für X- und Y-Richtung
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Initialisierung der Ausgabe-Arrays für die Gradientenkomponenten
    sobel_x = np.zeros_like(image)
    sobel_y = np.zeros_like(image)

    # Iteration über das Bild, ohne die Randpixel zu berücksichtigen
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            # Auszug des aktuellen 3x3 Bereichs
            region = image[i - 1:i + 2, j - 1:j + 2]
            # Berechnung der Gradientenkomponenten in X- und Y-Richtung
            sobel_x[i, j] = np.sum(region * Gx)
            sobel_y[i, j] = np.sum(region * Gy)

    gradient_magnitude = np.zeros_like(sobel_x, dtype=np.float32)
    # Berechnung der Gradientenstärke
    rows, cols = sobel_x.shape
    for i in range(rows):
        for j in range(cols):
            gradient_magnitude[i, j] = np.sqrt(sobel_x[i, j] ** 2 + sobel_y[i, j] ** 2)


    gradient_direction = np.zeros_like(sobel_x, dtype=float)

    for i in range(rows):
        for j in range(cols):
            # Berechnen des Winkels mit arctan2 für jedes Element
            gradient_direction[i, j] = np.arctan2(sobel_y[i, j], sobel_x[i, j])

    normalized_image = cv.normalize(gradient_direction, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    
    return gradient_magnitude, gradient_direction


def non_max_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    output = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            current_angle = np.degrees(gradient_direction[i, j]) % 180

            q = 255
            r = 255

            # Bestimmen der Nachbarn für die Non-Maxima Suppression
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
            else:
                output[i, j] = 0
    return output


def hysteresis(image, low_threshold, high_threshold):
    rows, cols = image.shape
    output = np.zeros_like(image)

    strong = 255
    weak = 100

    for i in range(rows):
        for j in range(cols):
            if image[i, j] >= high_threshold:
                output[i, j] = strong
            elif image[i, j] >= low_threshold:
                output[i, j] = weak

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if output[i, j] == weak:
                if ((output[i + 1, j - 1] == strong) or (output[i + 1, j] == strong) or (output[i + 1, j + 1] == strong)
                        or (output[i, j - 1] == strong) or (output[i, j + 1] == strong)
                        or (output[i - 1, j - 1] == strong) or (output[i - 1, j] == strong) or (
                                output[i - 1, j + 1] == strong)):
                    output[i, j] = strong
                else:
                    output[i, j] = 0

    return output



def normalize_image(image):
    normalized_image = (image - image.min()) / (image.max() - image.min()) * 255
    return normalized_image.astype(np.uint8)



bild_url = "loewe.jpeg"
image_gray = cv.imread(bild_url, cv.IMREAD_GRAYSCALE)
if not image_gray:
    exit(1)
binomial_filter1 = binomial_filter()
filterd_bild = apply_binomial_filter(image_gray, binomial_filter1)
sobel_bild = apply_sobel_filter_simple(filterd_bild)

gradient_magnitude, gradient_direction = sobel_bild
non_max_suppressed = non_max_suppression(gradient_magnitude, gradient_direction)
hysteresis_bild = hysteresis(non_max_suppressed, 100, 200)
normalize_image = normalize_image(gradient_direction)





cv.imshow('hysteresis_bild', normalize_image)
cv.waitKey(0)  # Warten auf Tastendruck
cv.destroyAllWindows()  # Schließen aller geöffneten Fenster

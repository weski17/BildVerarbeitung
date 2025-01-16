import cv2 as cv
import numpy as np
import math
def prepare_image(image_path):
    """
    Prepares an image for circle detection by converting it to grayscale and applying the Canny edge detector.

    Parameters:
    - image_path: Path to the input image.

    Returns:
    - edges: The edge-detected image.
    - original_image: The original image in color.
    """
    original_image = cv.imread(image_path)
    if original_image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None
    image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(image, threshold1=100, threshold2=200)  # Benutzung für schnelles Testing
    return edges, original_image

def deg_to_rad(degrees):
    """Konvertiert Grad in Radiant."""
    return degrees * math.pi / 180

def taylor_sin(x, n=5):
    """Berechnet Sinus mit Hilfe der Taylor-Reihe."""
    result = 0
    sign = 1
    for i in range(1, n*2, 2):
        term = sign * (x**i) / math.factorial(i)
        result += term
        sign *= -1
    return result

def taylor_cos(x, n=5):
    """Berechnet Cosinus mit Hilfe der Taylor-Reihe."""
    result = 1
    sign = -1
    for i in range(2, n*2, 2):
        term = sign * (x**i) / math.factorial(i)
        result += term
        sign *= -1
    return result


def HoughCircles(I, min_radius, max_radius):
    # Bild in Graustufen umwandeln, falls es nicht bereits in Graustufen ist
    if len(I.shape) == 3:
        I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)

    # Kanten im Bild finden
    edges = cv.Canny(I, 100, 200)

    rows, cols = edges.shape

    radien = max_radius - min_radius + 1
    # Akkumulator-Array initialisieren 3D
    Acc = np.zeros((rows, cols, radien), dtype=np.int32)

    # Durchlaufe alle Bildkoordinaten
    for u in range(rows):
        for v in range(cols):
            if edges[u, v] > 0:  # Edge pixel
                for r in range(min_radius, max_radius + 1):  # All possible radii
                    for theta_deg in range(0, 360, 20):  # Angle iteration in 20-degree steps
                        theta_rad = deg_to_rad(theta_deg)  # Convert degree to radian
                        a = int(u - r * taylor_cos(theta_rad))  # X-center
                        b = int(v - r * taylor_sin(theta_rad))  # Y-center
                        if 0 <= a < rows and 0 <= b < cols:
                            Acc[a, b, r - min_radius] += 1


    # Finde die maximalen Kreise im Akkumulator-Array
    MaxCircles = []
    threshold = 5  # Schwellenwert
    for a in range(rows): # Y-Koordinaten
        for b in range(cols): # X-Koordinaten
            for r in range(radien):
                if Acc[a, b, r] > threshold:
                    MaxCircles.append((a, b, r + min_radius))

    return MaxCircles

def visualize_circles(original_image, circles):
    """
    Visualizes detected circles on the original image.

    Parameters:
    - original_image: The original image in color.
    - circles: List of circles to draw.
    """
    for x, y, r in circles:
        cv.circle(original_image, (y, x), r, (0, 0, 200), 2)
    cv.imshow('Detected Circles', original_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    edge_image, original_image = prepare_image("loewe.jpeg")
    if edge_image is None or original_image is None:
        exit(0)
    min_radius = 20
    max_radius = 30
    circles = HoughCircles(edge_image, min_radius, max_radius)
    visualize_circles(original_image, circles)

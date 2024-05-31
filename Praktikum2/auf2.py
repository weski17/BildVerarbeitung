import cv2 as cv
import numpy as np

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
            if edges[u, v] > 0:  # Kantenpunkt
                for r in range(min_radius, max_radius + 1): # alle mögliche Radien
                    for theta in range(0, 360, 20): # Winkel Iteration in 20 Grad Schritten
                        a = int(u - r * np.cos(np.deg2rad(theta))) # X -Mittlepunkt
                        b = int(v - r * np.sin(np.deg2rad(theta))) # Y -Mittelpunkt
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

# Beispielverwendung
if __name__ == "__main__":
    edge_image, original_image = prepare_image("loewe.jpeg")
    if edge_image is None or original_image is None:
        exit(0)
    min_radius = 20
    max_radius = 30
    circles = HoughCircles(edge_image, min_radius, max_radius)
    visualize_circles(original_image, circles)

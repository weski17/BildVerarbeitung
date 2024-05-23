# Autor: Wael Eskeif
# Datum: 01/05/2024

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def computeHistogram(image_gray):
    # Check if the image has the correct dimensions.
    if len(image_gray.shape) != 2:
        raise ValueError("The provided image is not a grayscale image with two dimensions.")

    # Copy the provided image to leave the original image unchanged.
    image_tmp = image_gray.copy()

    # Determine the number of rows and columns in the image.
    rows, cols = image_tmp.shape

    # Initialize a histogram array with 256 zeros of type integer.
    histogram = np.zeros(256, dtype=int)

    # Loop through each row of the image.
    for row in range(rows):
        # Loop through each column in the row.
        for col in range(cols):
            # Read the grayscale value of the current pixel.
            pixel_value = image_tmp[row, col]
            # Increase the counter in the histogram at the index of the read grayscale value.
            histogram[pixel_value] += 1

    return histogram

def visualizeHistogram(histogram):
    # Start a new drawing.
    plt.figure()
    # Set the title of the diagram.
    plt.title('1D Histogram of the Grayscale Image')
    # Label the X-axis with 'Pixel Value'.
    plt.xlabel('Pixel Value')
    # Label the Y-axis with 'Number of Pixels'.
    plt.ylabel('Number of Pixels')
    # Create a bar chart showing the values from the histogram.
    plt.bar(range(256), histogram, color='black')
    # Limit the X-axis to display values from 0 to 256.
    plt.xlim([0, 256])
    # Display the diagram.
    plt.show()

def computeHistogram2D(image_color):
    if image_color is None:
        print("Image could not be loaded")
    else:
        # Convert from BGR to RGB
        image = cv.cvtColor(image_color, cv.COLOR_BGR2RGB)

        # Extract blue and green channels
        blue_channel = image[:, :, 2]
        green_channel = image[:, :, 1]

        # Determine image size
        image_height, image_width = blue_channel.shape

        # Create a 256x256 array filled with zeros
        histo = np.zeros((256, 256), dtype=int)

        # Iterate through each pixel
        for i in range(image_height):
            for j in range(image_width):
                # Save the blue and green color Value for the current pixel
                blue_value = blue_channel[i, j]
                green_value = green_channel[i, j]
                # Increment the histogram bin corresponding to the blue and green values
                histo[blue_value, green_value] += 1
        return histo
    


def visualizeHistogram2D(histogram2D):
    # Display the 2D histogram using matplotlib's imshow function.
    plt.imshow(histogram2D, interpolation='nearest', extent=[0, 256, 0, 256], aspect='auto', origin='lower')
    # Set the title of the plot.
    plt.title('2D Histogram of Blue and Green Channels')
    # Label the x-axis as 'Blue Channel'.
    plt.xlabel('Blue Channel')
    plt.ylabel('Green Channel')
    plt.colorbar()
    # Display the plot.
    plt.show()

def segment_histogram(histogram):
    # Check if the histogram is valid (not None)
    if histogram is None:
        raise ValueError("The provided histogram is invalid.")
    
    # Normalize the histogram to use the full range of 0-255 for display
    max_value = np.max(histogram)
    if max_value == 0:  # Prevent division by zero if the histogram is empty
        print("Histogram maximum value is zero, unable to normalize.")
        return
    image_for_display = np.uint8(255 * histogram / max_value)

    # Display the normalized histogram and allow the user to select a ROI
    cv.namedWindow('Histogram Display', cv.WINDOW_NORMAL)  # Allow window resizing
    cv.imshow('Histogram Display', image_for_display)
    cv.waitKey(1)  # Display the window until a key is pressed or ROI is selected
    roi = cv.selectROI('Histogram Display', image_for_display, showCrosshair=True)
    # Close the window after ROI selection
    cv.destroyAllWindows()  # Close the window after ROI selection

    # Check if a valid ROI was selected
    if roi == (0, 0, 0, 0):
        print("No region was selected.")
        return

    # Extract the selected region based on the ROI coordinates
    x, y, width, height = map(int, roi)
    extracted_region = histogram[y:y+height, x:x+width]
    # Create a new array to hold the segmented region
    rows, cols = extracted_region.shape
    segmented_region = np.zeros((rows, cols), dtype=np.uint8)
    # Apply a manual threshold to segment the region
    threshold_value = 100  
    for i in range(rows):
        for j in range(cols):
            # Apply a threshold test to each pixel
            segmented_region[i, j] = 255 if extracted_region[i, j] > threshold_value else 0

    # Display the segmentation result, ensuring visibility
    cv.namedWindow('Segmentation Result', cv.WINDOW_NORMAL)  # Allow window resizing
    cv.imshow('Segmentation Result', cv.resize(segmented_region, (200, 200)))  # Resize image for better visibility
    cv.waitKey(0)
    cv.destroyAllWindows()


# Load the image
image_color = cv.imread('strand.jpg') 
# Check if the image is present.
if image_color is None:
    raise ValueError("No image provided. Please provide a valid grayscale image.")

# Convert to a grayscale image
image_gray = cv.cvtColor(image_color, cv.COLOR_BGR2GRAY)

# (a) 1D Histogram
histogram = computeHistogram(image_gray)

visualizeHistogram(histogram)

# ------------------------------------------------------------------------
# (b) 2D Histogram
# ------------------------------------------------------------------------
histogram2D = computeHistogram2D(image_color)
# 2. Visualize your result
visualizeHistogram2D(histogram2D)

# ------------------------------------------------------------------------
# (c) Segmentation
# ------------------------------------------------------------------------
segment_histogram(histogram2D)

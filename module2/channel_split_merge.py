import cv2
import numpy as np

# Load the image
image_path = 'puppy.jpg'
image = cv2.imread(image_path)

if image is None:
        raise FileNotFoundError("Image not found. Make sure 'puppy.jpg' exists in the same directory.")

    # Split into B, G, R channels (OpenCV uses BGR by default)
    b_channel, g_channel, r_channel = cv2.split(image)

    # Save or display individual channels as grayscale images
    cv2.imshow("Red Channel (Grayscale)", r_channel)
    cv2.imshow("Green Channel (Grayscale)", g_channel)
    cv2.imshow("Blue Channel (Grayscale)", b_channel)

    # Merge back the channels to form the original image
    merged_original = cv2.merge([b_channel, g_channel, r_channel])
    cv2.imshow("Reconstructed Original Image", merged_original)

    # Swap Red and Green channels (GRB)
    merged_grb = cv2.merge([b_channel, r_channel, g_channel])
    cv2.imshow("Red-Green Swapped (GRB Image)", merged_grb)

    # Wait for key press to close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
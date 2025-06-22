import cv2
import os

# Load the image from the current directory
image = cv2.imread('brain.jpg')

# Check if the image was loaded successfully
if image is None:
    print("Error: Could not load image. Make sure 'brain.jpg' is in the same folder as this script.")
    exit()

# Display the image in a window
cv2.imshow('Brain Image', image)

# Wait until a key is pressed
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

# Get the user's desktop path
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

# Define the full path for the output file
output_path = os.path.join(desktop_path, 'brain_copy.jpg')

# Write the image to the desktop
cv2.imwrite(output_path, image)

print(f"Image successfully written to: {output_path}")

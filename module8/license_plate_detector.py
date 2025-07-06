import cv2
import numpy as np
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the Russian license plate cascade using absolute path
cascade_path = os.path.join(script_dir, 'haarcascade_russian_plate_number.xml')
plate_cascade = cv2.CascadeClassifier(cascade_path)

# List of images to process using absolute paths
image_files = [
    os.path.join(script_dir, 'russian_license_plate1.jpeg'),
    os.path.join(script_dir, 'russian_license_plate2.jpeg'),
    os.path.join(script_dir, 'non-russian_license_plate.jpeg')
]

def detect_plate(gray, cascade, try_preprocessing=False):
    # Run the detector
    plates = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    # If detection fails, try preprocessing
    if len(plates) == 0 and try_preprocessing:
        # Histogram equalization
        gray_eq = cv2.equalizeHist(gray)
        plates = cascade.detectMultiScale(gray_eq, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
        # If still not detected, try adaptive thresholding
        if len(plates) == 0:
            gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            plates = cascade.detectMultiScale(thresh, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    return plates

for img_file in image_files:
    print(f'Processing {img_file}')
    img = cv2.imread(img_file)
    if img is None:
        print(f"Could not open {img_file}, skipping...")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1st attempt: raw grayscale image
    plates = detect_plate(gray, plate_cascade)
    
    # 2nd attempt: try preprocessing if no plates found
    if len(plates) == 0:
        print("No plates detected, trying preprocessing...")
        plates = detect_plate(gray, plate_cascade, try_preprocessing=True)
    
    # Draw red bounding boxes around detected plates
    for (x, y, w, h) in plates:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # Extract, rotate, and scale plate region (placeholder for further processing)
        plate_roi = gray[y:y+h, x:x+w]
        # For now, just save the cropped region
        roi_filename = f"{os.path.splitext(img_file)[0]}_plate_roi.jpg"
        cv2.imwrite(roi_filename, plate_roi)
        print(f"Saved extracted plate region to {roi_filename}")

    # Save and show the results
    out_filename = f"{os.path.splitext(img_file)[0]}_detected.jpg"
    cv2.imwrite(out_filename, img)
    print(f"Saved result with bounding boxes to {out_filename}")

    # Optionally, display (uncomment for local run)
    # cv2.imshow('Detected Plates', img)
    # cv2.waitKey(0)

# cv2.destroyAllWindows()
print("Processing completed.")

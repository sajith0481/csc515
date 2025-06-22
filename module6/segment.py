import cv2
import matplotlib.pyplot as plt

# Helper function to display results
def show_images(titles, images, cmap=None):
    plt.figure(figsize=(18,6))
    for i, (title, img) in enumerate(zip(titles, images)):
        plt.subplot(1, len(images), i+1)
        plt.title(title)
        plt.imshow(img, cmap=cmap if cmap else None)
        plt.axis('off')
    plt.show()

# Load images (update paths as needed)
img_paths = {
    'Outdoor': 'outdoor.jpg',
    'Indoor': 'indoor.jpg',
    'Close-up': 'close-up.jpg'
}

for title, path in img_paths.items():
    # Read image in grayscale
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,  # you could use THRESH_BINARY_INV for inverted mask
        21,  # Block size (must be odd), tweak if needed
        10   # Constant subtracted from mean, tweak for sensitivity
    )
    # Show original and segmented image
    show_images([f'{title} (Original)', f'{title} (Segmented)'], [img, thresh], cmap='gray')

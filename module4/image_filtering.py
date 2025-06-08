import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('Mod4CT1.jpg', cv2.IMREAD_GRAYSCALE)

# Define kernel sizes
kernel_sizes = [3, 5, 7]

# Define sigma values for Gaussian filter
# Using sigma = 0.3 * ((kernel_size-1) * 0.5 - 1) + 0.8 as a rule of thumb
# This ensures sigma scales appropriately with kernel size
sigma_values = [0.8, 1.5]  # Two different sigma values

# Create figure for subplots
fig, axes = plt.subplots(len(kernel_sizes), 4, figsize=(15, 12))
fig.suptitle('Image Filtering Results', fontsize=16)

# Process image with different filters and kernel sizes
for i, kernel_size in enumerate(kernel_sizes):
    # Mean filter
    mean_filtered = cv2.blur(img, (kernel_size, kernel_size))
    axes[i, 0].imshow(mean_filtered, cmap='gray')
    axes[i, 0].set_title(f'Mean Filter ({kernel_size}x{kernel_size})')
    axes[i, 0].axis('off')
    
    # Median filter
    median_filtered = cv2.medianBlur(img, kernel_size)
    axes[i, 1].imshow(median_filtered, cmap='gray')
    axes[i, 1].set_title(f'Median Filter ({kernel_size}x{kernel_size})')
    axes[i, 1].axis('off')
    
    # Gaussian filter with first sigma value
    gaussian_filtered1 = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_values[0])
    axes[i, 2].imshow(gaussian_filtered1, cmap='gray')
    axes[i, 2].set_title(f'Gaussian Filter (σ={sigma_values[0]})')
    axes[i, 2].axis('off')
    
    # Gaussian filter with second sigma value
    gaussian_filtered2 = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_values[1])
    axes[i, 3].imshow(gaussian_filtered2, cmap='gray')
    axes[i, 3].set_title(f'Gaussian Filter (σ={sigma_values[1]})')
    axes[i, 3].axis('off')

# Add row labels
row_labels = ['3x3 Kernel', '5x5 Kernel', '7x7 Kernel']
for i, label in enumerate(row_labels):
    axes[i, 0].set_ylabel(label, fontsize=12)

# Adjust layout and display
plt.tight_layout()
plt.show() 
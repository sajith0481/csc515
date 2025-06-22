import cv2
import numpy as np

def apply_geometric_transformations(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    rows, cols = image.shape[:2]

    # 1. Translation
    tx, ty = 10, 10
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(image, translation_matrix, (cols, rows))

    # 2. Rotation
    angle = -10
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(translated, rotation_matrix, (cols, rows))

    # 3. Scaling
    scale_factor = 1.2
    scaled = cv2.resize(rotated, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    # 4. Perspective Correction (mock example with hardcoded points)
    # Define 4 source points (corners of the note in the skewed image)
    src_pts = np.float32([
        [20, 20],
        [100, 10],
        [20, 60],
        [100, 70]
    ])
    # Define 4 destination points (aligned rectangle)
    dst_pts = np.float32([
        [0, 0],
        [120, 0],
        [0, 60],
        [120, 60]
    ])
    # Compute perspective transform matrix and apply it
    perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(scaled, perspective_matrix, (130, 70))

    # Save the final image
    cv2.imwrite(output_path, warped)
    print(f"Transformed image saved to: {output_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "banknote.jpg"
    output_image_path = "transformed_banknote_full.jpg"
    apply_geometric_transformations(input_image_path, output_image_path)

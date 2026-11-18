"""
Experiment 5 - Morphological Operations
---------------------------------------
Goal: apply shape-based operations to a binary image:
erosion, dilation, boundary extraction, small-object removal, skeletonization.
Input : none - a synthetic image is drawn in code (no image file needed).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects

# --- Build a test image: black 300x300 canvas with white shapes ---
image = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(image, (50, 50), (150, 150), 255, -1)    # big square   (-1 = filled)
cv2.circle(image, (220, 100), 40, 255, -1)             # big circle
cv2.circle(image, (30, 250), 3, 255, -1)               # tiny speck (noise)
cv2.circle(image, (80, 270), 5, 255, -1)               # small speck (noise)
cv2.rectangle(image, (200, 220), (205, 225), 255, -1)  # small speck (noise)

# Structuring element: the 3x3 "probe" that slides over the image
kernel = np.ones((3, 3), np.uint8)

# Erosion: white regions shrink; tiny specks can vanish
eroded = cv2.erode(image, kernel, iterations=1)

# Dilation: white regions grow; gaps get filled
dilated = cv2.dilate(image, kernel, iterations=1)

# Boundary extraction: morphological gradient = dilation - erosion -> outlines only
boundary = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# Remove small objects: skimage wants a boolean image; drop blobs under 50 pixels
bool_image = image > 0
cleaned_bool = remove_small_objects(bool_image, min_size=50)
cleaned_objects = (cleaned_bool * 255).astype(np.uint8)  # back to 0/255 for display

# Skeletonization: thin each shape down to a 1-pixel-wide centerline
skeleton_bool = skeletonize(bool_image)
skeleton = (skeleton_bool * 255).astype(np.uint8)

# --- Display results in a 2x3 grid ---
titles = ['Original Image', 'Erosion', 'Dilation', 'Boundary Extraction', 'Removed Small Objects', 'Skeletonization']
images = [image, eroded, dilated, boundary, cleaned_objects, skeleton]

plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()

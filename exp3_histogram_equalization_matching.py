"""
Experiment 3 - Histogram Equalization & Histogram Matching
----------------------------------------------------------
Goal: improve image contrast by reshaping the intensity histogram.
  - Equalization: spread intensities evenly across 0-255.
  - Matching (specification): make the image's histogram look like
    that of a reference image.
Inputs: exp3_original.jpg (to be modified), exp3_reference.jpg (target look)
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms


def run_histogram_experiment(original_path, reference_path):
    # Load both as grayscale so each pixel is a single 0-255 intensity
    img_orig = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    img_ref = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)

    # imread returns None (not an exception) when a file can't be read
    if img_orig is None or img_ref is None:
        print("Error: Could not read images. Please check the file paths.")
        return

    # Histogram equalization: flattens the histogram -> stronger contrast
    img_eq = cv2.equalizeHist(img_orig)

    # Histogram matching: remap original's intensities to follow the reference's histogram
    img_matched = match_histograms(img_orig, img_ref)
    img_matched = np.clip(img_matched, 0, 255).astype('uint8')  # back to valid 8-bit

    images = [img_orig, img_eq, img_ref, img_matched]
    titles = ['Original Image', 'Equalized Image', 'Reference Image', 'Matched Image']

    plt.figure(figsize=(16, 8))

    for i in range(4):
        # Top row: the images
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        plt.title(titles[i])
        plt.axis('off')

        # Bottom row: their histograms (pixel count per intensity level)
        plt.subplot(2, 4, i + 5)
        plt.hist(images[i].ravel(), bins=256, range=[0, 256], color='gray', edgecolor='black', alpha=0.7)
        plt.title(titles[i] + ' Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


run_histogram_experiment('exp3_original.jpg', 'exp3_reference.jpg')

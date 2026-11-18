"""
Experiment 1.5 - Contrast Boost + Vignette Effect
-------------------------------------------------
Goal: (1) increase contrast/brightness, (2) darken the image edges
smoothly using a Gaussian mask (a "vignette" look).
Input : exp1_input.jpg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def contrast_and_vignette(image_path):
    img = cv2.imread(image_path)

    # Linear adjustment: output = alpha * pixel + beta
    #   alpha > 1 -> more contrast, beta > 0 -> brighter.
    # convertScaleAbs also clips to the 0-255 range for us.
    contrast_img = cv2.convertScaleAbs(img, alpha=1.3, beta=10)

    rows, cols = contrast_img.shape[:2]

    # Build a 2D Gaussian "bump": 1D Gaussians along each axis, multiplied together.
    # The center is bright (1.0) and it fades towards the edges.
    kernel_x = cv2.getGaussianKernel(cols, cols / 2)
    kernel_y = cv2.getGaussianKernel(rows, rows / 2)
    kernel_2d = kernel_y * kernel_x.T  # (rows x 1) * (1 x cols) = (rows x cols)

    # Normalize so the center value is exactly 1 (no change in the middle)
    mask = kernel_2d / kernel_2d.max()

    # Stack the mask 3 times so it can multiply all B, G, R channels
    mask_3d = np.dstack([mask] * 3)

    # Multiply pixels by the mask: center unchanged, edges darkened
    vignette_img = (contrast_img * mask_3d).astype(np.uint8)

    # BGR -> RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vignette_rgb = cv2.cvtColor(vignette_img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(vignette_rgb)
    axes[1].set_title("Contrast & Vignette Applied")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


contrast_and_vignette("exp1_input.jpg")

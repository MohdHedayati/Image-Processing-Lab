"""
Experiment 2 - Noise Smoothing with Linear & Non-Linear Filters
---------------------------------------------------------------
Goal: add noise to an image, then remove it with different filters and
compare how well each one works.
  - Salt & pepper noise (random black/white pixels) -> Mean (linear) vs Median (non-linear)
  - Gaussian noise (random intensity variation)     -> Gaussian (linear) filter

Key takeaway: the median filter ignores extreme outliers (it sorts the
neighborhood and picks the middle value), so it removes salt & pepper noise
much better than the mean filter, which gets pulled by the 0/255 pixels.
Input : exp2_input.jpg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def add_salt_and_pepper_noise(image, prob):
    """Randomly turn a fraction of pixels black (pepper) or white (salt).

    prob = chance of a pixel becoming black; the same chance applies to white,
    so the total noisy fraction is about 2 * prob.
    """
    noisy_image = np.zeros(image.shape, np.uint8)
    thres = 1 - prob  # random numbers above this become salt
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()      # uniform random number in [0, 1)
            if rdn < prob:
                noisy_image[i][j] = 0          # pepper (black)
            elif rdn > thres:
                noisy_image[i][j] = 255        # salt (white)
            else:
                noisy_image[i][j] = image[i][j]  # keep original pixel
    return noisy_image


def add_gaussian_noise(image, mean=0, sigma=25):
    """Add random values drawn from a Gaussian (bell-curve) distribution to every pixel.

    sigma controls the noise strength: larger sigma = grainier image.
    """
    noise = np.random.normal(mean, sigma, image.shape)
    # Use float to avoid uint8 overflow, then clip back into the valid 0-255 range
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Load as grayscale (single channel) to keep the experiment simple
original_image = cv2.imread('exp2_input.jpg', cv2.IMREAD_GRAYSCALE)

# ---------- Salt & pepper noise: linear vs non-linear filter ----------
sp_noisy_image = add_salt_and_pepper_noise(original_image, 0.05)

# Mean filter (linear): each pixel becomes the average of its 3x3 neighborhood.
# Extreme 0/255 pixels get averaged in, so noise is smeared rather than removed.
mean_filtered = cv2.blur(sp_noisy_image, (3, 3))

# Median filter (non-linear): each pixel becomes the middle value of its sorted
# 3x3 neighborhood. Outliers end up at the ends of the list and are ignored.
median_filtered = cv2.medianBlur(sp_noisy_image, 3)

# ---------- Gaussian noise: Gaussian smoothing ----------
gaussian_noisy_image = add_gaussian_noise(original_image, mean=0, sigma=25)

# Gaussian filter (linear): weighted average where pixels closer to the center
# count more, giving smoother and more natural results than a plain mean.
# (5, 5) = kernel size, 0 = let OpenCV choose the standard deviation from it.
gaussian_filtered = cv2.GaussianBlur(gaussian_noisy_image, (5, 5), 0)

# ---------- Display everything in a 2x3 grid ----------
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sp_noisy_image, cmap='gray')
plt.title('Salt & Pepper Noise')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(mean_filtered, cmap='gray')
plt.title('Mean Filtered')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(median_filtered, cmap='gray')
plt.title('Median Filtered')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.title('Gaussian Noise')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title('Gaussian Filtered')
plt.axis('off')

plt.tight_layout()
plt.show()

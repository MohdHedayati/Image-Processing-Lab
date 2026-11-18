"""
Experiment 1.2 - Color Channel Splitting & Grayscale Conversion
---------------------------------------------------------------
Goal: see what each color channel (B, G, R) contributes to an image,
and compare with the grayscale version.
Input : exp1_input.jpg
"""
import cv2
import matplotlib.pyplot as plt


def split_and_grayscale(image_path):
    # NOTE: OpenCV loads images in BGR order (not RGB!)
    img_bgr = cv2.imread(image_path)

    # Grayscale = weighted sum of channels (0.299R + 0.587G + 0.114B)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Split into three single-channel (2D) images.
    # Bright areas in a channel = lots of that color there.
    b, g, r = cv2.split(img_bgr)

    # Matplotlib expects RGB, so convert before displaying the color image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")
    axes[0].axis('off')

    # cmap='gray' is needed, otherwise matplotlib shows 2D arrays in false color
    axes[1].imshow(b, cmap='gray')
    axes[1].set_title("Blue Channel")
    axes[1].axis('off')

    axes[2].imshow(g, cmap='gray')
    axes[2].set_title("Green Channel")
    axes[2].axis('off')

    axes[3].imshow(r, cmap='gray')
    axes[3].set_title("Red Channel")
    axes[3].axis('off')

    axes[4].imshow(img_gray, cmap='gray')
    axes[4].set_title("Grayscale")
    axes[4].axis('off')

    plt.tight_layout()
    plt.show()


split_and_grayscale("exp1_input.jpg")

"""
Experiment 1.3 - Smart Mirror (Orientation-Dependent Flip)
----------------------------------------------------------
Goal: flip an image based on its shape.
  - Landscape (wider than tall) -> flip horizontally (left <-> right)
  - Portrait / square           -> flip vertically   (top <-> bottom)
Input : exp1_input.jpg
"""
import cv2
import matplotlib.pyplot as plt


def smart_mirror(image_path):
    img_bgr = cv2.imread(image_path)

    # shape is (rows, cols, channels) -> height first, then width
    height, width = img_bgr.shape[:2]

    # cv2.flip flag: 1 = horizontal flip, 0 = vertical flip, -1 = both
    if width > height:
        flipped_bgr = cv2.flip(img_bgr, 1)
        flip_type = "Horizontally"
    else:
        flipped_bgr = cv2.flip(img_bgr, 0)
        flip_type = "Vertically"

    # Convert BGR -> RGB for correct colors in matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    flipped_rgb = cv2.cvtColor(flipped_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original\n({width}x{height})")
    axes[0].axis('off')

    axes[1].imshow(flipped_rgb)
    axes[1].set_title(f"Flipped {flip_type}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


smart_mirror("exp1_input.jpg")

"""
Experiment 1.4 - Proportional Resize
------------------------------------
Goal: resize an image to a width of 500px while keeping its aspect ratio
(so it doesn't look stretched or squashed).
Input : exp1_input.jpg
Output: exp1_4_resized_500px.png
"""
from PIL import Image
import matplotlib.pyplot as plt


def resize_proportional(image_path, output_path="exp1_4_resized_500px.png"):
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Pick the new width, then derive the height from the original aspect ratio
    target_width = 500
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)

    img_resized = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

    # PNG can't store every JPEG mode (e.g. CMYK), so convert if needed
    if img_resized.mode not in ('RGB', 'RGBA'):
        img_resized = img_resized.convert('RGBA')

    img_resized.save(output_path, "PNG")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Original\n({original_width}x{original_height})")
    axes[0].axis('off')

    axes[1].imshow(img_resized)
    axes[1].set_title(f"Proportionally Resized\n({target_width}x{target_height})")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


resize_proportional("exp1_input.jpg")

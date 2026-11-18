"""
Experiment 1.1 - Square Thumbnail Generation
--------------------------------------------
Goal: turn any image into a 250x250 square thumbnail WITHOUT stretching it.
Idea: crop the largest centered square first, then shrink it.
Input : exp1_input.jpg
Output: exp1_1_thumbnail.png
"""
from PIL import Image
import matplotlib.pyplot as plt


def generate_square_thumbnail(image_path, output_path="exp1_1_thumbnail.png"):
    img = Image.open(image_path)
    original_width, original_height = img.size

    # Step 1: the square's side is the shorter side of the image
    shortest_side = min(original_width, original_height)

    # Step 2: compute crop box so the square is centered
    # (left/top = half of the leftover space on each axis)
    left = (original_width - shortest_side) / 2
    top = (original_height - shortest_side) / 2
    right = (original_width + shortest_side) / 2
    bottom = (original_height + shortest_side) / 2

    img_cropped = img.crop((left, top, right, bottom))

    # Step 3: shrink to 250x250. LANCZOS is a high-quality downscaling filter.
    img_thumbnail = img_cropped.resize((250, 250), Image.Resampling.LANCZOS)

    img_thumbnail.save(output_path)

    # Show original and thumbnail side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n({original_width}x{original_height})")
    axes[0].axis('off')

    axes[1].imshow(img_thumbnail)
    axes[1].set_title("Thumbnail\n(250x250)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


generate_square_thumbnail("exp1_input.jpg")

"""
Experiment 7 - Image Segmentation: Edge-Based vs Region-Based
-------------------------------------------------------------
Goal: segment an image in two different ways and compare them.
  - Edge-based  : Canny finds the outlines of objects.
  - Region-based: Watershed floods "basins" starting from sure-object seeds
                  and draws boundaries where the floods meet.
Input : exp7_input.jpg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('exp7_input.jpg')
# Keep an untouched copy: the watershed step below paints on `image` in place
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ---------- Edge-based segmentation ----------
blurred = cv2.GaussianBlur(gray, (5, 5), 0)   # blur first to suppress noise edges
edges = cv2.Canny(blurred, 50, 150)           # low / high hysteresis thresholds

# ---------- Region-based segmentation (Watershed) ----------
# 1. Binarize with Otsu's automatic threshold (INV -> objects become white)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 2. Opening (erode then dilate) removes small white noise
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 3. Sure background: dilating the objects -> anything outside is definitely background
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# 4. Sure foreground: pixels far from any background pixel (peaks of the distance map)
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# 5. Unknown region = between sure background and sure foreground (watershed decides it)
unknown = cv2.subtract(sure_bg, sure_fg)

# 6. Label each sure-foreground blob with its own integer marker
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1            # shift so background is 1 (0 is reserved for "unknown")
markers[unknown == 255] = 0      # mark unknown pixels as 0

# 7. Run watershed; boundary pixels come back labeled -1
markers = cv2.watershed(image, markers)
image[markers == -1] = [255, 0, 0]   # paint boundaries (note: BGR -> this is blue in OpenCV)

# ---------- Display ----------
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Edge-Based (Canny)')
plt.subplot(133), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Region-Based (Watershed)')
plt.show()

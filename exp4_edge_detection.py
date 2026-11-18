"""
Experiment 4 - Edge Detection Operators
---------------------------------------
Goal: compare classic edge detectors on the same image:
Roberts, Prewitt, Sobel, Laplacian and Canny.
Edges = places where intensity changes sharply, found via derivatives.
Input : exp4_input.jpg
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('exp4_input.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       # edge operators work on intensity
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   # only for display

# --- Roberts: tiny 2x2 diagonal kernels (fast, but sensitive to noise) ---
roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
# CV_64F keeps negative gradient values (uint8 would clip them to 0)
roberts_out_x = cv2.filter2D(gray, cv2.CV_64F, roberts_x)
roberts_out_y = cv2.filter2D(gray, cv2.CV_64F, roberts_y)
roberts = cv2.magnitude(roberts_out_x, roberts_out_y)  # sqrt(gx^2 + gy^2)

# --- Prewitt: 3x3 kernels, plain averaging across the gradient direction ---
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)   # vertical edges
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)   # horizontal edges
prewitt_out_x = cv2.filter2D(gray, cv2.CV_64F, prewitt_x)
prewitt_out_y = cv2.filter2D(gray, cv2.CV_64F, prewitt_y)
prewitt = cv2.magnitude(prewitt_out_x, prewitt_out_y)

# --- Sobel: like Prewitt but weights the center row/column more (smoother) ---
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # d/dx
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # d/dy
sobel = cv2.magnitude(sobel_x, sobel_y)

# --- Laplacian: second derivative, direction-independent, very noise sensitive ---
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# --- Canny: multi-stage (blur, gradient, thin edges, hysteresis thresholds) ---
# 100 / 200 = low / high thresholds: >200 strong edge, <100 discarded,
# in between kept only if connected to a strong edge.
canny = cv2.Canny(gray, 100, 200)

# --- Display everything in a 2x4 grid (the 8th slot is left empty) ---
plt.figure(figsize=(15, 8))

plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title('Original')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(roberts, cmap='gray')
plt.title('Roberts')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian')
plt.axis('off')

plt.subplot(2, 4, 7)
plt.imshow(canny, cmap='gray')
plt.title('Canny')
plt.axis('off')

plt.tight_layout()
plt.show()

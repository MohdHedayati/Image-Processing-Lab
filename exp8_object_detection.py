"""
Experiment 8 - Object Detection with YOLOv8
-------------------------------------------
Goal: detect and label objects in an image using a pretrained YOLOv8 model.
YOLO ("You Only Look Once") predicts bounding boxes + class + confidence
in a single pass through a neural network.
Input : exp8_input.jpg
Model : yolov8n.pt (the "nano" model: smallest/fastest; auto-downloaded if missing)
"""
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load pretrained weights (trained on the COCO dataset, 80 everyday object classes)
model = YOLO("yolov8n.pt")
image = cv2.imread("exp8_input.jpg")

if image is None:
    print("Error: Could not read image.")
else:
    # Run inference; `results` has one entry per input image
    results = model(image)

    print("\nDetection Results")
    for result in results:
        for box in result.boxes:                 # one box per detected object
            class_id = int(box.cls[0])           # numeric class index
            confidence = float(box.conf[0])      # 0-1 certainty score
            label = model.names[class_id]        # index -> readable name (e.g. "dog")

            print(f"Object: {label.capitalize()}")
            print(f"Confidence: {confidence:.2f}\n")

    # .plot() returns a copy of the image with boxes and labels drawn (BGR format)
    annotated_image = results[0].plot()

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # BGR -> RGB for matplotlib
    plt.title("YOLOv8 Object Detection Output")
    plt.axis("off")
    plt.show()

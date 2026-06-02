import cv2
import numpy as np

img = cv2.imread("images/ok_page_0.png")

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold to find non-white areas (the UI is mostly white background)
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    # We expect camera feeds to be roughly square-ish and large
    if w > 200 and h > 200 and w < 1000 and h < 1000:
        boxes.append((x, y, w, h))

boxes = sorted(boxes, key=lambda b: b[0])
print("Found boxes:", boxes)

# Let's save an image with the boxes drawn to verify
out = img.copy()
for i, (x, y, w, h) in enumerate(boxes):
    cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.putText(out, f"Box {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
cv2.imwrite("test_boxes.png", out)

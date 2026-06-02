import cv2
import numpy as np

img = cv2.imread("images/ok_page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cv2.imwrite("edges.png", edges)

# We know the UI has fixed coordinates for the camera views.
# Let's print the image shape
print("Image shape:", img.shape)

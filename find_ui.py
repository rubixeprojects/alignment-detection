import cv2
import numpy as np

img = cv2.imread("images/ok_page_0.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# The margins are white (255). We want the bounding box of anything not white.
non_white = cv2.findNonZero((gray < 250).astype(np.uint8))
x, y, w, h = cv2.boundingRect(non_white)
print(f"UI bounding box: x={x}, y={y}, w={w}, h={h}")

# Now within this UI, the left camera is roughly the left half, right camera is the right half.
# The camera images themselves have a gray frame. We can just use fixed proportions of the UI width/height.
# Looking at the UI, the top ~10% is header, the bottom ~30% is result panels.
# Let's say the camera feeds are from y + 0.15*h to y + 0.70*h

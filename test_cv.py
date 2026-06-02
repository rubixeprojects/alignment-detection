import cv2
import numpy as np
import sys

img = cv2.imread("images/ok_page_0.png")
if img is None:
    print("Could not read image")
    sys.exit(1)

# In the UI, LH mode is roughly on the left, RH on the right.
# We can find the large dark regions which are the camera images.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Filter for large rectangular contours
camera_boxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w > 300 and h > 300: # large enough to be a camera feed
        camera_boxes.append((x, y, w, h))

# Sort left to right
camera_boxes.sort(key=lambda b: b[0])
print("Found camera boxes:", camera_boxes)

if len(camera_boxes) >= 2:
    lh_box = camera_boxes[0]
    rh_box = camera_boxes[1]
    
    # Crop LH
    lh_img = img[lh_box[1]:lh_box[1]+lh_box[3], lh_box[0]:lh_box[0]+lh_box[2]]
    cv2.imwrite("lh_crop.png", lh_img)
    
    # Try finding circles on LH
    lh_gray = cv2.cvtColor(lh_img, cv2.COLOR_BGR2GRAY)
    lh_blur = cv2.GaussianBlur(lh_gray, (9, 9), 2)
    circles = cv2.HoughCircles(lh_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=100, param2=50, minRadius=50, maxRadius=200)
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        print("Found circles in LH:", circles)
        # Draw them
        output = lh_img.copy()
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        cv2.imwrite("lh_circles.png", output)

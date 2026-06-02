import cv2
import numpy as np
import glob
import os

def process_roi(camera_roi):
    gray = cv2.cvtColor(camera_roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(equalized, (9, 9), 2)
    
    # We expect a large circle (hub) with radius between 80 and 200
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=100,
        param1=100, 
        param2=50, 
        minRadius=80, 
        maxRadius=250
    )
    
    if circles is None:
        return False, "NG (Center not found - possible glare)"
        
    circles = np.round(circles[0, :]).astype("int")
    center_x, center_y, radius = circles[0]
    
    # Annular mask for notch
    expected_notch_radius = radius - 20
    if expected_notch_radius <= 0:
        return False, "NG (Invalid radius)"
        
    ring_thickness = 15
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), expected_notch_radius + ring_thickness, 255, -1)
    cv2.circle(mask, (center_x, center_y), max(1, expected_notch_radius - ring_thickness), 0, -1)
    
    ring_roi = cv2.bitwise_and(equalized, equalized, mask=mask)
    _, thresh = cv2.threshold(ring_roi, 60, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, "NG (Notch not found)"
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 10:
        return False, "NG (Notch too small)"
        
    M = cv2.moments(largest_contour)
    if M["m00"] == 0:
        return False, "NG (Invalid notch)"
        
    notch_x = int(M["m10"] / M["m00"])
    notch_y = int(M["m01"] / M["m00"])
    
    dx = notch_x - center_x
    dy = center_y - notch_y
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
        
    target_angle = 90.0
    tolerance = 15.0 # allow some tolerance
    
    is_ok = abs(angle_deg - target_angle) <= tolerance
    status = "OK" if is_ok else f"NG (Angle: {angle_deg:.1f})"
    
    return is_ok, status

def evaluate_image(img_path):
    img = cv2.imread(img_path)
    if img is None: return None, None
    
    # UI bounding box: y=528, h=698
    # Crop left and right cameras
    # Assuming UI is standard
    y_start = 588
    y_end = 1008
    lh_x_start = 100
    lh_x_end = 580
    rh_x_start = 680
    rh_x_end = 1160
    
    lh_roi = img[y_start:y_end, lh_x_start:lh_x_end]
    rh_roi = img[y_start:y_end, rh_x_start:rh_x_end]
    
    if lh_roi.size == 0 or rh_roi.size == 0:
        return None, None
        
    lh_ok, lh_status = process_roi(lh_roi)
    rh_ok, rh_status = process_roi(rh_roi)
    
    return lh_ok and rh_ok, f"LH: {lh_status}, RH: {rh_status}"

ok_images = glob.glob("images/ok_*.png")[:20]
ng_images = glob.glob("images/ng_*.png")[:20]

print("Evaluating OK images (Expected: OK)")
ok_correct = 0
for path in ok_images:
    res, status = evaluate_image(path)
    if res: ok_correct += 1
    # print(f"{os.path.basename(path)} -> {res} ({status})")
print(f"OK Accuracy: {ok_correct}/{len(ok_images)}")

print("\nEvaluating NG images (Expected: NG)")
ng_correct = 0
for path in ng_images:
    res, status = evaluate_image(path)
    if not res: ng_correct += 1
    # print(f"{os.path.basename(path)} -> {not res} ({status})")
print(f"NG Accuracy: {ng_correct}/{len(ng_images)}")

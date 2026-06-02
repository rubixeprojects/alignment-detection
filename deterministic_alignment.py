import cv2
import numpy as np

def process_wheel_housing(image_path):
    """
    Deterministically processes a wheel housing image to find the center,
    mitigate glare, and locate the alignment notch.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Crop to the camera ROI (You will need to adjust these coordinates 
    # to extract just the camera feed from the UI screenshots)
    # camera_roi = img[y1:y2, x1:x2]
    camera_roi = img.copy() # Assuming we are working on a cropped image
    
    # 2. Pre-processing (Cleaning the image)
    gray = cv2.cvtColor(camera_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    # This is crucial for mitigating the heavy glare seen in the RH images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(equalized, (9, 9), 2)
    
    # 3. Deterministic Center Finding using Hough Circles
    # We look for the prominent inner/outer circular hub
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
    
    center_x, center_y = None, None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Assume the most prominent circle is the hub
        center_x, center_y, radius = circles[0]
        cv2.circle(camera_roi, (center_x, center_y), radius, (0, 255, 0), 2)
        cv2.circle(camera_roi, (center_x, center_y), 5, (0, 0, 255), -1)
    else:
        return False, "Could not detect center hub", camera_roi
        
    # 4. Deterministic Notch Finding
    # We know the notch lies on a specific radial distance from the center.
    # We can create an annular mask (a ring) at that radius to isolate the notch search area.
    expected_notch_radius = radius - 20 # Adjust based on actual physical dimensions
    ring_thickness = 15
    
    mask = np.zeros_like(gray)
    cv2.circle(mask, (center_x, center_y), expected_notch_radius + ring_thickness, 255, -1)
    cv2.circle(mask, (center_x, center_y), expected_notch_radius - ring_thickness, 0, -1)
    
    # Apply mask to the equalized image
    ring_roi = cv2.bitwise_and(equalized, equalized, mask=mask)
    
    # Threshold the ring to find the dark notch
    _, thresh = cv2.threshold(ring_roi, 60, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask) # Ensure we only look inside the ring
    
    # Find contours of the notch
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return False, "Could not detect notch", camera_roi
        
    # Assume the largest contour in the ring is the notch
    largest_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(largest_contour)
    
    if M["m00"] != 0:
        notch_x = int(M["m10"] / M["m00"])
        notch_y = int(M["m01"] / M["m00"])
    else:
        return False, "Invalid notch contour", camera_roi
        
    cv2.rectangle(camera_roi, (notch_x-10, notch_y-10), (notch_x+10, notch_y+10), (0, 255, 0), 2)
    
    # 5. Calculate Angle
    # Calculate angle using atan2. 
    # Note: in images, y increases downwards, so we negate dy.
    dx = notch_x - center_x
    dy = center_y - notch_y
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # Convert to a 0-360 scale where 90 is straight up
    if angle_deg < 0:
        angle_deg += 360
        
    # Draw alignment lines
    # Horizontal reference
    cv2.line(camera_roi, (0, center_y), (camera_roi.shape[1], center_y), (255, 0, 0), 2)
    # Notch vector
    cv2.line(camera_roi, (center_x, center_y), (notch_x, notch_y), (0, 255, 255), 2)
    
    # Check if angle is within acceptable tolerance of 90 degrees (vertical)
    target_angle = 90.0
    tolerance = 5.0
    
    is_ok = abs(angle_deg - target_angle) <= tolerance
    status = "OK" if is_ok else f"NG (Angle: {angle_deg:.2f})"
    
    return is_ok, status, camera_roi

if __name__ == "__main__":
    # Example usage:
    # is_ok, status, result_img = process_wheel_housing("path_to_cropped_camera_feed.png")
    # cv2.imwrite("output.png", result_img)
    pass

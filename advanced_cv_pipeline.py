import cv2
import numpy as np

def adjust_gamma(image, gamma=1.0):
    """
    Builds a lookup table mapping pixel values [0, 255] to their adjusted gamma values.
    Helps in reducing severe glare.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def advanced_process_wheel_housing(image_path):
    """
    Advanced Deterministic Approach using Gamma Correction, CLAHE, and Polar Transformation.
    """
    # 1. Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
        
    # Assume the image is a cropped ROI of just the camera feed
    camera_roi = img.copy()
    
    # 2. High-Glare Mitigation
    # Apply Gamma correction to compress bright highlights and recover contrast
    gamma_corrected = adjust_gamma(camera_roi, gamma=2.2)
    
    gray = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to locally enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    
    # Blur to remove high-frequency noise
    blurred = cv2.GaussianBlur(equalized, (9, 9), 2)
    
    # 3. Find Absolute Center
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
        return False, "NG (Center not found - severe glare or missing part)", camera_roi
        
    circles = np.round(circles[0, :]).astype("int")
    center_x, center_y, radius = circles[0]
    
    # Draw center
    cv2.circle(camera_roi, (center_x, center_y), radius, (0, 255, 0), 2)
    cv2.circle(camera_roi, (center_x, center_y), 5, (0, 0, 255), -1)
    
    # 4. Polar Transformation
    # We want to search for the notch at a specific radius
    expected_notch_radius = radius - 20 # Adjust this based on physical calibration
    search_width = 30 # How thick the ring is
    
    # Define polar transformation parameters
    center = (float(center_x), float(center_y))
    max_radius = float(expected_notch_radius + search_width)
    
    # Warp to polar coordinates. 
    # cv2.WARP_POLAR_LINEAR maps angle to Y axis and radius to X axis.
    polar_img = cv2.warpPolar(equalized, (int(max_radius), 360), center, max_radius, cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS)
    
    # We only care about the strip containing the notch.
    # Crop the X-axis (radius) to just the notch area
    strip_start_x = expected_notch_radius - (search_width // 2)
    strip_end_x = expected_notch_radius + (search_width // 2)
    
    if strip_start_x < 0 or strip_end_x > polar_img.shape[1]:
        return False, "NG (Invalid radius calculated)", camera_roi
        
    notch_strip = polar_img[:, int(strip_start_x):int(strip_end_x)]
    
    # 5. 1D Projection to find the notch
    # Average the pixels along the X axis to get a 1D array representing brightness at each angle (0 to 359 degrees)
    # The notch will be the darkest spot (lowest value)
    brightness_profile = np.mean(notch_strip, axis=1)
    
    # Find the angle with the minimum brightness
    # (cv2.warpPolar maps angle 0 to Y=0, angle 359 to Y=359, but the starting 0 degree is on the right horizontally)
    notch_angle_raw = np.argmin(brightness_profile)
    
    # 6. Calculate True Angle
    # In OpenCV's warpPolar, Y = 0 is 0 degrees (pointing right / 3 o'clock).
    # Y = 90 is 90 degrees (pointing down / 6 o'clock).
    # Y = 270 is 270 degrees (pointing up / 12 o'clock).
    # Assuming "Vertical" alignment means 12 o'clock, the target Y is 270.
    
    target_angle = 270.0 
    tolerance = 5.0
    
    # To handle wrap-around (e.g., 359 and 0 are close)
    angle_diff = min(abs(notch_angle_raw - target_angle), 360 - abs(notch_angle_raw - target_angle))
    
    is_ok = angle_diff <= tolerance
    
    # Convert polar angle back to cartesian to draw the line on the output image
    # Note: OpenCV angles go clockwise
    notch_rad = np.radians(notch_angle_raw)
    notch_x = int(center_x + expected_notch_radius * np.cos(notch_rad))
    notch_y = int(center_y + expected_notch_radius * np.sin(notch_rad))
    
    # Draw reference line (Horizontal)
    cv2.line(camera_roi, (0, center_y), (camera_roi.shape[1], center_y), (255, 0, 0), 2)
    
    # Draw detected notch vector
    color = (0, 255, 0) if is_ok else (0, 0, 255)
    cv2.line(camera_roi, (center_x, center_y), (notch_x, notch_y), color, 2)
    cv2.circle(camera_roi, (notch_x, notch_y), 10, color, 2)
    
    # Convert target 270 to 90 format for output if needed
    display_angle = 360 - notch_angle_raw if notch_angle_raw > 180 else notch_angle_raw
    status = "OK" if is_ok else f"NG (Angle deviation: {angle_diff:.1f} deg)"
    
    return is_ok, status, camera_roi

if __name__ == "__main__":
    # Example Usage:
    # is_ok, status, img = advanced_process_wheel_housing("path_to_cropped_image.png")
    # cv2.imwrite("output_advanced.png", img)
    pass

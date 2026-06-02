"""
Machine Learning Training Pipeline for Wheel Housing Alignment
Using YOLOv8-Pose (Keypoint Detection)

Prerequisites:
pip install ultralytics

Dataset Structure Requirement:
Your dataset must be formatted in the standard YOLO pose format.
Directory structure:
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       ├── img3.jpg
│       └── img4.jpg
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── img2.txt
    └── val/
        ├── img3.txt
        └── img4.txt

Label format (img1.txt):
<class_index> <x_center> <y_center> <width> <height> <px1> <py1> <p1_visible> <px2> <py2> <p2_visible>
Example (Class 0, bounding box, then Center Keypoint and Notch Keypoint):
0 0.5 0.5 0.8 0.8 0.5 0.5 2 0.5 0.2 2
"""

from ultralytics import YOLO
import os

def train_yolo_pose_model():
    """
    Trains a YOLOv8 Nano Pose model to detect the center and notch keypoints.
    """
    
    # Create the data configuration file (yaml) dynamically
    yaml_content = """
path: ./dataset  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: wheel_housing

# Keypoints
kpt_shape: [2, 3]  # [number of keypoints, number of dims (x, y, visible)]
"""
    
    with open("wheel_alignment_pose.yaml", "w") as f:
        f.write(yaml_content)
        
    print("Created dataset configuration: wheel_alignment_pose.yaml")
    
    # 1. Load a pre-trained YOLOv8 Nano pose model
    # (Using the nano model 'yolov8n-pose.pt' because it's the fastest and easily runs on CPUs)
    print("Loading YOLOv8n-pose model...")
    model = YOLO('yolov8n-pose.pt') 
    
    # 2. Train the model
    # We train for 100 epochs, but you can increase this based on your dataset size.
    print("Starting training...")
    results = model.train(
        data='wheel_alignment_pose.yaml',
        epochs=100,
        imgsz=640,          # Resize images to 640x640 for training
        batch=16,           # Batch size (adjust based on your GPU/CPU RAM)
        device='',          # Leave empty for auto-detect (will use GPU if available, else CPU)
        name='wheel_alignment_run'
    )
    
    print("Training complete. Model saved in runs/pose/wheel_alignment_run/weights/best.pt")

def run_inference(image_path, model_path="runs/pose/wheel_alignment_run/weights/best.pt"):
    """
    Runs inference using the trained model and calculates the alignment angle.
    """
    import math
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    model = YOLO(model_path)
    results = model(image_path)
    
    for r in results:
        # Check if keypoints were detected
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            print("No keypoints detected.")
            continue
            
        # Get the keypoints for the first detected object [0]
        # xy is a tensor of shape (num_objects, num_keypoints, 2)
        keypoints = r.keypoints.xy[0].cpu().numpy() 
        
        if len(keypoints) >= 2:
            center_x, center_y = keypoints[0]
            notch_x, notch_y = keypoints[1]
            
            print(f"Detected Center: ({center_x:.1f}, {center_y:.1f})")
            print(f"Detected Notch:  ({notch_x:.1f}, {notch_y:.1f})")
            
            # Calculate angle (atan2)
            dx = notch_x - center_x
            dy = center_y - notch_y  # Invert Y because image Y increases downwards
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
                
            print(f"Alignment Angle: {angle_deg:.2f} degrees")
            
            # Assuming 90 degrees (pointing straight up) is OK
            is_ok = abs(angle_deg - 90.0) <= 5.0
            print(f"Status: {'OK' if is_ok else 'NG'}")

if __name__ == "__main__":
    print("Uncomment train_yolo_pose_model() to start training once your dataset is ready.")
    # train_yolo_pose_model()
    
    # Example inference:
    # run_inference("path_to_test_image.png")

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

def evaluate_model(model_path="runs/pose/wheel_alignment_run/weights/best.pt"):
    """
    Evaluates the trained model on the validation dataset to test the training output instances.
    This provides metrics like mAP (mean Average Precision) for both the bounding box and keypoints.
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return

    print(f"\\n--- Evaluating model: {model_path} ---")
    model = YOLO(model_path)
    
    # Run validation on the test/val split defined in the yaml
    metrics = model.val(data='wheel_alignment_pose.yaml')
    
    print("\\n--- Evaluation Results ---")
    print(f"Box mAP50-95:  {metrics.box.map:.4f} (Accuracy of finding the wheel housing)")
    print(f"Pose mAP50-95: {metrics.pose.map:.4f} (Accuracy of pinpointing the center and notch)")
    print("--------------------------\\n")

def run_inference(image_path, model_path="runs/pose/wheel_alignment_run/weights/best.pt", conf_threshold=0.6):
    """
    Runs inference using the trained model and calculates the alignment angle.
    Includes confidence checking to provide specific reasons for NG results.
    """
    import math
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    model = YOLO(model_path)
    results = model(image_path)
    
    print(f"\\n--- Inference Report for {image_path} ---")
    
    for r in results:
        # 1. Check if an object was even detected
        if r.boxes is None or len(r.boxes) == 0:
            print("Status: NG")
            print("Reason: No wheel housing detected. The image might be completely washed out by glare or the part is missing.")
            continue
            
        box_conf = float(r.boxes.conf[0].cpu().numpy())
        
        # 2. Check bounding box confidence
        if box_conf < conf_threshold:
            print("Status: NG")
            print(f"Reason: Low overall detection confidence ({box_conf:.2f} < {conf_threshold}). The image quality is likely too poor (e.g., severe glare).")
            continue

        # 3. Check keypoints existence
        if r.keypoints is None or len(r.keypoints.xy) == 0:
            print("Status: NG")
            print("Reason: Wheel housing detected, but keypoints (Center/Notch) could not be located.")
            continue
            
        keypoints = r.keypoints.xy[0].cpu().numpy() 
        
        # YOLOv8 pose models return keypoint confidences in r.keypoints.conf
        kpt_confs = r.keypoints.conf[0].cpu().numpy() if r.keypoints.conf is not None else [1.0, 1.0]
        
        if len(keypoints) >= 2:
            center_x, center_y = keypoints[0]
            notch_x, notch_y = keypoints[1]
            center_conf = float(kpt_confs[0])
            notch_conf = float(kpt_confs[1])
            
            print(f"Detected Center: ({center_x:.1f}, {center_y:.1f}) - Confidence: {center_conf:.2f}")
            print(f"Detected Notch:  ({notch_x:.1f}, {notch_y:.1f}) - Confidence: {notch_conf:.2f}")
            
            # 4. Check keypoint confidence
            if center_conf < conf_threshold or notch_conf < conf_threshold:
                print("Status: NG")
                print(f"Reason: Low confidence in exact alignment points (Center: {center_conf:.2f}, Notch: {notch_conf:.2f}). "
                      "The notch might be obscured by glare, dirt, or out of focus.")
                continue
            
            # 5. Calculate angle (atan2)
            dx = notch_x - center_x
            dy = center_y - notch_y  # Invert Y because image Y increases downwards
            
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)
            if angle_deg < 0:
                angle_deg += 360
                
            print(f"Alignment Angle: {angle_deg:.2f} degrees")
            
            # Assuming 90 degrees (pointing straight up) is OK
            is_ok = abs(angle_deg - 90.0) <= 5.0
            
            if is_ok:
                print("Status: OK")
                print("Reason: Notch is properly aligned within the 90 ± 5 degree tolerance.")
            else:
                print("Status: NG")
                print(f"Reason: Angular misalignment. Detected angle {angle_deg:.1f} is outside the acceptable tolerance.")

if __name__ == "__main__":
    print("Uncomment train_yolo_pose_model() to start training once your dataset is ready.")
    # train_yolo_pose_model()
    
    # To test the model on your validation dataset:
    # evaluate_model()
    
    # Example inference on a single image:
    # run_inference("path_to_test_image.png")

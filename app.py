import os
import io
import base64
import math
import cv2
import numpy as np
import threading
import time
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from ultralytics import YOLO

class CameraStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            raise ValueError(f"Failed to open video source: {src}")
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                time.sleep(0.05)
                continue
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        with self.read_lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return self.grabbed, None

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.stream.release()

global_camera_stream = None
latest_verdict = {
    "status": "IDLE",
    "angle": None,
    "message": "Camera not connected",
    "center_conf": 0.0,
    "notch_conf": 0.0,
    "detections": []
}

app = Flask(__name__, static_folder="static", template_folder="templates")

# Path to the trained YOLOv8 model weights
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "wheel-center-notch-detection/TRAINING YOLO v8 results/weights/best.pt")
if not os.path.exists(MODEL_PATH):
    # Fallback to current directory best.pt
    MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

print(f"Loading YOLOv8s model from {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Path to test samples
TEST_IMAGES_DIR = os.path.join(BASE_DIR, "wheel-center-notch-detection/test/images")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/samples")
def get_samples():
    if not os.path.exists(TEST_IMAGES_DIR):
        return jsonify([])
    files = sorted([f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return jsonify(files)

@app.route("/api/sample/<filename>")
def get_sample_image(filename):
    # Secure filename
    filename = os.path.basename(filename)
    return send_from_directory(TEST_IMAGES_DIR, filename)

def process_and_analyze(img):
    """
    Runs YOLOv8 model inference on the CV2 image, finds the center and notch,
    computes the alignment angle, draws visualization overlays, and encodes
    the output image to base64.
    """
    # 1. Run inference
    # Save image to a temp location because ultralytics YOLO model takes paths/numpy arrays
    results = model(img, conf=0.25)
    result = results[0]
    
    # Detections list to return
    detections = []
    
    center_coords = None
    notch_coords = None
    center_conf = 0.0
    notch_conf = 0.0
    
    # 2. Extract bounding boxes
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        
        cx = int((xyxy[0] + xyxy[2]) / 2.0)
        cy = int((xyxy[1] + xyxy[3]) / 2.0)
        
        detections.append({
            "class": "center" if cls_id == 0 else "notch",
            "confidence": conf,
            "bbox": xyxy,
            "center": [cx, cy]
        })
        
        # We look for the highest confidence detections for each class
        if cls_id == 0:
            if center_coords is None or conf > center_conf:
                center_coords = (cx, cy)
                center_conf = conf
        elif cls_id == 1:
            if notch_coords is None or conf > notch_conf:
                notch_coords = (cx, cy)
                notch_conf = conf

    # 3. Draw annotations on copy of image
    annotated_img = img.copy()
    
    # Draw all detected boxes first
    for d in detections:
        xyxy = d["bbox"]
        cls_name = d["class"]
        conf = d["confidence"]
        
        color = (46, 204, 113) if cls_name == "center" else (52, 152, 219) # Green-ish for center, Blue-ish for notch (BGR format)
        cv2.rectangle(annotated_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
        cv2.putText(annotated_img, f"{cls_name} ({conf:.2f})", (int(xyxy[0]), int(xyxy[1])-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    status = "NG"
    angle = None
    message = "Notch or Center not detected"
    
    if center_coords and notch_coords:
        cx, cy = center_coords
        nx, ny = notch_coords
        
        # Draw center point markers
        cv2.circle(annotated_img, (cx, cy), 5, (46, 204, 113), -1)
        cv2.circle(annotated_img, (nx, ny), 5, (52, 152, 219), -1)
        
        # Draw vector line connecting center to notch
        cv2.line(annotated_img, (cx, cy), (nx, ny), (241, 196, 15), 2) # Yellow line
        
        # Calculate alignment angle
        dx = nx - cx
        dy = cy - ny # Invert Y because image Y increases downwards
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
            
        angle = round(angle_deg, 2)
        
        # Check if angle is within 90 +/- 5 degrees tolerance
        target_angle = 90.0
        tolerance = 5.0
        is_ok = abs(angle - target_angle) <= tolerance
        
        if is_ok:
            status = "OK"
            message = f"Aligned at {angle:.1f} degrees (Within tolerance)"
            status_color = (46, 204, 113) # Green
        else:
            status = "NG"
            message = f"Misaligned at {angle:.1f} degrees"
            status_color = (231, 76, 60) # Red
            
        # Draw overlay status text
        cv2.putText(annotated_img, f"Status: {status}", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)
        cv2.putText(annotated_img, f"Angle: {angle:.2f} deg", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (241, 196, 15), 2, cv2.LINE_AA)
    else:
        # Drawing fail status
        cv2.putText(annotated_img, "Status: NG (Detection Failed)", (15, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (231, 76, 60), 2, cv2.LINE_AA)

    # 4. Encode result image to base64
    _, buffer = cv2.imencode('.png', annotated_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": status,
        "angle": angle,
        "message": message,
        "center_conf": round(center_conf, 2),
        "notch_conf": round(notch_conf, 2),
        "detections": detections,
        "image_data": f"data:image/png;base64,{encoded_img}",
        "annotated_frame": annotated_img
    }

@app.route("/api/predict", methods=["POST"])
def predict():
    # Check if a file was uploaded
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
            
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image file"}), 400
            
        result = process_and_analyze(img)
        result.pop("annotated_frame", None)
        return jsonify(result)
        
    # Check if a sample image is specified in JSON
    data = request.get_json() or {}
    sample_name = data.get("sample_name")
    if sample_name:
        sample_path = os.path.join(TEST_IMAGES_DIR, os.path.basename(sample_name))
        if not os.path.exists(sample_path):
            return jsonify({"error": "Sample image not found"}), 404
            
        img = cv2.imread(sample_path)
        if img is None:
            return jsonify({"error": "Could not read sample image"}), 500
            
        # Encode original image to base64 to display it
        _, buffer = cv2.imencode('.png', img)
        orig_encoded = base64.b64encode(buffer).decode('utf-8')
        
        result = process_and_analyze(img)
        result["original_image"] = f"data:image/png;base64,{orig_encoded}"
        result.pop("annotated_frame", None)
        return jsonify(result)
        
    return jsonify({"error": "No file uploaded or sample name provided"}), 400

@app.route("/api/camera/connect", methods=["POST"])
def connect_camera():
    global global_camera_stream, latest_verdict
    data = request.get_json() or {}
    url = data.get("url")
    
    # Check if url is digit (e.g. index for webcam like '0' or '1')
    if isinstance(url, str) and url.isdigit():
        url = int(url)
    elif not url:
        url = 0
        
    try:
        if global_camera_stream is not None:
            try:
                global_camera_stream.stop()
            except Exception:
                pass
            global_camera_stream = None
            
        print(f"Connecting to camera: {url} ...")
        global_camera_stream = CameraStream(url).start()
        latest_verdict = {
            "status": "WAITING",
            "angle": None,
            "message": "Connected. Awaiting frames...",
            "center_conf": 0.0,
            "notch_conf": 0.0,
            "detections": []
        }
        return jsonify({"success": True, "message": "Camera connected successfully"})
    except Exception as e:
        print(f"Connection error: {e}")
        return jsonify({"success": False, "message": f"Connection failed: {str(e)}"}), 500

@app.route("/api/camera/disconnect", methods=["POST"])
def disconnect_camera():
    global global_camera_stream, latest_verdict
    if global_camera_stream is not None:
        try:
            global_camera_stream.stop()
        except Exception:
            pass
        global_camera_stream = None
    latest_verdict = {
        "status": "IDLE",
        "angle": None,
        "message": "Camera disconnected",
        "center_conf": 0.0,
        "notch_conf": 0.0,
        "detections": []
    }
    return jsonify({"success": True, "message": "Camera disconnected"})

def gen_frames():
    global global_camera_stream, latest_verdict
    while True:
        if global_camera_stream is None:
            time.sleep(0.1)
            continue
            
        grabbed, frame = global_camera_stream.read()
        if not grabbed or frame is None:
            time.sleep(0.03)
            continue
            
        try:
            # Process frame using YOLO model
            result = process_and_analyze(frame)
            latest_verdict = {
                "status": "OK" if result["status"] == "OK" else "NG",
                "angle": result["angle"],
                "message": result["message"],
                "center_conf": result["center_conf"],
                "notch_conf": result["notch_conf"],
                "detections": result["detections"]
            }
            
            # Retrieve annotated frame and encode to jpg for MJPEG stream
            annotated_frame = result.get("annotated_frame")
            if annotated_frame is not None:
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"Frame processing error: {e}")
            time.sleep(0.03)

@app.route("/api/camera/stream")
def camera_stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/api/camera/status")
def camera_status():
    global latest_verdict
    return jsonify(latest_verdict)

if __name__ == "__main__":
    # Start app on local port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)

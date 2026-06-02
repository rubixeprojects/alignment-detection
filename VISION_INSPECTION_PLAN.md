# Goal Description

The objective is to overhaul the current vision inspection system for the wheel housing notch alignment. The new approach must robustly handle high-glare/overexposed images, provide a highly deterministic algorithmic fallback, and transition to a Machine Learning (ML) model for state-of-the-art accuracy.

## User Review Required

> [!IMPORTANT]
> **Data Leakage Warning**: To train an ML model, we *must* have access to the raw camera feed images **without** the UI's green and red crosshairs drawn over them. If we train the model on images that already have crosshairs, the model will just learn to "read" the crosshairs instead of looking at the actual metal part. Can you acquire a dataset of raw, unannotated camera images?

> [!WARNING]
> **Hardware Constraints**: Running an ML model in a factory environment requires specific hardware. Is the target deployment machine a standard CPU, an Edge device (like a Raspberry Pi / Jetson Nano), or does it have a dedicated GPU? This will dictate how large of a model we can train.

## 1. Advanced Deterministic Implementation (Computer Vision)

To improve upon the current deterministic method and specifically target the high-glare (white-out) images, we will implement a multi-stage preprocessing pipeline before attempting to calculate the angle.

### High-Glare Mitigation Pipeline
1. **Gamma Correction**: Glare washes out contrast in the highlights. Applying a non-linear gamma correction (e.g., gamma = 2.0) will compress the bright pixels and expand the dark pixels, revealing details hidden inside the glare.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Unlike standard thresholding, CLAHE divides the image into a grid (e.g., 8x8 tiles) and enhances contrast locally. This prevents the glaring regions from washing out the darker notch region.
3. **Morphological Black-Hat Transform**: A black-hat transform is excellent at finding dark defects (like a notch) on a bright, uneven background (like a glaring metal ring).

### Polar Transformation for Reliable Notch Detection
Instead of searching a circular ring in 2D space, we will use `cv2.linearPolar`.
- **The Concept**: We "unroll" the circular hub into a flat, rectangular strip. 
- **The Benefit**: In a flat strip, the notch simply becomes a dark vertical bar. We can average the pixels vertically (a 1D projection) and simply look for the deepest "valley" in the graph. This method is mathematically immune to scattered noise and glare, making it vastly superior to 2D blob detection.

## 2. Machine Learning Pipeline (Keypoint Detection)

While CV is good, ML is exceptional at handling lighting variations. Instead of training a simple image classifier (OK vs NG), the most robust method is to train a **Keypoint Detection Model** (specifically, **YOLOv8-Pose**).

### Why Keypoint Detection?
An OK/NG classifier doesn't explain *why* a part failed. A keypoint model will predict the exact `(x, y)` pixel coordinates of the features we care about, allowing us to mathematically calculate the angle ourselves.

### Phase 1: Data Preparation
1. **Dataset Collection**: Gather ~500 raw, unannotated images (a mix of OK, NG, and high-glare).
2. **Annotation**: Use a tool like CVAT or Roboflow. For every image, manually click two points:
   - **Keypoint 1**: The exact absolute center of the hub.
   - **Keypoint 2**: The exact center of the notch.
3. **Augmentation**: Artificially generate more data by applying random rotations, adding synthetic glare, and adjusting brightness. This forces the model to learn the physical structure of the notch, not just its lighting.

### Phase 2: Model Training
1. **Model Selection**: YOLOv8n-Pose (Nano) is extremely fast and can run in real-time (30+ FPS) on standard factory CPUs.
2. **Training**: Train the model to minimize the Euclidean distance between the predicted keypoints and your manual annotations. Because CNNs look at local textures and global context, the model will successfully "guess" where the center and notch are even if they are partially obscured by glare.

### Phase 3: Inference & Decision Logic
1. The camera captures a raw frame and passes it to the YOLOv8 model.
2. The model outputs two coordinates: `Center(x, y)` and `Notch(x, y)`.
3. The system calculates the angle using trigonometry.
4. **Deterministic Decision**: If the angle is exactly 90 degrees (with a 5 degree tolerance), the system outputs **OK**. Otherwise, **NG**.

## Verification Plan

### Automated Tests
- Build a Python evaluation script that runs the advanced CV Polar Transform method on the extracted raw images.
- Train a tiny proof-of-concept YOLOv8-pose model on a small subset of the data to verify the loss decreases and predictions are stable.

### Manual Verification
- Review edge-cases (the worst glare images) by manually verifying the drawn bounding boxes and angles predicted by both the deterministic script and the ML model.

# HomeScan

## Installation Guide

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)
- Git

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/a-laz/homescan_v1.git
   cd homescan_v1
   ```

2. **Set Up Virtual Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   ```bash
   # Create .env file
   touch .env
   
   # Add required environment variables to .env:
   # DATABASE_URL=your_database_url
   # SECRET_KEY=your_secret_key
   # Add any other required environment variables
   ```

5. **Initialize Database**
   ```bash
   python manage.py migrate
   ```

6. **Create Admin User** (Optional)
   ```bash
   python manage.py createsuperuser
   ```

7. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

### Troubleshooting

- If you encounter any package installation errors, try:
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- For database connection issues:
  - Verify your database credentials in .env
  - Ensure your database service is running

### System Requirements

- Memory: 4GB RAM minimum
- Storage: 1GB free space
- OS: macOS, Linux, or Windows


Next Step:
Here's a comprehensive approach to improve object detection and measurement accuracy by combining multiple models:

```python
class EnhancedFurnitureDetector:
    def __init__(self):
        # Primary object detection models
        self.detr = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5x')  # or larger model
        
        # Depth estimation
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "MiDaS")  # or DPT_Large
        
        # Instance segmentation for better boundaries
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        
        # 3D reconstruction (optional)
        self.nerf_model = instant_ngp.NGP()  # or other NeRF implementation
        
        # Camera parameters estimation
        self.camera_calibrator = CameraCalibrator()
        
        # Pose estimation for better angle handling
        self.pose_estimator = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
```

Here's how these models work together:

1. **Object Detection Ensemble**:
```python
    def detect_objects(self, image):
        """Combine DETR and YOLO predictions"""
        # DETR detection
        detr_results = self.detr(image)
        
        # YOLO detection
        yolo_results = self.yolo(image)
        
        # Ensemble the results using weighted box fusion
        combined_boxes = weighted_box_fusion(
            [detr_results['boxes'], yolo_results['boxes']],
            [detr_results['scores'], yolo_results['scores']],
            [detr_results['labels'], yolo_results['labels']],
            weights=[0.6, 0.4]
        )
        
        return combined_boxes
```

2. **Depth Estimation**:
```python
    def estimate_depth(self, image):
        """Get accurate depth map using MiDaS"""
        depth_input = self.depth_transform(image).to(self.device)
        with torch.no_grad():
            depth_map = self.depth_model(depth_input)
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        return depth_map
```

3. **Instance Segmentation**:
```python
    def get_precise_boundaries(self, image, boxes):
        """Get precise object boundaries using Mask R-CNN"""
        masks = self.mask_rcnn(image)['masks']
        refined_boxes = []
        
        for box, mask in zip(boxes, masks):
            # Refine box boundaries using mask
            y_indices, x_indices = torch.where(mask > 0.5)
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            refined_boxes.append([x_min, y_min, x_max, y_max])
            
        return refined_boxes
```

4. **3D Reconstruction** (optional):
```python
    def reconstruct_3d(self, image_sequence):
        """Create 3D model for complex measurements"""
        # Initialize NeRF
        self.nerf_model.train(image_sequence)
        
        # Generate 3D representation
        volume = self.nerf_model.extract_mesh()
        
        return volume
```

5. **Improved Measurement Pipeline**:
```python
    def calculate_dimensions(self, image, depth_map=None):
        # Get initial detections
        boxes = self.detect_objects(image)
        
        # Refine boundaries
        precise_boxes = self.get_precise_boundaries(image, boxes)
        
        # Get depth information
        if depth_map is None:
            depth_map = self.estimate_depth(image)
        
        # Get camera parameters
        camera_matrix = self.camera_calibrator.calibrate_from_image(image)
        
        # Calculate dimensions using all available information
        dimensions = []
        for box in precise_boxes:
            # Get depth at object location
            object_depth = depth_map[box[1]:box[3], box[0]:box[2]].mean()
            
            # Calculate real-world dimensions using camera parameters
            real_dims = self.convert_to_real_dimensions(
                box, object_depth, camera_matrix
            )
            
            dimensions.append(real_dims)
            
        return dimensions
```

6. **Multi-View Support**:
```python
    def measure_from_multiple_views(self, images):
        """Get more accurate measurements using multiple views"""
        all_measurements = []
        reconstructed_3d = None
        
        # If enough views, try 3D reconstruction
        if len(images) >= 3:
            reconstructed_3d = self.reconstruct_3d(images)
        
        # Get measurements from each view
        for image in images:
            measurements = self.calculate_dimensions(image)
            all_measurements.append(measurements)
        
        # Combine measurements using statistical methods
        final_measurements = self.combine_measurements(
            all_measurements, reconstructed_3d
        )
        
        return final_measurements
```

To use this enhanced system:

```python
detector = EnhancedFurnitureDetector()

# Single view measurement
image = cv2.imread('furniture.jpg')
dimensions = detector.calculate_dimensions(image)

# Multi-view measurement (more accurate)
images = [cv2.imread(f) for f in ['view1.jpg', 'view2.jpg', 'view3.jpg']]
dimensions = detector.measure_from_multiple_views(images)
```

Key benefits:
1. Multiple object detection models reduce false positives
2. Depth estimation provides better distance measurements
3. Instance segmentation gives more precise boundaries
4. 3D reconstruction allows for complex measurements
5. Multi-view support increases accuracy
6. Camera calibration improves measurement precision

Required packages:
```python
requirements = [
    'torch',
    'torchvision',
    'opencv-python',
    'transformers',
    'ultralytics',  # for YOLOv5
    'timm',
    'instant-ngp',  # for NeRF
    'numpy',
    'scipy'
]
```

This combination of models provides redundancy and cross-validation, leading to more accurate measurements. The system can fall back to simpler methods when certain components aren't available or when processing speed is a priority.
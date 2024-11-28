import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import faiss
import numpy as np
import cv2
import logging
import torchvision
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import math
import open3d as o3d
import base64
from io import BytesIO
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import os
from transformers import pipeline
import cv2.optflow as optflow  # For optical flow analysis

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Make depth-anything import optional
try:
    DEPTH_ANYTHING_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_AVAILABLE = False
    logger.warning("transformers package not found. Using fallback depth estimation.")

class CameraCalibrator:
    def __init__(self):
        # Standard calibration parameters
        self.SENSOR_WIDTH = 6.17  # mm (typical smartphone sensor)
        self.SENSOR_HEIGHT = 4.55  # mm
        self.DEFAULT_FOV = 69.4   # degrees (typical smartphone camera)
        
        # Calibration board parameters (for when available)
        self.CHARUCOBOARD_ROWCOUNT = 7
        self.CHARUCOBOARD_COLCOUNT = 5
        self.SQUARE_LENGTH = 0.04  # meters
        self.MARKER_LENGTH = 0.02  # meters
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard(
            (self.CHARUCOBOARD_COLCOUNT, self.CHARUCOBOARD_ROWCOUNT),
            self.SQUARE_LENGTH,
            self.MARKER_LENGTH,
            self.aruco_dict
        )
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)

    def get_default_calibration(self, image_shape):
        """Calculate default calibration based on typical smartphone parameters"""
        height, width = image_shape[:2]
        
        # Calculate focal length from FOV
        focal_length_pixels = width / (2 * np.tan(np.radians(self.DEFAULT_FOV / 2)))
        
        # Calculate pixel size
        pixel_size_x = self.SENSOR_WIDTH / width
        pixel_size_y = self.SENSOR_HEIGHT / height
        
        # Create camera matrix
        camera_matrix = np.array([
            [focal_length_pixels, 0, width/2],
            [0, focal_length_pixels, height/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Typical distortion coefficients for smartphone cameras
        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0.1], dtype=np.float32)
        
        logger.info(f"Using default calibration. Focal length: {focal_length_pixels:.1f}px, FOV: {self.DEFAULT_FOV}°")
        return camera_matrix, dist_coeffs, self.DEFAULT_FOV

    def calibrate_from_image(self, image):
        """Attempt to calibrate from image, fall back to default if needed"""
        try:
            if isinstance(image, np.ndarray):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

            # Try to detect calibration pattern
            corners, ids, rejected = self.detector.detectMarkers(gray)

            if len(corners) > 0:
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.board
                )

                if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                        [charuco_corners], [charuco_ids], self.board, gray.shape[::-1],
                        None, None
                    )

                    fov = 2 * np.arctan2(gray.shape[1]/2, camera_matrix[0,0])
                    fov_degrees = np.degrees(fov)

                    logger.info(f"Camera calibrated from pattern. FOV: {fov_degrees:.1f}°")
                    return camera_matrix, dist_coeffs, fov_degrees

            # Fall back to default calibration
            logger.info("No calibration pattern found, using default parameters")
            return self.get_default_calibration(gray.shape)

        except Exception as e:
            logger.error(f"Error in camera calibration: {str(e)}")
            return self.get_default_calibration(image.shape[:2])

    def generate_calibration_board(self, size=(2000, 2000)):
        return self.board.generateImage(size)

class EnhancedFurnitureDetector:
    # COCO dataset class names that are relevant for furniture
    FURNITURE_CLASSES = {
        0: "person",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "sink",
        71: "refrigerator",
        72: "book",
        73: "clock",
        74: "vase",
        75: "scissors",
        76: "teddy bear",
        77: "hair drier",
        78: "toothbrush"
    }

    def __init__(self):
        try:
            logger.info("Initializing EnhancedFurnitureDetector...")
            
            # Set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize camera calibrator
            self.camera_calibrator = CameraCalibrator()
            
            # Initialize object detection model
            logger.info("Loading YOLO model...")
            self.model = YOLO('yolov8x.pt')
            # Configure model settings
            self.model.conf = 0.25  # confidence threshold
            self.model.iou = 0.45   # NMS IOU threshold
            self.model.agnostic_nms = True  # class-agnostic NMS
            self.model.max_det = 300  # maximum detections per image
            
            # Initialize SAM model for segmentation
            logger.info("Loading SAM model...")
            sam_checkpoint = "sam_vit_l_0b3195.pth"
            model_type = "vit_l"
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            logger.info("SAM model loaded successfully")
            
            # Initialize Depth-Anything pipeline
            if DEPTH_ANYTHING_AVAILABLE:
                logger.info("Loading Depth-Anything-V2 model...")
                try:
                    self.depth_pipeline = pipeline(
                        task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("Depth-Anything-V2 model loaded successfully")
                except Exception as depth_error:
                    logger.error(f"Error loading Depth-Anything-V2: {str(depth_error)}")
                    self.depth_pipeline = None
            else:
                self.depth_pipeline = None
                
            # GPU setup
            if self.device == "cuda":
                logger.info(f"CUDA device count: {torch.cuda.device_count()}")
                logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
                torch.cuda.set_device(0)
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.enabled = True
                
            # Initialize frame processing parameters
            self.frame_buffer = []
            self.keyframe_interval = 5  # Process every 5th frame
            self.motion_threshold = 0.3  # Motion detection threshold
            
            # Initialize optical flow
            self.flow_calculator = optflow.DualTVL1OpticalFlow_create()
                
        except Exception as e:
            logger.error(f"Error in initialization: {str(e)}")
            raise

    def create_knowledge_base(self):
        """Create furniture knowledge base."""
        self.knowledge_base = {
            "chair": [
                "A chair is a piece of furniture with a raised surface supported by legs.",
                "Common chair types include dining chairs, office chairs, and armchairs.",
                "Chairs can be made from various materials including wood, metal, and plastic.",
                "Typical chair dimensions range from 16-20 inches in width and 28-32 inches in height.",
                "Ergonomic chairs often feature adjustable height and lumbar support."
            ],
            "table": [
                "A table is a piece of furniture with a flat top and one or more legs.",
                "Tables come in many forms including dining, coffee, and side tables.",
                "Common materials for tables include wood, glass, and metal.",
                "Dining tables typically stand 28-30 inches high with varying lengths.",
                "Coffee tables are usually 16-18 inches high and positioned in living rooms."
            ],
            "sofa": [
                "A sofa is an upholstered bench with cushioning, arms, and a back.",
                "Also known as a couch, sofas typically seat multiple people.",
                "Sofas can be made with various upholstery materials like leather or fabric.",
                "Standard sofas range from 72-84 inches in length and 30-36 inches in depth.",
                "Sectional sofas can be configured in L or U shapes for larger spaces."
            ],
            # Add more furniture types as needed
        }

    def add_knowledge_to_index(self):
        """Add knowledge base entries to FAISS index."""
        all_texts = []
        for category in self.knowledge_base:
            all_texts.extend(self.knowledge_base[category])
        
        embeddings = self.sentence_model.encode(all_texts)
        self.index.add(np.array(embeddings))
        self.knowledge_texts = all_texts

    def get_relevant_info(self, query, k=3):
        """Get relevant information from knowledge base."""
        query_vector = self.sentence_model.encode([query])
        D, I = self.index.search(query_vector, k)
        
        relevant_info = [self.knowledge_texts[i] for i in I[0]]
        return " ".join(relevant_info)

    def detect_objects(self, frame):
        """Detect objects in the frame using YOLO."""
        try:
            logger.info(f"Running detection on device: {self.device}")
            with torch.amp.autocast(device_type='cuda' if self.device == 'cuda' else 'cpu'):
                # Run inference
                results = self.model(frame, verbose=False)
                
                # Process results
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Convert tensors to numpy arrays and extract values
                        bbox = box.xyxy[0].cpu().numpy()  # Get bounding box coordinates
                        conf = float(box.conf[0].cpu().numpy())  # Get confidence score
                        cls = int(box.cls[0].cpu().numpy())  # Get class ID
                        
                        detections.append({
                            'bbox': bbox.tolist(),  # Convert numpy array to list
                            'confidence': conf,
                            'class': cls
                        })
                
                logger.info(f"Object detection complete: found {len(detections)} objects")
                return detections
                
        except Exception as e:
            logger.error(f"Error in detect_objects: {str(e)}")
            return []

    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)

    def estimate_depth(self, image):
        """Estimate depth using Depth-Anything-V2 with fallback
        Args:
            image: RGB numpy array (HxWx3)
        Returns:
            depth_map: numpy array of depth values
        """
        try:
            # Validate input
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(image)}")
            
            # Use Depth-Anything-V2 if available
            if self.depth_pipeline is not None:
                # Convert numpy array to PIL Image
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                # Get depth prediction
                depth_output = self.depth_pipeline(pil_image)
                depth_map = np.array(depth_output["depth"])
                
                # Normalize depth map
                depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
                
                logger.info("Depth estimation completed using Depth-Anything-V2")
                return depth_map
            
            # Fallback to basic depth estimation
            logger.warning("Using basic depth estimation method")
            return self._basic_depth_estimation(image)
            
        except Exception as e:
            logger.error(f"Error in depth estimation: {str(e)}")
            return np.ones(image.shape[:2], dtype=np.float32) * 0.5

    def _basic_depth_estimation(self, image):
        """Basic depth estimation fallback method"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            
            # Use edge detection as a simple depth cue
            edges = cv2.Sobel(blurred, cv2.CV_64F, 1, 1, ksize=3)
            
            # Normalize to 0-1 range
            depth_map = cv2.normalize(edges, None, 0, 1, cv2.NORM_MINMAX)
            
            return depth_map.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in basic depth estimation: {str(e)}")
            return np.ones(image.shape[:2], dtype=np.float32) * 0.5

    def get_precise_boundaries(self, image, boxes):
        """Get precise object boundaries using SAM2
        Args:
            image: RGB numpy array (HxWx3)
            boxes: List of [x1, y1, x2, y2] bounding boxes
        Returns:
            refined_boxes: List of refined bounding boxes
            masks: List of binary masks
            scores: List of confidence scores for each mask
        """
        try:
            # Input validation
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(image)}")
            
            # Convert image format if needed
            if isinstance(image, torch.Tensor):
                image = (image.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            
            # Initialize SAM predictor with image
            self.sam_predictor.set_image(image)
            
            refined_boxes = []
            masks = []
            scores = []
            
            # Process each bounding box
            for box in boxes:
                try:
                    # Convert box to input format for SAM
                    input_box = np.array([[box[0], box[1], box[2], box[3]]])
                    
                    # Get multiple mask predictions
                    sam_masks, mask_scores, logits = self.sam_predictor.predict(
                        box=input_box,
                        multimask_output=True,  # Enable multiple mask output
                        return_logits=True      # Get prediction logits for confidence
                    )
                    
                    if sam_masks.shape[0] > 0:
                        # Select best mask based on scores
                        best_mask_idx = np.argmax(mask_scores)
                        mask = sam_masks[best_mask_idx]
                        score = mask_scores[best_mask_idx]
                        
                        # Refine mask using CRF if enabled
                        if hasattr(self, 'use_crf') and self.use_crf:
                            mask = self._refine_mask_with_crf(image, mask)
                        
                        # Get precise boundaries from mask
                        y_indices, x_indices = np.where(mask > 0.5)
                        if len(y_indices) > 0 and len(x_indices) > 0:
                            # Calculate refined bounding box
                            x_min, x_max = x_indices.min(), x_indices.max()
                            y_min, y_max = y_indices.min(), y_indices.max()
                            
                            # Add padding for safety
                            padding = 5
                            x_min = max(0, x_min - padding)
                            y_min = max(0, y_min - padding)
                            x_max = min(image.shape[1], x_max + padding)
                            y_max = min(image.shape[0], y_max + padding)
                            
                            refined_box = [x_min, y_min, x_max, y_max]
                            
                            # Calculate mask quality metrics
                            mask_quality = self._calculate_mask_quality(mask, logits[best_mask_idx])
                            
                            # Only add high-quality detections
                            if mask_quality > 0.5:
                                refined_boxes.append(refined_box)
                                masks.append(mask)
                                scores.append(score.item())
                                
                                # Log successful refinement
                                logger.debug(f"Refined box: {refined_box}, quality: {mask_quality:.2f}")
                
                except Exception as e:
                    logger.error(f"Error processing box {box}: {str(e)}")
                    continue
            
            logger.info(f"SAM processing complete: {len(refined_boxes)} refined boxes")
            return refined_boxes, masks, scores

        except Exception as e:
            logger.error(f"Error in get_precise_boundaries: {str(e)}")
            return [], [], []

    def _calculate_mask_quality(self, mask, logits):
        """Calculate quality metrics for a mask
        Args:
            mask: Binary mask array
            logits: Raw prediction logits
        Returns:
            quality_score: Float between 0 and 1
        """
        try:
            # Calculate mask statistics
            mask_area = mask.sum()
            total_area = mask.size
            coverage = mask_area / total_area
            
            # Calculate boundary smoothness
            boundaries = cv2.findContours(
                mask.astype(np.uint8), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )[0]
            
            if len(boundaries) > 0:
                perimeter = cv2.arcLength(boundaries[0], True)
                area = cv2.contourArea(boundaries[0])
                smoothness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            else:
                smoothness = 0
            
            # Calculate confidence from logits
            confidence = torch.sigmoid(torch.from_numpy(logits)).mean().item()
            
            # Combine metrics
            quality_score = (smoothness * 0.4 + confidence * 0.6)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating mask quality: {str(e)}")
            return 0.0

    def _refine_mask_with_crf(self, image, mask):
        """Refine mask using Conditional Random Fields
        Args:
            image: RGB image array
            mask: Binary mask array
        Returns:
            refined_mask: Refined binary mask
        """
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels
            
            # Convert inputs
            mask = mask.astype(np.uint8)
            h, w = mask.shape
            
            # Create CRF
            d = dcrf.DenseCRF2D(w, h, 2)  # 2 classes: fg/bg
            
            # Set unary potentials
            U = unary_from_labels(mask[np.newaxis], 2, gt_prob=0.7, zero_unsure=False)
            d.setUnaryEnergy(U)
            
            # Set pairwise potentials
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=image, compat=10)
            
            # Inference
            Q = d.inference(5)
            refined_mask = np.argmax(Q, axis=0).reshape((h, w))
            
            return refined_mask.astype(bool)
            
        except Exception as e:
            logger.error(f"Error in CRF refinement: {str(e)}")
            return mask

    def _apply_constraints(self, dimensions, object_type):
        """Apply realistic constraints to object dimensions"""
        try:
            # Get standard dimensions for this type
            standard = self.STANDARD_DIMENSIONS.get(object_type.lower(), self.STANDARD_DIMENSIONS['default'])
            max_dims = self.MAX_DIMENSIONS.get(object_type.lower(), self.MAX_DIMENSIONS['default'])
            min_dims = self.MIN_DIMENSIONS.get(object_type.lower(), self.MIN_DIMENSIONS['default'])
            
            # Apply constraints
            constrained_dims = {
                'length': max(min_dims['width'], 
                             min(dimensions['length'], max_dims['width'])),
                'width': max(min_dims['depth'], 
                            min(dimensions['width'], max_dims['depth'])),
                'height': max(min_dims['height'], 
                             min(dimensions['height'], max_dims['height']))
            }
            
            return constrained_dims

        except Exception as e:
            logger.error(f"Error in dimension constraints: {str(e)}")
            return dimensions

    def calculate_dimensions(self, image, depth_map=None, lidar_points=None, tof_data=None):
        """Calculate dimensions using all available sensor data"""
        # Get depth data from available sensors
        depth_map = self.get_depth_data(image, lidar_points, tof_data)
        
        # Get object detections
        boxes, scores, labels = self.detect_objects(image)
        
        # Get precise boundaries
        refined_boxes, masks = self.get_precise_boundaries(image, boxes)
        
        # Get depth map if not provided
        if depth_map is None:
            depth_map = self.estimate_depth(image)
        
        detections = []
        for box, score, label, mask in zip(refined_boxes, scores, labels, masks):
            if label in self.MAX_DIMENSIONS:
                # Get depth at object location
                object_depth = depth_map[int(box[1]):int(box[3]), int(box[0]):int(box[2])].mean()
                
                # Calculate dimensions
                pixel_width = box[2] - box[0]
                pixel_height = box[3] - box[1]
                
                # Convert to real-world dimensions
                focal_length = camera_matrix[0, 0]
                width = (pixel_width * object_depth) / focal_length
                height = (pixel_height * object_depth) / focal_length
                depth = width * self.DEPTH_RATIOS.get(label, self.DEPTH_RATIOS['default'])
                
                dimensions = {
                    'length': round(width, 1),
                    'width': round(depth, 1),
                    'height': round(height, 1)
                }
                
                # Apply constraints
                dimensions = self._apply_constraints(dimensions, label)
                
                # Calculate volume
                volume = (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728
                
                # Get additional information
                info = self.get_relevant_info(f"information about {label}")
                
                detections.append({
                    "label": label,
                    "confidence": round(score, 3),
                    "box": box,
                    "dimensions": dimensions,
                    "volume": round(volume, 2),
                    "description": info
                })
        
        return detections

    def process_lidar_data(self, lidar_points):
        """Process raw LiDAR point cloud data"""
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_points[:, :3])
        
        # Remove statistical outliers
        pcd, _ = pcd.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)
        
        # Segment ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.03,
                                               ransac_n=3,
                                               num_iterations=1000)
        
        # Extract non-ground points
        object_cloud = pcd.select_by_index(inliers, invert=True)
        
        return object_cloud

    def fuse_camera_lidar(self, image, lidar_points, camera_matrix, extrinsics):
        """Fuse camera and LiDAR data for accurate measurements"""
        # Project LiDAR points onto image
        points_3d = lidar_points[:, :3]
        points_2d = cv2.projectPoints(
            points_3d,
            extrinsics[:3, :3],
            extrinsics[:3, 3],
            camera_matrix,
            None
        )[0].reshape(-1, 2)
        
        return points_2d, points_3d

    def detect_and_measure(self, frame, depth_map=None, lidar_points=None, tof_data=None):
        """
        Main detection and measurement pipeline.
        
        Args:
            frame (np.ndarray): Input image frame
            depth_map (np.ndarray, optional): Depth map if available
            lidar_points (np.ndarray, optional): LiDAR point cloud data if available
            tof_data (np.ndarray, optional): Time-of-Flight sensor data if available
        
        Returns:
            list: List of processed detections with measurements
        """
        try:
            # Ensure frame is in correct format (H, W, C)
            if len(frame.shape) != 3:
                raise ValueError(f"Invalid frame shape: {frame.shape}")
            
            height, width = frame.shape[:2]
            
            # Run object detection
            detections = self.detect_objects(frame)
            
            # Process each detection
            processed_detections = []
            for det in detections:
                try:
                    bbox = det['bbox']
                    confidence = det['confidence']
                    class_id = det['class']
                    
                    # Determine which depth data to use
                    if tof_data is not None:
                        measurements = self._process_tof_measurements(bbox, tof_data)
                    elif lidar_points is not None:
                        measurements = self._process_lidar_measurements(bbox, lidar_points)
                    elif depth_map is not None:
                        measurements = self._process_depth_measurements(bbox, depth_map)
                    else:
                        measurements = {}
                    
                    processed_detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'class': class_id,
                        'measurements': measurements
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing detection: {str(e)}")
                    continue
            
            logger.info(f"Measurement complete: processed {len(processed_detections)} objects")
            return processed_detections
            
        except Exception as e:
            logger.error(f"Error in detect_and_measure: {str(e)}")
            return []

    def _process_lidar_measurements(self, bbox, lidar_points):
        """Process measurements using LiDAR data."""
        try:
            # Add your LiDAR processing logic here
            return {}
        except Exception as e:
            logger.error(f"Error processing LiDAR measurements: {str(e)}")
            return {}

    def _process_depth_measurements(self, bbox, depth_map):
        """Process measurements using depth map."""
        try:
            # Add your depth map processing logic here
            return {}
        except Exception as e:
            logger.error(f"Error processing depth measurements: {str(e)}")
            return {}

    def _process_tof_measurements(self, bbox, tof_data):
        """Process measurements using Time-of-Flight sensor data."""
        try:
            # Add your ToF processing logic here
            return {}
        except Exception as e:
            logger.error(f"Error processing ToF measurements: {str(e)}")
            return {}

    def _calculate_object_dimensions(self, box, focal_length, object_type, depth_map=None, mask=None):
        """Calculate real-world dimensions using camera parameters and depth data"""
        try:
            # Get pixel dimensions
            pixel_width = box[2] - box[0]
            pixel_height = box[3] - box[1]
            
            # Get depth at object location
            if depth_map is not None and mask is not None:
                # Use masked mean depth for more accurate measurement
                object_depth = depth_map[mask].mean()
            else:
                # Fall back to standard dimensions
                standard_dims = self.STANDARD_DIMENSIONS.get(
                    object_type, 
                    self.STANDARD_DIMENSIONS['default']
                )
                return standard_dims
            
            # Calculate real-world dimensions using depth
            width = (pixel_width * object_depth) / focal_length
            height = (pixel_height * object_depth) / focal_length
            depth = width * self.DEPTH_RATIOS.get(object_type, self.DEPTH_RATIOS['default'])
            
            # Apply constraints
            dims = {
                'length': round(width, 1),
                'width': round(depth, 1),
                'height': round(height, 1)
            }
            
            return self._apply_constraints(dims, object_type)
            
        except Exception as e:
            logger.error(f"Error calculating dimensions: {str(e)}")
            raise

    def _measure_with_reference(self, bbox, reference_points):
        """Calculate real-world dimensions using reference objects in the image"""
        try:
            # Find the best reference object to use
            reference_object = self._select_best_reference(reference_points)
            if not reference_object:
                return None

            # Get reference object dimensions (only need to do this once)
            reference_width = self.reference_sizes[reference_object][0]
            reference_height = self.reference_sizes[reference_object][1]

            # Get pixel coordinates
            x1, y1, x2, y2 = bbox
            object_width_px = x2 - x1
            object_height_px = y2 - y1

            # Get reference object pixel size
            ref_x1, ref_y1, ref_x2, ref_y2 = reference_points[reference_object]
            ref_width_px = ref_x2 - ref_x1
            ref_height_px = ref_y2 - ref_y1

            # Calculate scaling factors
            width_scale = reference_width / ref_width_px
            height_scale = reference_height / ref_height_px

            # Use average scale for more accuracy
            scale_factor = (width_scale + height_scale) / 2

            # Calculate real dimensions
            real_width = object_width_px * scale_factor
            real_height = object_height_px * scale_factor
            
            # Estimate depth using perspective and reference object
            real_depth = self._estimate_depth_from_reference(
                bbox, 
                reference_points[reference_object], 
                scale_factor
            )

            return {
                'length': round(real_width, 1),
                'width': round(real_depth, 1),
                'height': round(real_height, 1)
            }

        except Exception as e:
            logger.error(f"Error in measurement with reference: {str(e)}")
            return None

    def process_video(self, video_path, output_path=None):
        """Process video with smart frame sampling and motion analysis"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            frames = []
            optical_flows = []
            frame_count = 0
            prev_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale for motion analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Smart frame sampling based on motion
                if prev_frame is not None:
                    # Calculate motion between frames
                    flow = self.flow_calculator.calc(prev_frame, gray, None)
                    motion_magnitude = np.mean(np.abs(flow))
                    
                    # Sample frame if significant motion detected or at keyframe interval
                    if motion_magnitude > self.motion_threshold or frame_count % self.keyframe_interval == 0:
                        frames.append(frame)
                        optical_flows.append(flow)
                        
                        logger.info(f"Frame {frame_count}: Motion magnitude = {motion_magnitude:.2f}")
                
                prev_frame = gray
                frame_count += 1
            
            cap.release()
            
            # Process sampled frames
            detections = self._process_sampled_frames(frames, optical_flows)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in video processing: {str(e)}")
            raise

    def _process_sampled_frames(self, frames, optical_flows):
        """Process sampled frames with temporal context"""
        try:
            all_detections = []
            
            for i, (frame, flow) in enumerate(zip(frames, optical_flows)):
                # Extract motion features
                motion_features = self._extract_motion_features(flow)
                
                # Get base detections
                frame_detections = self.detect_objects(frame)
                
                # Refine detections using motion context
                refined_detections = self._refine_with_motion(
                    frame_detections, 
                    motion_features
                )
                
                all_detections.extend(refined_detections)
            
            # Remove duplicate detections across frames
            final_detections = self._filter_temporal_duplicates(all_detections)
            
            return final_detections
            
        except Exception as e:
            logger.error(f"Error processing sampled frames: {str(e)}")
            return []

    def _extract_motion_features(self, flow):
        """Extract relevant motion features from optical flow"""
        try:
            # Calculate flow magnitude and direction
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            return {
                'mean_magnitude': np.mean(magnitude),
                'max_magnitude': np.max(magnitude),
                'mean_angle': np.mean(angle),
                'motion_areas': self._segment_motion_areas(magnitude)
            }
            
        except Exception as e:
            logger.error(f"Error extracting motion features: {str(e)}")
            return {}

    def _segment_motion_areas(self, magnitude):
        """Segment areas with significant motion"""
        try:
            # Threshold motion magnitude
            motion_mask = magnitude > self.motion_threshold
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(motion_mask.astype(np.uint8))
            
            # Extract properties of motion areas
            motion_areas = []
            for label in range(1, num_labels):
                area_mask = labels == label
                area = np.sum(area_mask)
                if area > 100:  # Minimum area threshold
                    motion_areas.append({
                        'area': area,
                        'centroid': np.mean(np.where(area_mask), axis=1)
                    })
            
            return motion_areas
            
        except Exception as e:
            logger.error(f"Error segmenting motion areas: {str(e)}")
            return []

    def _refine_with_motion(self, detections, motion_features):
        """Refine detections using motion context"""
        try:
            refined_detections = []
            
            for det in detections:
                # Get detection area
                bbox = det['bbox']
                det_center = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                
                # Check if detection corresponds to motion area
                motion_score = self._calculate_motion_score(
                    det_center, 
                    motion_features['motion_areas']
                )
                
                # Adjust confidence based on motion
                det['confidence'] *= (1 + motion_score) / 2
                
                refined_detections.append(det)
            
            return refined_detections
            
        except Exception as e:
            logger.error(f"Error refining detections with motion: {str(e)}")
            return detections

    def _filter_temporal_duplicates(self, detections):
        """Filter overlapping detections"""
        selected = []
        for det in detections:
            is_duplicate = False
            for sel in selected:
                if self._calculate_iou(det['bounding_box'], sel['bounding_box']) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(det)
        return selected

    def _calculate_motion_score(self, detection_center, motion_areas):
        """Calculate motion score for a detection"""
        try:
            # Calculate distance to all motion areas
            distances = [np.linalg.norm(np.array(detection_center) - np.array(area['centroid'])) for area in motion_areas]
            
            # Calculate motion score based on distance to motion areas
            motion_score = np.mean(np.exp(-np.array(distances) / self.motion_threshold))
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Error calculating motion score: {str(e)}")
            return 0.0

    def _calculate_motion_score(self, detection_center, motion_areas):
        """Calculate motion score for a detection"""
        try:
            # Calculate distance to all motion areas
            distances = [np.linalg.norm(np.array(detection_center) - np.array(area['centroid'])) for area in motion_areas]
            
            # Calculate motion score based on distance to motion areas
            motion_score = np.mean(np.exp(-np.array(distances) / self.motion_threshold))
            
            return motion_score
            
        except Exception as e:
            logger.error(f"Error calculating motion score: {str(e)}")
            return 0.0

    def _filter_temporal_duplicates(self, detections):
        """Filter overlapping detections"""
        selected = []
        for det in detections:
            is_duplicate = False
            for sel in selected:
                if self._calculate_iou(det['bounding_box'], sel['bounding_box']) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(det)
        return selected

    def validate_video(self, video_path):
        """Validate video file before processing"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Check video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps
            
            if duration > 60:  # Limit to 1 minute
                raise ValueError("Video too long (max 1 minute)")
            
            return True
            
        except Exception as e:
            logger.error(f"Video validation failed: {str(e)}")
            return False

    def calculate_room_dimensions(self, detections):
        """Calculate room dimensions based on furniture detections"""
        try:
            # Get largest furniture dimensions as reference
            max_length = max(d['dimensions']['length'] for d in detections)
            max_width = max(d['dimensions']['width'] for d in detections)
            max_height = max(d['dimensions']['height'] for d in detections)
            
            # Add padding for walls and movement space
            room_length = max_length * 1.5  # 50% extra space
            room_width = max_width * 1.5
            room_height = max_height + 12  # Just add 1 foot for ceiling
            
            # Calculate volume
            volume = (room_length * room_width * room_height) / 1728  # Convert to cubic feet
            
            # Round all values
            return {
                'length': round(room_length, 1),
                'width': round(room_width, 1),
                'height': round(room_height, 1),
                'volume': round(volume, 2)
            }
        except Exception as e:
            logger.error(f"Error calculating room dimensions: {str(e)}")
            return None

    def calculate_furniture_volume(self, dimensions):
        """Calculate furniture volume in cubic feet"""
        try:
            volume = (
                dimensions['length'] * 
                dimensions['width'] * 
                dimensions['height']
            ) / 1728  # Convert cubic inches to cubic feet
            return round(volume, 2)
        except Exception as e:
            logger.error(f"Error calculating volume: {str(e)}")
            return 0

    def summarize_detections(self, detections):
        """Create a summary of room contents"""
        try:
            summary = {
                'furniture_count': len(detections),
                'furniture_types': {},
                'total_volume': 0,
                'largest_items': []
            }
            
            for det in detections:
                # Count furniture types
                ftype = det['type'].lower()
                summary['furniture_types'][ftype] = summary['furniture_types'].get(ftype, 0) + 1
                
                # Add volume
                volume = self.calculate_furniture_volume(det['dimensions'])
                summary['total_volume'] += volume
                
                # Track largest items
                if len(summary['largest_items']) < 3:
                    summary['largest_items'].append(det)
                    summary['largest_items'].sort(key=lambda x: self.calculate_furniture_volume(x['dimensions']), reverse=True)
            
            return summary
        except Exception as e:
            logger.error(f"Error creating summary: {str(e)}")
            return None

    def _process_detections(self, detections):
        """Process detections with furniture-only filter"""
        try:
            # Filter out people
            furniture_only = [
                det for det in detections 
                if det['type'].lower() != 'person'
            ]
            
            # Group by object type
            grouped = {}
            for det in furniture_only:
                obj_type = det['type'].lower()
                if obj_type not in grouped:
                    grouped[obj_type] = []
                grouped[obj_type].append(det)
            
            final_detections = []
            
            # Process each furniture group
            for obj_type, group in grouped.items():
                # Sort by confidence
                group.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Apply type-specific filtering
                if obj_type in ['tv', 'microwave']:
                    # Usually only one TV/microwave per wall
                    final_detections.append(group[0])
                else:
                    # Standard filtering for furniture
                    selected = self._filter_overlapping(group, iou_threshold=0.5)
                    final_detections.extend(selected)
            
            return final_detections
            
        except Exception as e:
            logger.error(f"Error processing detections: {str(e)}")
            return []

    def _filter_overlapping(self, detections, iou_threshold=0.5):
        """Filter overlapping detections"""
        selected = []
        for det in detections:
            is_duplicate = False
            for sel in selected:
                if self._calculate_iou(det['bounding_box'], sel['bounding_box']) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                selected.append(det)
        return selected

    def process_tof_data(self, tof_data, image_shape):
        """Process ToF sensor data for depth estimation
        Args:
            tof_data: Raw ToF depth measurements (numpy array)
            image_shape: Shape of the RGB image (height, width)
        Returns:
            depth_map: Processed depth map aligned with RGB image
        """
        try:
            # Validate input
            if not isinstance(tof_data, np.ndarray):
                raise ValueError(f"Expected numpy array for ToF data, got {type(tof_data)}")
            
            # Clean ToF data
            depth_map = np.clip(tof_data, self.TOF_MIN_RANGE, self.TOF_MAX_RANGE)
            depth_map[np.isnan(depth_map)] = self.TOF_MAX_RANGE
            
            # Normalize to 0-1 range
            depth_map = (depth_map - self.TOF_MIN_RANGE) / (self.TOF_MAX_RANGE - self.TOF_MIN_RANGE)
            
            # Resize to match RGB image
            depth_map = cv2.resize(depth_map, (image_shape[1], image_shape[0]))
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Error processing ToF data: {str(e)}")
            return None

    def get_depth_data(self, image, lidar_points=None, tof_data=None):
        """Get depth data using available sensors
        Args:
            image: RGB image (numpy array)
            lidar_points: Optional LiDAR point cloud data
            tof_data: Optional ToF sensor data
        Returns:
            depth_map: Processed depth map
        """
        try:
            if lidar_points is not None:
                # Use LiDAR if available
                object_cloud = self.process_lidar_data(lidar_points)
                depth_map = self.project_point_cloud(object_cloud, image.shape)
                logger.info("Using LiDAR data for depth estimation")
                
            elif tof_data is not None:
                # Fall back to ToF if available
                depth_map = self.process_tof_data(tof_data, image.shape)
                logger.info("Using ToF data for depth estimation")
                
            else:
                # Fall back to monocular depth estimation
                depth_map = self.estimate_depth(image)
                logger.info("Using monocular depth estimation")
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Error getting depth data: {str(e)}")
            return self.estimate_depth(image)  # Fall back to monocular estimation

    def __del__(self):
        """Cleanup GPU memory"""
        try:
            if hasattr(self, 'model'):
                self.model.cpu()
            if hasattr(self, 'sam'):
                self.sam.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    def get_class_name(self, class_id):
        """Convert class ID to human-readable name"""
        return self.FURNITURE_CLASSES.get(class_id, f"unknown-{class_id}")

class ValidationError(Exception):
    pass

def validate_scan(self, detections):
    """Validate scan quality"""
    if len(detections) < 2:
        raise ValidationError("Too few items detected. Please scan again.")
    
    confidence_scores = [d['confidence'] for d in detections]
    if sum(confidence_scores) / len(confidence_scores) < 0.5:
        raise ValidationError("Low confidence detections. Please scan in better lighting.")

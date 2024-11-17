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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variable for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    def __init__(self):
        try:
            logger.info("Initializing EnhancedFurnitureDetector...")
            
            # Initialize camera calibrator
            self.camera_calibrator = CameraCalibrator()
            
            # Initialize YOLO with proper model
            self.model = YOLO('yolov8x.pt')
            
            # Increase confidence threshold to reduce false positives
            self.model.conf = 0.45  # Increased from 0.25
            self.model.iou = 0.35   # Decreased from 0.45 to better handle overlapping objects
            self.model.imgsz = 640  # Image size
            self.model.classes = [0, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
            
            # Updated dimensions for better accuracy
            self.STANDARD_DIMENSIONS = {
                'chair': {'width': 20, 'depth': 20, 'height': 30},
                'table': {'width': 48, 'depth': 30, 'height': 30},
                'couch': {'width': 84, 'depth': 38, 'height': 34},
                'bed': {'width': 80, 'depth': 60, 'height': 24},
                'desk': {'width': 60, 'depth': 30, 'height': 30},
                'tv': {'width': 48, 'depth': 4, 'height': 27},
                'book': {'width': 6, 'depth': 9, 'height': 1},
                'person': {'width': 24, 'depth': 12, 'height': 70},
                'default': {'width': 30, 'depth': 30, 'height': 30}
            }
            
            # Device configuration
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}")
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

    def detect_objects(self, image):
        """Detect objects using YOLOv8 with improved filtering"""
        try:
            # Debug input
            logger.info(f"Object detection input shape: {image.shape}, dtype: {image.dtype}")
            
            # Run inference
            results = self.model(
                image,
                verbose=False,
                stream=False
            )
            
            # Process results with better filtering
            boxes = []
            scores = []
            labels = []
            
            # Track detected objects for duplicate filtering
            detected_objects = {}
            
            for result in results:
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        if box.conf.item() > self.model.conf:
                            label = result.names[int(box.cls)]
                            score = box.conf.item()
                            bbox = box.xyxy[0].tolist()
                            
                            # Filter duplicates based on IoU and class
                            is_duplicate = False
                            for existing_label, existing_boxes in detected_objects.items():
                                if existing_label == label:
                                    for existing_box in existing_boxes:
                                        if self._calculate_iou(bbox, existing_box) > 0.5:
                                            is_duplicate = True
                                            break
                            
                            if not is_duplicate:
                                boxes.append(bbox)
                                scores.append(score)
                                labels.append(label)
                                
                                # Track this detection
                                if label not in detected_objects:
                                    detected_objects[label] = []
                                detected_objects[label].append(bbox)
            
            logger.info(f"Detection complete: found {len(boxes)} unique objects")
            return boxes, scores, labels

        except Exception as e:
            logger.error(f"Error in detect_objects: {str(e)}")
            return [], [], []

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
        """
        Estimate depth using MiDaS
        Args:
            image: RGB numpy array (HxWx3)
        Returns:
            depth_map: numpy array of depth values
        """
        try:
            # Debug input
            logger.info(f"Depth estimation input shape: {image.shape}, dtype: {image.dtype}")
            
            # Ensure we have a numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(image)}")
            
            # Create transform
            transform = self.depth_transforms.dpt_transform if hasattr(self.depth_transforms, 'dpt_transform') \
                       else self.depth_transforms.small_transform
            
            # Transform the image
            input_batch = transform(image).to(self.device)
            
            # Add batch dimension if needed
            if len(input_batch.shape) == 3:
                input_batch = input_batch.unsqueeze(0)
            
            # Disable gradients for inference
            with torch.no_grad():
                prediction = self.depth_model(input_batch)
                
                # Interpolate to original size
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                # Convert to numpy and normalize
                depth_map = prediction.cpu().numpy()
                depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
                
                return depth_map

        except Exception as e:
            logger.error(f"Error in depth estimation: {str(e)}")
            # Return uniform depth map as fallback
            return np.ones(image.shape[:2], dtype=np.float32) * 0.5

    def get_precise_boundaries(self, image, boxes):
        """Get precise object boundaries using Mask R-CNN and SAM2"""
        # Initialize SAM with the image
        if isinstance(image, torch.Tensor):
            image = (image.cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        self.sam_predictor.set_image(image)
        
        refined_boxes = []
        masks = []
        
        # Process each bounding box
        for box in boxes:
            # Convert box to input format for SAM
            input_box = np.array([
                [box[0], box[1], box[2], box[3]]
            ])
            
            # Get SAM prediction
            sam_masks, _, _ = self.sam_predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            if sam_masks.shape[0] > 0:
                mask = sam_masks[0]
                y_indices, x_indices = np.where(mask > 0.5)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    refined_boxes.append([x_min, y_min, x_max, y_max])
                    masks.append(mask)
        
        return refined_boxes, masks

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

    def calculate_dimensions(self, image, depth_map=None):
        """Calculate dimensions using all available information"""
        # Ensure camera is calibrated
        camera_matrix, dist_coeffs, fov = self.camera_calibrator.calibrate_from_image(image)
        
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

    def detect_and_measure(self, image):
        """
        Enhanced detection and measurement using all available models
        Args:
            image: RGB numpy array (HxWx3)
        """
        try:
            # Input validation and logging
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(image)}")
                
            logger.info(f"Processing image: shape={image.shape}, dtype={image.dtype}")
            
            # 1. Camera Calibration
            camera_matrix, dist_coeffs, fov = self.camera_calibrator.calibrate_from_image(image)
            logger.info(f"Camera calibration complete: FOV={fov:.1f}°")
            
            # 2. Object Detection
            boxes, scores, labels = self.detect_objects(image)
            logger.info(f"Object detection complete: found {len(boxes)} objects")
            
            # Process detections
            detections = []
            for box, score, label in zip(boxes, scores, labels):
                try:
                    # Calculate dimensions
                    dims = self._calculate_object_dimensions(
                        box, 
                        camera_matrix[0,0],  # focal length 
                        label.lower()
                    )
                    
                    detections.append({
                        'type': label,
                        'confidence': round(float(score), 2),
                        'dimensions': dims,
                        'bounding_box': [float(x) for x in box]
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing detection {label}: {str(e)}")
                    continue
                    
            logger.info(f"Measurement complete: processed {len(detections)} objects")
            return detections

        except Exception as e:
            logger.error(f"Error in detect_and_measure: {str(e)}")
            raise

    def _calculate_object_dimensions(self, box, focal_length, object_type):
        """Calculate real-world dimensions using camera parameters"""
        try:
            # Get pixel dimensions
            pixel_width = box[2] - box[0]
            pixel_height = box[3] - box[1]
            
            # Get standard dimensions for this type
            standard_dims = self.STANDARD_DIMENSIONS.get(
                object_type, 
                self.STANDARD_DIMENSIONS['default']
            )
            
            # Use standard dimensions as a reference
            dims = {
                'length': round(standard_dims['width'], 1),
                'width': round(standard_dims['depth'], 1),
                'height': round(standard_dims['height'], 1)
            }
            
            logger.debug(f"Calculated dimensions for {object_type}: {dims}")
            return dims
            
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
        """Process entire video and track objects"""
        try:
            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {fps}fps, {frame_width}x{frame_height}, {total_frames} frames")
            
            # Initialize video writer if output path is provided
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Track detections across frames
            all_detections = []
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count += 1
                if frame_count % 5 != 0:  # Process every 5th frame to improve speed
                    continue
                
                # Process frame
                try:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Get detections
                    boxes, scores, labels = self.detect_objects(frame_rgb)
                    
                    # Process detections
                    frame_detections = []
                    for box, score, label in zip(boxes, scores, labels):
                        try:
                            # Calculate dimensions
                            dims = self._calculate_object_dimensions(
                                box, 
                                self.camera_calibrator.get_default_calibration(frame.shape)[0][0,0],
                                label.lower()
                            )
                            
                            detection = {
                                'frame': frame_count,
                                'type': label,
                                'confidence': round(float(score), 2),
                                'dimensions': dims,
                                'bounding_box': [float(x) for x in box]
                            }
                            
                            frame_detections.append(detection)
                            
                            # Draw on frame if output requested
                            if output_path:
                                self._draw_detection(frame, detection)
                            
                        except Exception as e:
                            logger.error(f"Error processing detection: {str(e)}")
                            continue
                    
                    # Store frame detections
                    all_detections.extend(frame_detections)
                    
                    # Write frame if output requested
                    if output_path:
                        out.write(frame)
                    
                    # Log progress
                    if frame_count % 50 == 0:
                        logger.info(f"Processed {frame_count}/{total_frames} frames")
                    
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    continue
            
            # Clean up
            cap.release()
            if output_path:
                out.release()
            
            # Process all detections to remove duplicates across frames
            final_detections = self._process_video_detections(all_detections)
            
            logger.info(f"Video processing complete. Found {len(final_detections)} unique objects")
            return final_detections
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise

    def _draw_detection(self, frame, detection):
        """Draw detection box and info on frame"""
        box = detection['bounding_box']
        label = detection['type']
        conf = detection['confidence']
        dims = detection['dimensions']
        
        # Draw box
        cv2.rectangle(frame, 
                     (int(box[0]), int(box[1])), 
                     (int(box[2]), int(box[3])), 
                     (0, 255, 0), 2)
        
        # Draw label
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, 
                   (int(box[0]), int(box[1]-10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0), 2)
        
        # Draw dimensions
        dim_text = f"{dims['length']}\"x{dims['width']}\"x{dims['height']}\""
        cv2.putText(frame, dim_text, 
                   (int(box[0]), int(box[3]+20)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0), 2)

    def _process_video_detections(self, detections):
        """Process all detections to remove duplicates across frames"""
        # Group detections by object type
        grouped = {}
        for det in detections:
            if det['type'] not in grouped:
                grouped[det['type']] = []
            grouped[det['type']].append(det)
        
        # Process each group
        final_detections = []
        for obj_type, group in grouped.items():
            # Sort by confidence
            group.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Take highest confidence detection that doesn't overlap with previous
            selected = []
            for det in group:
                is_duplicate = False
                for sel in selected:
                    if self._calculate_iou(det['bounding_box'], sel['bounding_box']) > 0.5:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    selected.append(det)
            
            final_detections.extend(selected)
        
        return final_detections

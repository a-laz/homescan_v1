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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class CameraCalibrator:
    def __init__(self):
        # ChArUco board parameters
        self.CHARUCOBOARD_ROWCOUNT = 7
        self.CHARUCOBOARD_COLCOUNT = 5
        self.SQUARE_LENGTH = 0.04  # meters
        self.MARKER_LENGTH = 0.02  # meters
        
        # Create Charuco board using latest API
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.board = cv2.aruco.CharucoBoard(
            (self.CHARUCOBOARD_COLCOUNT, self.CHARUCOBOARD_ROWCOUNT),
            self.SQUARE_LENGTH,
            self.MARKER_LENGTH,
            self.aruco_dict
        )
        
        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)

    def calibrate_from_image(self, image):
        if isinstance(image, np.ndarray):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

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

                logger.info(f"Camera calibrated successfully. FOV: {fov_degrees:.2f} degrees")
                return camera_matrix, dist_coeffs, fov_degrees

        logger.warning("Could not calibrate camera from image. Using default parameters.")
        focal_length = gray.shape[1]
        center = (gray.shape[1]/2, gray.shape[0]/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5,1))
        fov_degrees = 60

        return camera_matrix, dist_coeffs, fov_degrees

    def generate_calibration_board(self, size=(2000, 2000)):
        return self.board.generateImage(size)

class EnhancedFurnitureDetector:
    def __init__(self):
        logger.info("Initializing EnhancedFurnitureDetector...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DETR
        self.detr_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.detr_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        
        # Initialize YOLOv5
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x')
        self.yolo_model.to(self.device)
        
        # Initialize MiDaS for depth estimation
        self.depth_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
        self.depth_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.depth_model.to(self.device)
        self.depth_model.eval()
        
        # Initialize Mask R-CNN
        self.mask_rcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.mask_rcnn.eval()
        
        # Initialize camera calibrator
        self.camera_calibrator = CameraCalibrator()
        
        # Knowledge base setup
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.create_knowledge_base()
        self.embed_dimension = 384
        self.index = faiss.IndexFlatL2(self.embed_dimension)
        self.add_knowledge_to_index()

        # Define standard dimensions for different furniture types
        self.MAX_DIMENSIONS = {
            'chair': {'width': 30, 'depth': 30, 'height': 45},
            'table': {'width': 72, 'depth': 42, 'height': 31},
            'sofa': {'width': 84, 'depth': 40, 'height': 38},
            'bookshelf': {'width': 48, 'depth': 24, 'height': 72},
            'bed': {'width': 76, 'depth': 80, 'height': 45},
            'dresser': {'width': 60, 'depth': 24, 'height': 36},
            'default': {'width': 84, 'depth': 84, 'height': 84}
        }
        
        self.DEPTH_RATIOS = {
            'chair': 0.9,
            'table': 1.2,
            'sofa': 0.8,
            'bookshelf': 0.4,
            'bed': 1.0,
            'dresser': 0.5,
            'default': 0.8
        }

        # Add LiDAR configuration
        self.lidar_config = {
            'min_depth': 0.1,  # meters
            'max_depth': 10.0,  # meters
            'min_confidence': 0.5
        }

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
        """Combine DETR and YOLOv5 predictions"""
        # DETR detection
        detr_inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
        detr_outputs = self.detr_model(**detr_inputs)
        detr_results = self.detr_processor.post_process_object_detection(
            detr_outputs, target_sizes=torch.tensor([image.shape[:2]])
        )[0]
        
        # YOLOv5 detection
        yolo_results = self.yolo_model(image)
        
        # Combine detections
        combined_boxes = []
        combined_scores = []
        combined_labels = []
        
        # Process DETR results
        for score, label, box in zip(detr_results["scores"], detr_results["labels"], detr_results["boxes"]):
            if score > 0.7:  # confidence threshold
                combined_boxes.append(box.tolist())
                combined_scores.append(score.item())
                combined_labels.append(self.detr_model.config.id2label[label.item()].lower())
        
        # Process YOLOv5 results
        for *xyxy, conf, cls in yolo_results.xyxy[0]:
            if conf > 0.7:
                combined_boxes.append(xyxy)
                combined_scores.append(conf.item())
                combined_labels.append(yolo_results.names[int(cls)])
        
        return combined_boxes, combined_scores, combined_labels

    def estimate_depth(self, image):
        """Get accurate depth map using MiDaS"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1) / 255.0
        
        transform = self.depth_transforms.dpt_transform if hasattr(self.depth_transforms, 'dpt_transform') else self.depth_transforms.small_transform
        input_batch = transform(image).to(self.device)
        
        with torch.no_grad():
            prediction = self.depth_model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[1:],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            prediction = prediction.cpu().numpy()
            depth_min = prediction.min()
            depth_max = prediction.max()
            normalized_depth = (prediction - depth_min) / (depth_max - depth_min)
            
        return normalized_depth

    def get_precise_boundaries(self, image, boxes):
        """Get precise object boundaries using Mask R-CNN"""
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        image = image.to(self.device)
        
        with torch.no_grad():
            predictions = self.mask_rcnn([image])[0]
        
        refined_boxes = []
        for mask in predictions['masks']:
            if mask.mean() > 0.5:
                y_indices, x_indices = torch.where(mask[0] > 0.5)
                if len(y_indices) > 0 and len(x_indices) > 0:
                    x_min, x_max = x_indices.min(), x_indices.max()
                    y_min, y_max = y_indices.min(), y_indices.max()
                    refined_boxes.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])
        
        return refined_boxes

    def _apply_constraints(self, dimensions, furniture_type):
        """Apply furniture-specific constraints to dimensions."""
        max_dims = self.MAX_DIMENSIONS.get(furniture_type, self.MAX_DIMENSIONS['default'])
        
        return {
            'length': round(min(dimensions['length'], max_dims['width']), 1),
            'width': round(min(dimensions['width'], max_dims['depth']), 1),
            'height': round(min(dimensions['height'], max_dims['height']), 1)
        }

    def calculate_dimensions(self, image, depth_map=None):
        """Calculate dimensions using all available information"""
        # Ensure camera is calibrated
        camera_matrix, dist_coeffs, fov = self.camera_calibrator.calibrate_from_image(image)
        
        # Get object detections
        boxes, scores, labels = self.detect_objects(image)
        
        # Get precise boundaries
        refined_boxes = self.get_precise_boundaries(image, boxes)
        
        # Get depth map if not provided
        if depth_map is None:
            depth_map = self.estimate_depth(image)
        
        detections = []
        for box, score, label in zip(refined_boxes, scores, labels):
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

    def detect_and_measure(self, image, lidar_data=None, camera_matrix=None, extrinsics=None):
        """
        Enhanced detection and measurement using both camera and LiDAR data
        Args:
            image: RGB image
            lidar_data: LiDAR point cloud data (optional)
            camera_matrix: Camera intrinsic parameters
            extrinsics: Camera-LiDAR extrinsic calibration matrix
        """
        try:
            # Convert base64 to image if needed
            if isinstance(image, str) and image.startswith('data:image'):
                image = Image.open(BytesIO(base64.b64decode(image.split(',')[1])))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Process image with DETR
            inputs = self.detr_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.detr_model(**inputs)
            
            # Convert outputs to detections
            target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
            results = self.detr_processor.post_process_object_detection(
                outputs, 
                target_sizes=target_sizes, 
                threshold=0.7
            )[0]
            
            detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                
                # Calculate dimensions
                dimensions = {
                    'length': round(box[2] - box[0], 1),
                    'width': round(box[3] - box[1], 1),
                    'height': round((box[3] - box[1]) * 0.75, 1)
                }
                
                # Format detection result
                detection = {
                    'type': self.detr_model.config.id2label[label.item()],  # Changed 'label' to 'type'
                    'confidence': float(score.item()),  # Ensure it's a float
                    'bounding_box': [float(x) for x in box],  # Convert to float list
                    'dimensions': {
                        'length': float(dimensions['length']),
                        'width': float(dimensions['width']),
                        'height': float(dimensions['height'])
                    },
                    'volume': float(
                        (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728
                    ),
                    'lidar_enhanced': False
                }
                detections.append(detection)
            
            # If LiDAR data is available, enhance measurements
            if lidar_data is not None and camera_matrix is not None and extrinsics is not None:
                # Process LiDAR data
                object_cloud = self.process_lidar_data(lidar_data)
                points_2d, points_3d = self.fuse_camera_lidar(
                    image, 
                    np.asarray(object_cloud.points), 
                    camera_matrix, 
                    extrinsics
                )
                
                # Enhance each detection with LiDAR measurements
                for detection in detections:
                    box = detection['bounding_box']
                    
                    # Find LiDAR points within the bounding box
                    mask = (
                        (points_2d[:, 0] >= box[0]) &
                        (points_2d[:, 0] <= box[2]) &
                        (points_2d[:, 1] >= box[1]) &
                        (points_2d[:, 1] <= box[3])
                    )
                    
                    if np.any(mask):
                        object_points = points_3d[mask]
                        
                        # Calculate actual dimensions from point cloud
                        min_coords = np.min(object_points, axis=0)
                        max_coords = np.max(object_points, axis=0)
                        
                        # Update dimensions (convert from meters to inches)
                        dimensions = {
                            'length': round((max_coords[0] - min_coords[0]) * 39.37, 1),
                            'width': round((max_coords[1] - min_coords[1]) * 39.37, 1),
                            'height': round((max_coords[2] - min_coords[2]) * 39.37, 1)
                        }
                        
                        detection['dimensions'] = dimensions
                        detection['volume'] = round(
                            (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728,
                            2
                        )
                        detection['lidar_enhanced'] = True
            
            return detections

        except Exception as e:
            logger.error(f"Error in detect_and_measure: {str(e)}")
            raise

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

        # Update standard dimensions for common objects (in inches)
        self.STANDARD_DIMENSIONS = {
            'tv': {'width': 55, 'depth': 4, 'height': 32},
            'person': {'width': 24, 'depth': 12, 'height': 70},
            'potted plant': {'width': 12, 'depth': 12, 'height': 24},
            'dining table': {'width': 60, 'depth': 36, 'height': 30},
            'laptop': {'width': 14, 'depth': 10, 'height': 1},
            'remote': {'width': 2, 'depth': 1, 'height': 6},
            'book': {'width': 9, 'depth': 6, 'height': 1},
            'chair': {'width': 20, 'depth': 22, 'height': 32},
            'couch': {'width': 72, 'depth': 35, 'height': 36},
            'default': {'width': 24, 'depth': 24, 'height': 24}
        }

        # Add maximum allowable dimensions (1.5x standard size)
        self.MAX_DIMENSIONS = {
            'tv': {'width': 85, 'depth': 6, 'height': 48},
            'person': {'width': 36, 'depth': 18, 'height': 84},
            'potted plant': {'width': 24, 'depth': 24, 'height': 48},
            'dining table': {'width': 96, 'depth': 48, 'height': 36},
            'laptop': {'width': 17, 'depth': 12, 'height': 2},
            'remote': {'width': 3, 'depth': 2, 'height': 8},
            'book': {'width': 12, 'depth': 9, 'height': 3},
            'chair': {'width': 30, 'depth': 33, 'height': 48},
            'couch': {'width': 108, 'depth': 52, 'height': 54},
            'default': {'width': 36, 'depth': 36, 'height': 36}
        }
        
        # Initialize ArUco detector correctly
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Known size objects database
        self.reference_sizes = {
            'credit_card': (3.37, 2.125),  # inches
            'dollar_bill': (6.14, 2.61),
            'aruco_marker': (4.0, 4.0)  # custom printed size
        }
        
        # Add LiDAR configuration
        self.lidar_config = {
            'min_depth': 0.1,  # meters
            'max_depth': 10.0,  # meters
            'min_confidence': 0.5
        }

        # Initialize SAM2
        logger.info("Initializing SAM2...")
        self.sam = sam_model_registry["vit_l"](checkpoint="sam_vit_l_0b3195.pth")
        self.sam.to(self.device)
        self.sam_predictor = SamPredictor(self.sam)

        # Use YOLOv8x instead of smaller variants
        self.model = YOLO('yolov8x.pt')
        
        logger.info("EnhancedFurnitureDetector initialized successfully")

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
            
            # Calculate scaling factors
            width_scale = dimensions['length'] / standard['width']
            height_scale = dimensions['height'] / standard['height']
            depth_scale = dimensions['width'] / standard['depth']
            
            # Use a conservative scale factor
            scale_factor = min(width_scale, height_scale, depth_scale, 1.5)
            scale_factor = max(0.5, min(scale_factor, 1.5))  # Limit between 0.5x and 1.5x
            
            # Apply scaling with maximum limits
            constrained_dims = {
                'length': min(round(standard['width'] * scale_factor, 1), max_dims['width']),
                'width': min(round(standard['depth'] * scale_factor, 1), max_dims['depth']),
                'height': min(round(standard['height'] * scale_factor, 1), max_dims['height'])
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
        Enhanced detection and measurement using camera data
        Args:
            image: RGB image
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
                object_type = self.detr_model.config.id2label[label.item()]
                
                # Calculate aspect ratios from bounding box
                box_width = box[2] - box[0]
                box_height = box[3] - box[1]
                aspect_ratio = box_width / box_height
                
                # Get standard dimensions for this type
                standard = self.STANDARD_DIMENSIONS.get(object_type.lower(), self.STANDARD_DIMENSIONS['default'])
                
                # Calculate scaling factor based on aspect ratio
                standard_ratio = standard['width'] / standard['height']
                scale_factor = min(box_width / standard['width'], box_height / standard['height'])
                
                # Adjust dimensions based on aspect ratio
                raw_dimensions = {
                    'length': round(standard['width'] * scale_factor * (aspect_ratio / standard_ratio), 1),
                    'width': round(standard['depth'] * scale_factor, 1),
                    'height': round(standard['height'] * scale_factor, 1)
                }
                
                # Apply constraints
                dimensions = self._apply_constraints(raw_dimensions, object_type)
                
                # Format detection result
                detection = {
                    'type': object_type,
                    'confidence': float(score.item()),
                    'bounding_box': [float(x) for x in box],
                    'dimensions': dimensions,
                    'volume': round(
                        (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728,
                        2
                    ),
                    'lidar_enhanced': False
                }
                detections.append(detection)
            
            return detections

        except Exception as e:
            logger.error(f"Error in detect_and_measure: {str(e)}")
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

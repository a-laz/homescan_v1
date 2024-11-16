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

    def detect_and_measure(self, image):
        """
        Main method to detect and measure furniture in an image.
        Returns a list of dictionaries containing detection and measurement information.
        """
        try:
            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            elif isinstance(image, str):
                image = cv2.imread(image)
            elif isinstance(image, bytes):
                nparr = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Ensure we have a valid numpy array
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Unable to convert image to numpy array. Image type: {type(image)}")
            
            # Ensure image is in the correct format and make a copy
            if len(image.shape) == 2:  # Grayscale to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # If BGR (OpenCV default), convert to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is contiguous and in the correct dtype
            image = np.ascontiguousarray(image, dtype=np.uint8)
            
            # Add debug logging
            logger.info(f"Image shape after conversion: {image.shape}")
            logger.info(f"Image dtype: {image.dtype}")
            logger.info(f"Image is contiguous: {image.flags['C_CONTIGUOUS']}")
            
            # Process image through DETR
            inputs = self.detr_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.detr_model(**inputs)
            
            # Define maximum reasonable dimensions (in inches)
            MAX_DIMENSIONS = {
                'bed': {'width': 84, 'depth': 84, 'height': 48},
                'dining table': {'width': 72, 'depth': 42, 'height': 30},
                'chair': {'width': 30, 'depth': 30, 'height': 45},
                'sofa': {'width': 84, 'depth': 40, 'height': 38},
                'tv': {'width': 65, 'depth': 4, 'height': 40},
                'person': {'width': 24, 'depth': 12, 'height': 72},
                'remote': {'width': 6, 'depth': 2, 'height': 8},
                'knife': {'width': 12, 'depth': 1, 'height': 2},
                'umbrella': {'width': 36, 'depth': 4, 'height': 36},
                'suitcase': {'width': 30, 'depth': 12, 'height': 20},
                'default': {'width': 48, 'depth': 24, 'height': 48}
            }

            # Assumed camera parameters if calibration fails
            ASSUMED_FOV = 60  # degrees
            ASSUMED_DISTANCE = 120  # inches (10 feet)
            
            # Process DETR results
            target_sizes = torch.tensor([image.shape[:2]])
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes
            )[0]
            
            formatted_detections = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                if score > 0.7:  # Confidence threshold
                    box = box.tolist()
                    label_name = self.detr_model.config.id2label[label.item()]
                    
                    # Get max dimensions for this object type
                    max_dims = MAX_DIMENSIONS.get(label_name.lower(), MAX_DIMENSIONS['default'])
                    
                    # Calculate pixel dimensions
                    pixel_width = box[2] - box[0]
                    pixel_height = box[3] - box[1]
                    
                    # Calculate real-world dimensions using FOV
                    image_width = image.shape[1]
                    fov_radians = math.radians(ASSUMED_FOV)
                    
                    # Calculate width based on FOV
                    object_width = (pixel_width / image_width) * (2 * ASSUMED_DISTANCE * math.tan(fov_radians / 2))
                    object_height = (pixel_height / image_width) * (2 * ASSUMED_DISTANCE * math.tan(fov_radians / 2))
                    
                    # Estimate depth based on object type
                    depth_ratio = {
                        'bed': 1.0,
                        'dining table': 0.6,
                        'chair': 1.0,
                        'sofa': 0.5,
                        'tv': 0.1,
                        'person': 0.5,
                        'remote': 0.3,
                        'knife': 0.1,
                        'umbrella': 0.1,
                        'suitcase': 0.4,
                        'default': 0.5
                    }
                    object_depth = object_width * depth_ratio.get(label_name.lower(), depth_ratio['default'])
                    
                    # Apply maximum constraints
                    dimensions = {
                        'length': min(object_width, max_dims['width']),
                        'width': min(object_depth, max_dims['depth']),
                        'height': min(object_height, max_dims['height'])
                    }
                    
                    # Round dimensions to one decimal place
                    dimensions = {k: round(v, 1) for k, v in dimensions.items()}
                    
                    # Calculate volume in cubic feet
                    volume = (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728
                    
                    formatted_detection = {
                        'type': label_name,
                        'confidence': float(score),
                        'dimensions': dimensions,
                        'volume': round(volume, 2),
                        'bounding_box': box,
                        'description': f"Detected {label_name}"
                    }
                    formatted_detections.append(formatted_detection)
            
            return formatted_detections

        except Exception as e:
            logger.error(f"Error in detect_and_measure: {str(e)}")
            logger.error(f"Image type: {type(image)}")
            if isinstance(image, np.ndarray):
                logger.error(f"Image shape: {image.shape}")
                logger.error(f"Image dtype: {image.dtype}")
            raise Exception(f"Error detecting and measuring furniture: {str(e)}")
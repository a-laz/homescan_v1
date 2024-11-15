import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import cv2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DimensionDetector:
    def __init__(self):
        self.reference_width = 8.5  # inches (standard letter paper width)
        self.reference_height = 11  # inches (standard letter paper height)
        
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
        
        # Define typical depth ratios for different furniture types
        self.DEPTH_RATIOS = {
            'chair': 0.9,
            'table': 1.2,
            'sofa': 0.8,
            'bookshelf': 0.4,
            'bed': 1.0,
            'dresser': 0.5,
            'default': 0.8
        }
        
    def calculate_dimensions(self, image, bbox, furniture_type='default', depth_map=None):
        """Calculate furniture dimensions using the bounding box."""
        pixel_width = bbox[2] - bbox[0]
        pixel_height = bbox[3] - bbox[1]
        
        # Log input parameters
        logger.debug(f"Calculating dimensions for {furniture_type}: pixel_width={pixel_width}, pixel_height={pixel_height}")
        
        # First try to use reference object
        reference_pixels = self.detect_reference_object(image)
        
        if reference_pixels and self._is_valid_reference(reference_pixels):
            logger.info("Using reference object for measurements")
            dimensions = self._calculate_with_reference(pixel_width, pixel_height, reference_pixels, furniture_type)
        else:
            logger.info("Using ratio-based estimation")
            dimensions = self.estimate_dimensions_from_ratios(pixel_width, pixel_height, furniture_type)
        
        # Apply furniture-specific constraints
        dimensions = self._apply_constraints(dimensions, furniture_type)
        
        logger.info(f"Final dimensions: {dimensions}")
        return dimensions

    def _is_valid_reference(self, reference_pixels):
        """Validate reference object measurements."""
        min_size = 10  # minimum pixel size
        max_size = 1000  # maximum pixel size
        expected_ratio = self.reference_width / self.reference_height  # ~0.773 for letter paper
        actual_ratio = reference_pixels['width'] / reference_pixels['height']
        
        return (min_size < reference_pixels['width'] < max_size and
                min_size < reference_pixels['height'] < max_size and
                0.7 < actual_ratio / expected_ratio < 1.3)

    def _calculate_with_reference(self, pixel_width, pixel_height, reference_pixels, furniture_type):
        """Calculate dimensions using reference object."""
        pixels_per_inch = reference_pixels['width'] / self.reference_width
        width = pixel_width / pixels_per_inch
        height = pixel_height / pixels_per_inch
        depth = width * self.DEPTH_RATIOS.get(furniture_type, self.DEPTH_RATIOS['default'])
        
        return {
            'length': round(width, 1),
            'width': round(depth, 1),
            'height': round(height, 1)
        }

    def detect_reference_object(self, image):
        """Detect a reference object (like a sheet of paper) in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_match = None
        best_score = float('inf')
        target_ratio = self.reference_width / self.reference_height
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Rectangle
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                ratio_diff = abs(aspect_ratio - target_ratio)
                
                if ratio_diff < best_score and w > 20 and h > 20:  # Minimum size threshold
                    best_score = ratio_diff
                    best_match = {'width': w, 'height': h}
        
        return best_match

    def estimate_dimensions_from_ratios(self, pixel_width, pixel_height, furniture_type='default'):
        """Estimate dimensions using typical furniture ratios."""
        # Camera parameters
        assumed_distance = 60  # inches (5 feet - reasonable default distance)
        assumed_fov = 60  # degrees
        image_width = 1920  # Update based on actual camera specs
        image_height = 1080
        
        # Calculate base dimensions
        width = (pixel_width / image_width) * assumed_distance * np.tan(np.radians(assumed_fov/2)) * 2
        height = (pixel_height / image_height) * assumed_distance * np.tan(np.radians(assumed_fov/2)) * 2
        depth = width * self.DEPTH_RATIOS.get(furniture_type, self.DEPTH_RATIOS['default'])
        
        return {
            'length': round(width, 1),
            'width': round(depth, 1),
            'height': round(height, 1)
        }

    def _apply_constraints(self, dimensions, furniture_type):
        """Apply furniture-specific constraints to dimensions."""
        max_dims = self.MAX_DIMENSIONS.get(furniture_type, self.MAX_DIMENSIONS['default'])
        
        return {
            'length': round(min(dimensions['length'], max_dims['width']), 1),
            'width': round(min(dimensions['width'], max_dims['depth']), 1),
            'height': round(min(dimensions['height'], max_dims['height']), 1)
        }

class EnhancedFurnitureDetector:
    def __init__(self):
        logger.info("Initializing EnhancedFurnitureDetector...")

        # Initialize DETR model for object detection
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        # Initialize dimension detector
        self.dimension_detector = DimensionDetector()
        
        # Initialize sentence transformer for text embeddings
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create knowledge base and FAISS index
        self.create_knowledge_base()
        self.embed_dimension = 384
        self.index = faiss.IndexFlatL2(self.embed_dimension)
        self.add_knowledge_to_index()

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
            "bookshelf": [
                "A bookshelf is a piece of furniture with horizontal shelves for storing books and items.",
                "Bookshelves can be freestanding, wall-mounted, or built-in units.",
                "Common materials include wood, metal, and engineered wood products.",
                "Standard bookshelves range from 36-72 inches in height and 24-36 inches in width.",
                "Adjustable shelving allows for customizable storage configurations."
            ],
            "bed": [
                "A bed is a piece of furniture used for sleeping and resting.",
                "Common bed sizes include twin, full, queen, and king dimensions.",
                "Beds typically consist of a frame, headboard, and support system.",
                "Platform beds eliminate the need for a box spring foundation.",
                "Storage beds incorporate drawers or lifting mechanisms for additional space."
            ],
            "dresser": [
                "A dresser is a chest of drawers for storing clothing and personal items.",
                "Dressers come in various configurations with multiple drawer layouts.",
                "Traditional dressers are made of wood with metal drawer slides.",
                "Standard dressers are 30-36 inches high and 60-72 inches wide.",
                "Modern dressers may include mirror attachments or jewelry storage."
            ]
        }

    def add_knowledge_to_index(self):
        """Add knowledge base items to FAISS index."""
        self.text_items = []
        for furniture, descriptions in self.knowledge_base.items():
            self.text_items.extend(descriptions)
        embeddings = self.sentence_model.encode(self.text_items)
        self.index.add(np.array(embeddings))

    def get_relevant_info(self, query, k=3):
        """Get relevant information from knowledge base."""
        query_vector = self.sentence_model.encode([query])
        distances, indices = self.index.search(query_vector, k)
        return [self.text_items[idx] for idx in indices[0]]

    def detect_and_measure(self, image):
        logger.info("Processing image for detection...")

        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        # Process image for object detection
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # Convert outputs
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.7
        )[0]
        
        detections = []
        
        # Process each detection
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.model.config.id2label[label.item()].lower()
            
            if label_name in self.dimension_detector.MAX_DIMENSIONS:
                # Get dimensions with furniture type
                dimensions = self.dimension_detector.calculate_dimensions(
                    np.array(image), box, furniture_type=label_name
                )
                
                # Calculate volume (convert to cubic feet)
                volume = (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728
                
                # Get relevant information
                info = self.get_relevant_info(f"information about {label_name}")
                
                detections.append({
                    "label": label_name,
                    "confidence": round(score.item(), 3),
                    "box": box,
                    "dimensions": dimensions,
                    "volume": round(volume, 2),
                    "description": info
                })
        
        return detections
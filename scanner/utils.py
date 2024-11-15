import cv2
import numpy as np
from .rag_detector import FurnitureRAGDetector
import math

class DimensionDetector:
    def __init__(self):
        # Initialize with a reference object's known dimensions (e.g., a standard sheet of paper)
        self.reference_width = 8.5  # inches (standard letter paper width)
        self.reference_height = 11  # inches (standard letter paper height)
        
    def calculate_dimensions(self, image, bbox, depth_map=None):
        """
        Calculate furniture dimensions using the bounding box and optional depth information.
        Uses reference object detection or depth estimation for more accurate measurements.
        """
        # Get pixel dimensions
        pixel_width = bbox[2] - bbox[0]
        pixel_height = bbox[3] - bbox[1]
        
        # Detect reference object (if present)
        reference_pixels = self.detect_reference_object(image)
        
        if reference_pixels:
            # Calculate ratio using reference object
            pixels_per_inch = reference_pixels['width'] / self.reference_width
            
            # Calculate actual dimensions
            width = pixel_width / pixels_per_inch
            height = pixel_height / pixels_per_inch
            
            # Estimate depth using perspective analysis or depth map
            if depth_map is not None:
                depth = self.estimate_depth_from_map(depth_map, bbox)
            else:
                depth = self.estimate_depth_from_perspective(width, height)
                
            return {
                'length': round(width, 1),  # inches
                'width': round(depth, 1),   # inches
                'height': round(height, 1)  # inches
            }
        else:
            # Fallback to approximate dimensions using typical furniture ratios
            return self.estimate_dimensions_from_ratios(pixel_width, pixel_height)
    
    def detect_reference_object(self, image):
        """
        Detect a reference object (like a sheet of paper) in the image
        Returns the pixel dimensions of the reference object if found
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular shapes with paper-like aspect ratio
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4:  # Rectangle
                (x, y, w, h) = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Check if aspect ratio matches standard paper (with some tolerance)
                if 0.7 <= aspect_ratio <= 0.9:  # US Letter aspect ratio ≈ 0.773
                    return {'width': w, 'height': h}
        
        return None

    def estimate_depth_from_perspective(self, width, height):
        """
        Estimate object depth using perspective analysis
        """
        # Use typical furniture depth ratios based on furniture type
        typical_ratios = {
            'chair': 0.8,    # depth typically 80% of width
            'table': 1.2,    # depth typically 120% of width
            'sofa': 0.9,     # depth typically 90% of width
            'bookshelf': 0.4 # depth typically 40% of width
        }
        
        # Default to average ratio if furniture type unknown
        return width * 0.8

    def estimate_dimensions_from_ratios(self, pixel_width, pixel_height):
        """
        Estimate dimensions using typical furniture ratios when reference object isn't available
        """
        # Assume a typical viewing distance and field of view
        assumed_distance = 120  # inches
        assumed_fov = 60       # degrees
        
        # Calculate approximate dimensions
        width = (pixel_width / 640) * assumed_distance * math.tan(math.radians(assumed_fov/2)) * 2
        height = (pixel_height / 480) * assumed_distance * math.tan(math.radians(assumed_fov/2)) * 2
        depth = width * 0.8    # Typical depth ratio
        
        return {
            'length': round(width, 1),
            'width': round(depth, 1),
            'height': round(height, 1)
        }

class EnhancedFurnitureDetector(FurnitureRAGDetector):
    def __init__(self):
        super().__init__()
        self.dimension_detector = DimensionDetector()
        
    def detect_and_measure(self, image):
        # Get basic detections from parent class
        detections = self.detect_and_describe(image)
        
        # Add dimensions for each detection
        for detection in detections:
            bbox = detection['box']
            dimensions = self.dimension_detector.calculate_dimensions(image, bbox)
            
            # Calculate volume in cubic feet
            volume = (dimensions['length'] * dimensions['width'] * dimensions['height']) / 1728
            
            # Add to detection data
            detection['dimensions'] = dimensions
            detection['volume'] = round(volume, 2)
            detection['position'] = self.estimate_position(bbox, image.shape)
            
        return detections
    
    def estimate_position(self, bbox, image_shape):
        """
        Estimate relative position in the room based on bounding box position
        """
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        
        return {
            'x': round(center_x / image_shape[1], 2),  # Relative x position (0-1)
            'y': round(center_y / image_shape[0], 2),  # Relative y position (0-1)
            'z': round(1 - (bbox[3] / image_shape[0]), 2)  # Estimated depth based on vertical position
        }
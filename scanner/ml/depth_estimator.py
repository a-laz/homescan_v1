import torch
import numpy as np
from typing import Dict, List
import cv2

class DepthEstimator:
    def __init__(self):
        # Initialize MiDaS for depth estimation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        self.model.to(self.device)
        self.model.eval()
        
        # Transform for MiDaS input
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
        
        # Optical flow parameters for Farneback algorithm
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        self.prev_frame = None
        self.prev_depth = None
        
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """Estimate depth for a single frame"""
        # Transform input for model
        input_batch = self.transform(frame).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Apply temporal smoothing if we have previous frames
        if self.prev_frame is not None and self.prev_depth is not None:
            depth_map = self._apply_temporal_smoothing(frame, depth_map)
            
        # Update previous frame info
        self.prev_frame = frame.copy()
        self.prev_depth = depth_map.copy()
        
        return depth_map
    
    def _apply_temporal_smoothing(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Apply temporal smoothing using Farneback optical flow"""
        # Calculate optical flow using Farneback algorithm
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(self.prev_frame, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
            None,
            **self.flow_params
        )
        
        # Warp previous depth map according to flow
        h, w = frame.shape[:2]
        flow_map = np.column_stack((
            flow[..., 0].flatten(),
            flow[..., 1].flatten()
        ))
        
        warped_points = np.mgrid[0:h, 0:w].reshape(2, -1).T + flow_map
        warped_points = warped_points.reshape(h, w, 2)
        
        warped_depth = cv2.remap(
            self.prev_depth,
            warped_points[..., 0].astype(np.float32),
            warped_points[..., 1].astype(np.float32),
            cv2.INTER_LINEAR
        )
        
        # Blend current and warped depth maps
        alpha = 0.7  # Weight for current frame
        smoothed_depth = alpha * depth_map + (1 - alpha) * warped_depth
        
        return smoothed_depth
    
    def get_point_cloud(self, frame: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """Convert depth map to 3D point cloud"""
        h, w = depth_map.shape
        
        # Create mesh grid of coordinates
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Assume a simple pinhole camera model
        fx = fy = w  # Approximate focal length
        cx, cy = w/2, h/2  # Principal point at image center
        
        # Convert image coordinates to 3D points
        z = depth_map.reshape(-1)
        x = (x.reshape(-1) - cx) * z / fx
        y = (y.reshape(-1) - cy) * z / fy
        
        # Stack coordinates
        points = np.stack([x, y, z], axis=1)
        
        return points
    
    def estimate_depth_mvs(self, frames: List[np.ndarray]) -> np.ndarray:
        """Estimate depth using multi-view stereo when multiple frames available"""
        try:
            # Process first frame as baseline
            depth_map = self.estimate_depth(frames[0])
            
            if len(frames) > 1:
                # Track keypoints between frames
                orb = cv2.ORB_create()
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                for next_frame in frames[1:]:
                    # Detect keypoints and compute descriptors
                    kp1, des1 = orb.detectAndCompute(frames[0], None)
                    kp2, des2 = orb.detectAndCompute(next_frame, None)
                    
                    # Match keypoints
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Get matched point coordinates
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
                    
                    # Calculate fundamental matrix
                    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
                    
                    # Triangulate points
                    if F is not None:
                        # Use triangulated points to refine depth
                        refined_depth = self._refine_depth_with_mvs(depth_map, pts1, pts2, F)
                        depth_map = refined_depth
            
            return depth_map
            
        except Exception as e:
            print(f"Error in MVS depth estimation: {str(e)}")
            return self.estimate_depth(frames[0])  # Fallback to single view
    
    def _refine_depth_with_mvs(self, depth_map: np.ndarray, pts1: np.ndarray, 
                              pts2: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Refine depth map using multi-view stereo correspondences"""
        # Create interpolation grid
        h, w = depth_map.shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate disparities from matched points
        disparities = np.linalg.norm(pts2 - pts1, axis=1)
        
        # Interpolate disparities to full resolution
        from scipy.interpolate import griddata
        full_disparities = griddata(
            pts1, 
            disparities, 
            (grid_x, grid_y), 
            method='cubic',
            fill_value=np.mean(disparities)
        )
        
        # Blend with original depth map
        alpha = 0.7  # Weight for original depth
        refined_depth = alpha * depth_map + (1 - alpha) * full_disparities
        
        return refined_depth

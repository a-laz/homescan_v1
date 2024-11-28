import numpy as np
from collections import defaultdict
import torch
from typing import List, Dict, Tuple

class Track:
    def __init__(self, detection, track_id):
        self.track_id = track_id
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.class_id = detection['class']
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = [self.bbox]
        
    def update(self, detection):
        self.bbox = detection['bbox']
        self.confidence = detection['confidence']
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.history.append(self.bbox)
        if len(self.history) > 30:  # Keep last 30 frames
            self.history.pop(0)

class SimpleTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.frame_count = 0
        self.next_id = 0
        
    def _calculate_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        iou = intersection / (area1 + area2 - intersection + 1e-6)
        return iou
    
    def _hungarian_matching(self, cost_matrix):
        """Simple greedy matching as fallback for Hungarian algorithm"""
        rows, cols = cost_matrix.shape
        used_cols = set()
        used_rows = set()
        matches = []
        
        # Sort all costs by value
        costs = []
        for i in range(rows):
            for j in range(cols):
                costs.append((cost_matrix[i, j], i, j))
        costs.sort()
        
        # Greedy assignment
        for cost, row, col in costs:
            if row not in used_rows and col not in used_cols:
                matches.append((row, col))
                used_rows.add(row)
                used_cols.add(col)
                
        return matches
    
    def update(self, detections):
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Handle first frame
        if len(self.tracks) == 0:
            for det in detections:
                self.tracks.append(Track(det, self.next_id))
                self.next_id += 1
            return self.get_active_tracks()
        
        # Calculate IoU between all tracks and detections
        cost_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - self._calculate_iou(track.bbox, det['bbox'])
        
        # Match detections to tracks
        matches = self._hungarian_matching(cost_matrix)
        
        # Update matched tracks
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))
        
        for track_idx, det_idx in matches:
            if cost_matrix[track_idx, det_idx] < 1 - self.iou_threshold:
                self.tracks[track_idx].update(detections[det_idx])
                unmatched_tracks.remove(track_idx)
                unmatched_detections.remove(det_idx)
        
        # Handle unmatched detections
        for det_idx in unmatched_detections:
            self.tracks.append(Track(detections[det_idx], self.next_id))
            self.next_id += 1
        
        # Update unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].time_since_update += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update < self.max_age]
        
        return self.get_active_tracks()
    
    def get_active_tracks(self):
        """Return tracks that meet minimum hit requirement"""
        active_tracks = []
        for track in self.tracks:
            if track.hits >= self.min_hits and track.time_since_update == 0:
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence,
                    'class': track.class_id,
                    'age': track.age,
                    'hits': track.hits
                })
        return active_tracks

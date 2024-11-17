# scanner/models.py
from django.db import models
from django.contrib.auth.models import User
from django.db.models import Avg

class Room(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    length = models.FloatField(null=True, blank=True)
    width = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    volume = models.FloatField(null=True, blank=True)
    last_scan = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name

    @property
    def dimensions(self):
        """Return dimensions as a dictionary for compatibility"""
        return {
            'length': self.length,
            'width': self.width,
            'height': self.height
        }

    def total_volume(self):
        """Calculate room volume in cubic feet"""
        return round((self.length * self.width * self.height) / 1728, 2)  # Convert cubic inches to cubic feet

    def get_dimensions(self):
        """Get room dimensions based on furniture detections"""
        detections = self.furnituredetection_set.all()
        if not detections:
            return {'length': 0, 'width': 0, 'height': 0}
        
        # Get maximum dimensions from all detections
        max_length = max((d.length for d in detections), default=0)
        max_width = max((d.width for d in detections), default=0)
        max_height = max((d.height for d in detections), default=0)
        
        return {
            'length': round(max_length, 1),
            'width': round(max_width, 1),
            'height': round(max_height, 1)
        }

    def update_dimensions(self, dimensions):
        self.length = dimensions['length']
        self.width = dimensions['width']
        self.height = dimensions['height']
        self.volume = dimensions['volume']
        self.save()

    def get_summary(self):
        """Get room summary including furniture counts and volumes"""
        detections = self.furnituredetection_set.all()
        
        # Count furniture types
        furniture_types = {}
        total_volume = 0
        
        for detection in detections:
            # Count furniture types
            if detection.furniture_type in furniture_types:
                furniture_types[detection.furniture_type] += 1
            else:
                furniture_types[detection.furniture_type] = 1
            
            # Sum volumes
            total_volume += detection.volume
        
        return {
            'furniture_count': len(detections),
            'total_volume': round(total_volume, 2),
            'furniture_types': furniture_types
        }

    def get_recent_detections(self):
        """Get recent detections for this room"""
        return self.furnituredetection_set.all().order_by('-timestamp')[:50]

class FurnitureDetection(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    furniture_type = models.CharField(max_length=50)
    length = models.FloatField(default=0.0)
    width = models.FloatField(default=0.0)
    height = models.FloatField(default=0.0)
    volume = models.FloatField(default=0.0)
    confidence = models.FloatField(default=0.0)
    image_data = models.TextField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    video_data = models.FileField(upload_to='videos/', null=True, blank=True)
    frame_number = models.IntegerField(null=True, blank=True)

    def __str__(self):
        return f"{self.furniture_type} in {self.room.name}"

    class Meta:
        ordering = ['-timestamp']

class ScanSession(models.Model):
    room = models.ForeignKey(Room, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    scan_duration = models.IntegerField()  # seconds
    detection_count = models.IntegerField()
    average_confidence = models.FloatField()
    
    def calculate_metrics(self):
        detections = FurnitureDetection.objects.filter(scan_session=self)
        self.detection_count = detections.count()
        self.average_confidence = detections.aggregate(Avg('confidence'))['confidence__avg']
        self.save()
# scanner/models.py
from django.db import models
from django.contrib.auth.models import User

class Room(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    length = models.FloatField(default=0.0)
    width = models.FloatField(default=0.0)
    height = models.FloatField(default=0.0)
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

    def __str__(self):
        return f"{self.furniture_type} in {self.room.name}"

    class Meta:
        ordering = ['-timestamp']
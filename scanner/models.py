# scanner/models.py
from django.db import models
from django.contrib.auth.models import User

class Room(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    dimensions = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

    def total_volume(self):
        if self.dimensions:
            return (
                self.dimensions.get('length', 0) *
                self.dimensions.get('width', 0) *
                self.dimensions.get('height', 0)
            ) / 1728  # Convert cubic inches to cubic feet
        return 0

class FurnitureDetection(models.Model):
    room = models.ForeignKey('Room', on_delete=models.CASCADE)
    furniture_type = models.CharField(max_length=100)
    confidence = models.FloatField()
    dimensions = models.JSONField()
    volume = models.FloatField()
    image = models.TextField(null=True, blank=True)
    position = models.JSONField(null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.furniture_type} in {self.room.name}"
# scanner/serializers.py
from rest_framework import serializers
from .models import Room, FurnitureDetection

class RoomSerializer(serializers.ModelSerializer):
    total_volume = serializers.FloatField(read_only=True)
    
    class Meta:
        model = Room
        fields = ['id', 'name', 'description', 'dimensions', 'created_at', 'total_volume']
        read_only_fields = ['created_at']

    def validate_dimensions(self, value):
        required_keys = ['length', 'width', 'height']
        if not all(key in value for key in required_keys):
            raise serializers.ValidationError(
                "Dimensions must include length, width, and height"
            )
        for key in required_keys:
            if not isinstance(value[key], (int, float)) or value[key] <= 0:
                raise serializers.ValidationError(
                    f"{key} must be a positive number"
                )
        return value

class FurnitureDetectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = FurnitureDetection
        fields = [
            'id', 'room', 'furniture_type', 'confidence', 
            'dimensions', 'volume', 'image', 'timestamp'
        ]
        read_only_fields = ['timestamp']

    def validate_confidence(self, value):
        if not 0 <= value <= 1:
            raise serializers.ValidationError(
                "Confidence must be between 0 and 1"
            )
        return value
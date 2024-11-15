# scanner/api_views.py
from rest_framework import generics, permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from django.shortcuts import get_object_or_404
from .models import Room, FurnitureDetection
from .serializers import RoomSerializer, FurnitureDetectionSerializer

class RoomList(generics.ListCreateAPIView):
    serializer_class = RoomSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Room.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

class RoomDetail(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = RoomSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Room.objects.filter(user=self.request.user)

class DetectionList(generics.ListCreateAPIView):
    serializer_class = FurnitureDetectionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return FurnitureDetection.objects.filter(room__user=self.request.user)

class DetectionDetail(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = FurnitureDetectionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return FurnitureDetection.objects.filter(room__user=self.request.user)

class ProcessScan(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        room_id = request.data.get('room_id')
        room = get_object_or_404(Room, id=room_id, user=request.user)
        
        # Process the scan logic here
        return Response({'status': 'success'})

class RoomStats(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, room_id):
        room = get_object_or_404(Room, id=room_id, user=request.user)
        detections = FurnitureDetection.objects.filter(room=room)
        
        stats = {
            'total_items': detections.count(),
            'total_volume': sum(d.volume for d in detections),
            'furniture_types': detections.values('furniture_type').distinct().count(),
            'furniture_breakdown': {}
        }
        
        for detection in detections:
            if detection.furniture_type not in stats['furniture_breakdown']:
                stats['furniture_breakdown'][detection.furniture_type] = 0
            stats['furniture_breakdown'][detection.furniture_type] += 1
        
        return Response(stats)

class ScanHistory(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request, room_id):
        room = get_object_or_404(Room, id=room_id, user=request.user)
        detections = FurnitureDetection.objects.filter(room=room).order_by('-timestamp')
        serializer = FurnitureDetectionSerializer(detections, many=True)
        return Response(serializer.data)
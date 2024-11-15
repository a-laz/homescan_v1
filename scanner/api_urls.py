# scanner/api_urls.py
from django.urls import path
from . import api_views

app_name = 'scanner_api'

urlpatterns = [
    # Room endpoints
    path('rooms/', api_views.RoomList.as_view(), name='room-list'),
    path('rooms/<int:pk>/', api_views.RoomDetail.as_view(), name='room-detail'),
    
    # Scanning endpoints
    path('scan/', api_views.ProcessScan.as_view(), name='process-scan'),
    path('detections/', api_views.DetectionList.as_view(), name='detection-list'),
    path('detections/<int:pk>/', api_views.DetectionDetail.as_view(), name='detection-detail'),
    
    # Statistics endpoints
    path('rooms/<int:room_id>/stats/', api_views.RoomStats.as_view(), name='room-stats'),
    path('rooms/<int:room_id>/history/', api_views.ScanHistory.as_view(), name='scan-history'),
]
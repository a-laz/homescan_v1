from django.urls import path
from . import views

app_name = 'scanner'

urlpatterns = [
    path('', views.home, name='home'),
    path('room/<int:room_id>/', views.room_detail, name='room_detail'),
    path('room/<int:room_id>/scan/', views.scan_room, name='scan_room'),
    path('room/add/', views.add_room, name='add_room'),
    path('room/<int:room_id>/edit/', views.edit_room, name='edit_room'),
    path('api/process-frame/', views.process_frame, name='process_frame'),
]
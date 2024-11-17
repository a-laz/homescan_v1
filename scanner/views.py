# scanner/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from .models import Room, FurnitureDetection
from .forms import RoomForm
from .ml.detector import EnhancedFurnitureDetector
import json
import base64
import cv2
import numpy as np
from datetime import datetime
import logging
from PIL import Image
import io
import os
import uuid
from django.views.decorators.http import require_http_methods
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Initialize the detector
detector = EnhancedFurnitureDetector()

# Configure logging
logger = logging.getLogger(__name__)

def home(request):
    """Home page view showing all rooms."""
    if request.user.is_authenticated:
        rooms = Room.objects.filter(user=request.user).order_by('-created_at')
        total_scans = FurnitureDetection.objects.filter(room__user=request.user).count()
        context = {
            'rooms': rooms,
            'total_scans': total_scans,
            'room_form': RoomForm()
        }
    else:
        context = {}
    return render(request, 'scanner/home.html', context)

@login_required
def room_detail(request, room_id):
    """Detailed view of a single room."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    detections = FurnitureDetection.objects.filter(room=room).order_by('-timestamp')[:50]
    
    context = {
        'room': room,
        'detections': detections,
    }
    return render(request, 'scanner/room_detail.html', context)

@login_required
def scan_room(request, room_id):
    """Room scanning interface."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    context = {
        'room': room,
        'recent_detections': FurnitureDetection.objects.filter(room=room).order_by('-timestamp')[:5]
    }
    return render(request, 'scanner/scan.html', context)

@login_required
def scan_history(request, room_id):
    """View scan history for a room."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    detections = FurnitureDetection.objects.filter(room=room).order_by('-timestamp')
    
    # Group detections by date
    grouped_detections = {}
    for detection in detections:
        date = detection.timestamp.date()
        if date not in grouped_detections:
            grouped_detections[date] = []
        grouped_detections[date].append(detection)
    
    context = {
        'room': room,
        'grouped_detections': grouped_detections
    }
    return render(request, 'scanner/history.html', context)

@csrf_exempt
@login_required
def process_frame(request):
    """Process uploaded frame from scanner."""
    try:
        result = {
            'status': 'success',
            'detections': []
        }

        data = json.loads(request.body)
        room_id = data.get('room_id')
        image_data = data.get('image')
        lidar_data = data.get('lidar_points')  # LiDAR point cloud if available
        tof_data = data.get('tof_data')  # ToF depth data if available
        
        if image_data and room_id:
            room = get_object_or_404(Room, id=room_id)
            
            try:
                # Convert base64 image to numpy array
                image_format, image_str = image_data.split(';base64,')
                image_bytes = base64.b64decode(image_str)
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                
                # Convert sensor data if provided
                lidar_points = None
                if lidar_data:
                    lidar_points = np.array(json.loads(lidar_data))
                
                tof_array = None
                if tof_data:
                    tof_bytes = base64.b64decode(tof_data.split(',')[1])
                    tof_array = np.frombuffer(tof_bytes, dtype=np.float32).reshape(-1, 1)
                
                # Get detections using available sensor data
                detections = detector.detect_and_measure(
                    image_np,
                    lidar_points=lidar_points,
                    tof_data=tof_array
                )
                logger.info(f"Detections found: {len(detections)}")

            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                raise ValueError(f"Invalid image data: {str(e)}")

            # Save detections to database
            for detection in detections:
                dims = detection.get('dimensions', {})
                if dims:
                    # Calculate volume
                    volume = (dims.get('length', 0) * 
                            dims.get('width', 0) * 
                            dims.get('height', 0)) / 1728

                    # Save to database
                    FurnitureDetection.objects.create(
                        room=room,
                        furniture_type=detection.get('type', '')[:50],
                        length=round(float(dims.get('length', 0)), 1),
                        width=round(float(dims.get('width', 0)), 1),
                        height=round(float(dims.get('height', 0)), 1),
                        volume=round(float(volume), 2),
                        confidence=round(float(detection.get('confidence', 0)), 2),
                        image_data=image_data if len(image_data) < 100000 else None
                    )

            # Process image for response
            if image_np.shape[0] > 400 or image_np.shape[1] > 400:
                ratio = 400 / max(image_np.shape[0], image_np.shape[1])
                new_size = (int(image_np.shape[1] * ratio), int(image_np.shape[0] * ratio))
                image_np = cv2.resize(image_np, new_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Convert back to base64 for response
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 
                                   [cv2.IMWRITE_JPEG_QUALITY, 50])
            image_data = base64.b64encode(buffer).decode()
            
            if len(image_data) > 100000:
                new_size = (int(image_np.shape[1] * 0.7), int(image_np.shape[0] * 0.7))
                image_np = cv2.resize(image_np, new_size, interpolation=cv2.INTER_LANCZOS4)
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR), 
                                       [cv2.IMWRITE_JPEG_QUALITY, 40])
                image_data = base64.b64encode(buffer).decode()
            
            result['image'] = f'data:image/jpeg;base64,{image_data}'

        response = JsonResponse(result)
        if len(response.content) > 500000:
            del result['image']
            result['status'] = 'partial'
            result['message'] = 'Image data omitted due to size constraints'
            response = JsonResponse(result)

        return response

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@login_required
def add_room(request):
    if request.method == 'POST':
        try:
            room = Room.objects.create(
                user=request.user,
                name=request.POST['name'],
                description=request.POST['description'],
                length=float(request.POST['length']),
                width=float(request.POST['width']),
                height=float(request.POST['height'])
            )
            return redirect('scanner:room_detail', room_id=room.id)
        except (KeyError, ValueError) as e:
            messages.error(request, 'Invalid room data provided.')
            return redirect('scanner:home')
    return redirect('scanner:home')

@login_required
def edit_room(request, room_id):
    """Edit existing room."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    
    if request.method == 'POST':
        form = RoomForm(request.POST, instance=room)
        if form.is_valid():
            form.save()
            messages.success(request, 'Room updated successfully!')
            return redirect('scanner:room_detail', room_id=room.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RoomForm(instance=room)
    
    return render(request, 'scanner/room_form.html', {
        'form': form,
        'room': room,
        'is_edit': True
    })

@login_required
def delete_room(request, room_id):
    """Delete room."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    
    if request.method == 'POST':
        room.delete()
        messages.success(request, 'Room deleted successfully!')
        return redirect('scanner:home')
    
    return render(request, 'scanner/delete_room.html', {'room': room})

@login_required
def furniture_detail(request, detection_id):
    """Detailed view of a furniture detection."""
    detection = get_object_or_404(FurnitureDetection, id=detection_id, room__user=request.user)
    context = {
        'detection': detection,
        'room': detection.room
    }
    return render(request, 'scanner/furniture_detail.html', context)

@login_required
def download_report(request, room_id):
    """Generate and download room report."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    detections = FurnitureDetection.objects.filter(room=room).order_by('-timestamp')
    
    # Create report logic here
    context = {
        'room': room,
        'detections': detections,
        'generated_at': datetime.now()
    }
    return render(request, 'scanner/report.html', context)

@login_required
def room_statistics(request, room_id):
    """View room statistics."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    detections = FurnitureDetection.objects.filter(room=room)
    
    # Calculate statistics
    stats = {
        'total_items': detections.count(),
        'total_volume': sum(d.volume for d in detections),
        'furniture_types': detections.values('furniture_type').distinct().count(),
        'latest_scan': detections.order_by('-timestamp').first(),
        'furniture_breakdown': {
            ftype: detections.filter(furniture_type=ftype).count()
            for ftype in detections.values_list('furniture_type', flat=True).distinct()
        }
    }
    
    context = {
        'room': room,
        'stats': stats
    }
    return render(request, 'scanner/statistics.html', context)

@require_http_methods(["POST"])
def process_video(request):
    try:
        video_file = request.FILES.get('video')
        if not video_file:
            return JsonResponse({'error': 'No video file provided'}, status=400)
        
        # Save video temporarily
        temp_path = f'/tmp/upload_{uuid.uuid4()}.mp4'
        with open(temp_path, 'wb+') as destination:
            for chunk in video_file.chunks():
                destination.write(chunk)
        
        # Process video
        detector = EnhancedFurnitureDetector()
        detections = detector.process_video(temp_path)
        
        # Calculate room dimensions
        room_dims = detector.calculate_room_dimensions(detections)
        
        # Create summary
        summary = detector.summarize_detections(detections)
        
        # Update room
        room = Room.objects.get(id=room_id)
        room.update_dimensions(room_dims)
        
        # Clean up
        os.remove(temp_path)
        
        # Send progress updates
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"room_{room_id}",
            {
                "type": "processing_update",
                "progress": progress
            }
        )
        
        return JsonResponse({
            'success': True,
            'detections': detections,
            'room_dimensions': room_dims,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def scan_room_video(request, room_id):
    """Video scanning interface."""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    context = {
        'room': room,
        'recent_detections': FurnitureDetection.objects.filter(room=room).order_by('-timestamp')[:5]
    }
    return render(request, 'scanner/scan_video.html', context)

@login_required
def export_room_data(request, room_id):
    """Export room data as PDF or CSV"""
    room = get_object_or_404(Room, id=room_id, user=request.user)
    detections = FurnitureDetection.objects.filter(room=room)
    
    if request.GET.get('format') == 'pdf':
        # Generate PDF report
        response = generate_pdf_report(room, detections)
    else:
        # Generate CSV
        response = generate_csv_report(room, detections)
    
    return response

@login_required
def compare_rooms(request):
    """Compare dimensions and contents of different rooms"""
    rooms = Room.objects.filter(user=request.user)
    context = {
        'rooms': rooms,
        'comparisons': calculate_room_comparisons(rooms)
    }
    return render(request, 'scanner/compare_rooms.html', context)
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
        
        if image_data and room_id:
            room = get_object_or_404(Room, id=room_id)
            
            # Process image
            image_format, image_str = image_data.split(';base64,')
            image_bytes = base64.b64decode(image_str)
            image = Image.open(io.BytesIO(image_bytes))

            # Get detections without LiDAR for now
            detections = detector.detect_and_measure(image)
            logger.info(f"Detections found: {len(detections)}")

            # Save detections to database
            for detection in detections:
                dims = detection.get('dimensions', {})
                if dims:
                    # Calculate volume
                    volume = (dims.get('length', 0) * 
                            dims.get('width', 0) * 
                            dims.get('height', 0)) / 1728  # Convert to cubic feet

                    # Save to database
                    FurnitureDetection.objects.create(
                        room=room,
                        furniture_type=detection.get('type', '')[:50],
                        length=round(float(dims.get('length', 0)), 1),
                        width=round(float(dims.get('width', 0)), 1),
                        height=round(float(dims.get('height', 0)), 1),
                        volume=round(float(volume), 2),
                        confidence=round(float(detection.get('confidence', 0)), 2),
                        image_data=image_data if len(image_data) < 100000 else None  # Only save small images
                    )

            # Optimize detections for response
            for detection in detections:
                optimized_detection = {
                    'type': detection.get('type', '')[:50],
                    'confidence': round(float(detection.get('confidence', 0)), 2),
                }
                dims = detection.get('dimensions', {})
                if dims:
                    optimized_detection['dimensions'] = {
                        k: round(float(v), 1) 
                        for k, v in dims.items()
                        if isinstance(v, (int, float)) and v > 0
                    }
                result['detections'].append(optimized_detection)

            # Process image for response
            max_size = 400
            if image.width > max_size or image.height > max_size:
                ratio = max_size / max(image.width, image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image = image.convert('RGB')
            image.save(buffer, format='JPEG', quality=50, optimize=True)
            
            image_data = base64.b64encode(buffer.getvalue()).decode()
            if len(image_data) > 100000:
                image = image.resize((int(image.width * 0.7), int(image.height * 0.7)))
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=40, optimize=True)
                image_data = base64.b64encode(buffer.getvalue()).decode()
            
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
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
    detections = FurnitureDetection.objects.filter(room=room).order_by('-timestamp')
    total_volume = sum(detection.volume for detection in detections)
    
    context = {
        'room': room,
        'detections': detections,
        'total_volume': round(total_volume, 2)
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
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            room_id = data.get('room_id')
            image_data = data.get('image')
            lidar_data = data.get('lidar')
            
            if not room_id or not image_data:
                return JsonResponse({
                    'status': 'error',
                    'message': 'Missing room_id or image data'
                })
            
            # Get room
            room = get_object_or_404(Room, id=room_id, user=request.user)
            
            # Convert base64 to image
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                
                # Decode base64 to bytes
                image_bytes = base64.b64decode(image_data)
                
                # Convert to numpy array
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    raise ValueError("Failed to decode image")
                
                logger.info(f"Image shape before detection: {image.shape}")
                logger.info(f"Image dtype before detection: {image.dtype}")
                
            except Exception as e:
                return JsonResponse({
                    'status': 'error',
                    'message': f"Error processing image: {str(e)}"
                })
            
            # Process LiDAR data if available
            if lidar_data:
                lidar_points = np.array(lidar_data['points'])
                camera_matrix = np.array(data.get('camera_matrix'))
                extrinsics = np.array(data.get('extrinsics'))
                
                detections = detector.detect_and_measure(
                    image,
                    lidar_data=lidar_points,
                    camera_matrix=camera_matrix,
                    extrinsics=extrinsics
                )
            else:
                detections = detector.detect_and_measure(image)
                
            logger.info(f"Detections found: {len(detections)}")

            if not detections:
                return JsonResponse({
                    'status': 'success',
                    'detections': [],
                    'message': 'No furniture detected in frame'
                })
            
            # Draw boxes and labels
            for detection in detections:
                box = detection["bounding_box"]  # Updated to match new detector output
                dimensions = detection["dimensions"]
                volume = detection["volume"]
                label = f"{detection['type']} ({volume} cu ft)"  # Updated to match new detector output
                
                # Draw bounding box
                cv2.rectangle(
                    image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2
                )
                
                # Draw label with dimensions
                label_text = f"{label}\n{dimensions['length']}\"x{dimensions['width']}\"x{dimensions['height']}\""
                y = int(box[1] - 10)
                for i, line in enumerate(label_text.split('\n')):
                    y = y + 20
                    cv2.putText(
                        image,
                        line,
                        (int(box[0]), y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )
            
            # Convert back to base64
            _, buffer = cv2.imencode('.jpg', image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Save detections to database
            for detection in detections:
                FurnitureDetection.objects.create(
                    room=room,
                    furniture_type=detection['type'],  # Updated to match new detector output
                    confidence=detection['confidence'],
                    dimensions=detection['dimensions'],
                    volume=float(detection['volume']),  # Convert string to float
                    image=f'data:image/jpeg;base64,{img_base64}',
                )
            
            return JsonResponse({
                'status': 'success',
                'detections': detections,
                'image': f'data:image/jpeg;base64,{img_base64}'
            })
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'message': str(e)
            })
    
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

@login_required
def add_room(request):
    """Add new room."""
    if request.method == 'POST':
        form = RoomForm(request.POST)
        if form.is_valid():
            room = form.save(commit=False)
            room.user = request.user
            room.save()
            messages.success(request, 'Room added successfully!')
            return redirect('scanner:room_detail', room_id=room.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = RoomForm()
    
    return render(request, 'scanner/room_form.html', {'form': form})

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
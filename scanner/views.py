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
    
    # Get room summary
    summary = room.get_summary()
    
    context = {
        'room': room,
        'detections': detections,
        'summary': summary,
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

def process_image_data(image_data):
    """Convert base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
            
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        logger.info(f"Processing image: shape={image.shape}, dtype={image.dtype}")
        return image
        
    except Exception as e:
        logger.error(f"Error processing image data: {str(e)}")
        raise

@csrf_exempt
@login_required
def process_frame(request):
    try:
        data = json.loads(request.body)
        room_id = data.get('room_id')
        image_data = data.get('image')
        
        if image_data and room_id:
            room = get_object_or_404(Room, id=room_id)
            
            # Process image and get detections with tracking info
            image_np = process_image_data(image_data)
            detections = detector.detect_and_measure(image_np)
            
            logger.info(f"Detections found: {len(detections)}")
            
            # Save detections to database
            for detection in detections:
                try:
                    bbox = detection.get('bbox', [])
                    class_id = detection.get('class')
                    track_id = detection.get('track_id', None)  # New field from tracker
                    
                    if len(bbox) == 4:
                        # Calculate dimensions from bounding box
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        
                        # Get the cropped image of the detection
                        cropped_image = image_np[y1:y2, x1:x2]
                        
                        # Convert cropped image to base64
                        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR))
                        cropped_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        # Calculate real-world dimensions (in inches)
                        pixel_width = x2 - x1
                        pixel_height = y2 - y1
                        
                        # Convert pixels to inches (approximate)
                        PIXELS_PER_INCH = 10  # This needs calibration
                        width_inches = pixel_width / PIXELS_PER_INCH
                        height_inches = pixel_height / PIXELS_PER_INCH
                        depth_inches = (width_inches + height_inches) / 2  # Estimate depth
                        
                        # Create detection record
                        detection_obj = FurnitureDetection.objects.create(
                            room=room,
                            furniture_type=detector.get_class_name(class_id),
                            length=width_inches,
                            width=depth_inches,
                            height=height_inches,
                            volume=(width_inches * depth_inches * height_inches) / 1728,  # Convert to cubic feet
                            confidence=float(detection.get('confidence', 0.0)),
                            image_data=cropped_base64,
                            track_id=track_id  # Store the track ID if you want to reference it later
                        )
                        
                        logger.info(f"""
                        Saved detection:
                        - Type: {detector.get_class_name(class_id)}
                        - Track ID: {track_id}
                        - Dimensions: {width_inches:.1f}" × {depth_inches:.1f}" × {height_inches:.1f}"
                        - Confidence: {detection.get('confidence', 0.0):.2f}
                        """)
                
                except Exception as e:
                    logger.error(f"Error processing detection: {str(e)}")
                    continue
            
            return JsonResponse({
                'status': 'success',
                'detections': detections
            })
            
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return JsonResponse({
            'status': 'error',
            'message': f'Invalid image data: {str(e)}'
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
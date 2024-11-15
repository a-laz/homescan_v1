# scanner/error_views.py
from django.shortcuts import render

def custom_404(request, exception):
    """Custom 404 error page."""
    context = {
        'error_code': '404',
        'error_message': 'Page not found',
        'error_details': 'The page you are looking for might have been removed, had its name changed, or is temporarily unavailable.'
    }
    return render(request, 'scanner/errors/error.html', context, status=404)

def custom_500(request):
    """Custom 500 error page."""
    context = {
        'error_code': '500',
        'error_message': 'Server error',
        'error_details': 'Something went wrong on our end. Please try again later.'
    }
    return render(request, 'scanner/errors/error.html', context, status=500)

def custom_403(request, exception):
    """Custom 403 error page."""
    context = {
        'error_code': '403',
        'error_message': 'Access denied',
        'error_details': 'You do not have permission to access this resource.'
    }
    return render(request, 'scanner/errors/error.html', context, status=403)
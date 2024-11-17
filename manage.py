#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import logging
import traceback
import pdb
import signal
from datetime import datetime

# Configure logging with more detailed format for migrations
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'debug-{datetime.now().strftime("%Y%m%d-%H%M%S")}.log')
    ]
)

logger = logging.getLogger(__name__)

def debug_handler(sig, frame):
    """Handler to enter pdb on SIGUSR1"""
    pdb.set_trace()

def setup_migration_debugging():
    """Setup specific debugging for migrations"""
    if 'migrate' in sys.argv:
        # Enable detailed SQL logging
        os.environ['DJANGO_LOG_LEVEL'] = 'DEBUG'
        os.environ['DJANGO_SETTINGS_MODULE'] = 'homescan.settings'
        
        # Log all SQL queries
        os.environ['DJANGO_DB_LOG_SQL'] = 'True'
        
        logger.info("Migration debugging enabled")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        
        # Log migration-specific information
        logger.info(f"Migration command: {' '.join(sys.argv)}")
        
        try:
            import django
            logger.info(f"Django version: {django.get_version()}")
        except ImportError:
            logger.error("Django import failed")

def setup_debug():
    """Setup debug environment and signal handlers"""
    try:
        # Register signal handler for debugging
        signal.signal(signal.SIGUSR1, debug_handler)
        
        # Set environment variables for debugging
        if '--debug' in sys.argv:
            sys.argv.remove('--debug')
            os.environ['DJANGO_DEBUG'] = 'True'
            os.environ['PYTHONBREAKPOINT'] = 'pdb.set_trace'
            logger.info("Debug mode enabled")
            
        # Setup migration-specific debugging
        setup_migration_debugging()
            
    except Exception as e:
        logger.error(f"Failed to setup debug environment: {e}")

def handle_migration_error(exc):
    """Special handling for migration errors"""
    logger.error("Migration error occurred:", exc_info=True)
    
    # Log database configuration
    try:
        from django.conf import settings
        logger.info("Database configuration:")
        for key, value in settings.DATABASES['default'].items():
            if key != 'PASSWORD':  # Don't log sensitive data
                logger.info(f"  {key}: {value}")
    except:
        logger.error("Could not log database configuration")
    
    # Check for common migration issues
    error_text = str(exc)
    if "no such table" in error_text:
        logger.error("Database table missing - try running makemigrations first")
    elif "already exists" in error_text:
        logger.error("Table conflict - check for conflicting migrations")
    elif "django_migrations" in error_text:
        logger.error("Migration table issue - check database connectivity")
    
    if os.environ.get('DJANGO_DEBUG') == 'True':
        print("\nEntering post-mortem debugging...")
        pdb.post_mortem()

def main():
    """Run administrative tasks with enhanced debugging."""
    try:
        # Setup debugging
        setup_debug()
        
        # Log startup information
        logger.info(f"Starting Django with args: {sys.argv}")
        logger.info(f"Python version: {sys.version}")
        
        # Set Django settings module
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'homescan.settings')
        
        try:
            from django.core.management import execute_from_command_line
            from django.conf import settings
        except ImportError as exc:
            logger.error("Failed to import Django", exc_info=True)
            raise ImportError(
                "Couldn't import Django. Are you sure it's installed and "
                "available on your PYTHONPATH environment variable? Did you "
                "forget to activate a virtual environment?"
            ) from exc
        
        # Execute command
        execute_from_command_line(sys.argv)
        
    except Exception as e:
        if 'migrate' in sys.argv:
            handle_migration_error(e)
        else:
            logger.error("Unhandled exception:", exc_info=True)
            
            if os.environ.get('DJANGO_DEBUG') == 'True':
                print("\nEntering post-mortem debugging...")
                pdb.post_mortem()
        raise

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.critical("Fatal error:", exc_info=True)
        sys.exit(1)

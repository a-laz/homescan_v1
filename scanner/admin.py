# scanner/admin.py
from django.contrib import admin
from .models import Room, FurnitureDetection

@admin.register(Room)
class RoomAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'created_at', 'total_volume')
    list_filter = ('user', 'created_at')
    search_fields = ('name', 'description', 'user__username')
    readonly_fields = ('created_at',)

@admin.register(FurnitureDetection)
class FurnitureDetectionAdmin(admin.ModelAdmin):
    list_display = ('furniture_type', 'room', 'confidence', 'volume', 'timestamp')
    list_filter = ('furniture_type', 'room', 'timestamp')
    search_fields = ('furniture_type', 'room__name')
    readonly_fields = ('timestamp',)
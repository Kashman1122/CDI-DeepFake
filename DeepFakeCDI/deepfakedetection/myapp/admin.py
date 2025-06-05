from django.contrib import admin
from .models import UserProfile, VideoAnalysis, VectorDB

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'phone', 'is_paid', 'created_at')
    list_filter = ('is_paid',)
    search_fields = ('user__username', 'user__email', 'phone')

@admin.register(VideoAnalysis)
class VideoAnalysisAdmin(admin.ModelAdmin):
    list_display = ('user', 'prediction', 'confidence', 'fake_percentage', 'real_percentage', 'created_at')
    list_filter = ('prediction', 'confidence', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)

@admin.register(VectorDB)
class VectorDBAdmin(admin.ModelAdmin):
    list_display = ('user', 'video_analysis', 'created_at')
    search_fields = ('user__username',)
    readonly_fields = ('created_at',)

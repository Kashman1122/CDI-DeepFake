from django.db import models
from django.contrib.auth.models import User
import json
from django.utils import timezone


class UserProfile(models.Model):
    """
    Extended user profile model with additional user information
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone = models.CharField(max_length=20, blank=True, null=True)
    is_paid = models.BooleanField(default=False)
    created_at = models.DateTimeField(default=timezone.now)  # Default value instead of auto_now_add
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username}'s Profile"


class VideoAnalysis(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_path = models.CharField(max_length=500)
    prediction = models.CharField(max_length=10)
    confidence = models.CharField(max_length=20)
    fake_percentage = models.FloatField()
    real_percentage = models.FloatField()
    avg_fake_frames = models.FloatField()
    avg_real_frames = models.FloatField()
    frame_count = models.IntegerField()
    metadata_json = models.TextField(blank=True, null=True)
    fake_scores_json = models.TextField(blank=True, null=True)  # Add this field
    created_at = models.DateTimeField(auto_now_add=True)

    def get_fake_scores(self):
        """Helper method to get fake_scores as a list"""
        if self.fake_scores_json:
            try:
                return json.loads(self.fake_scores_json)
            except json.JSONDecodeError:
                return []
        return []

    def get_chart_data(self):
        """Helper method to get formatted chart data"""
        fake_scores = self.get_fake_scores()
        if fake_scores:
            return {
                'frame_numbers': list(range(1, len(fake_scores) + 1)),
                'fake_scores': fake_scores,
                'threshold': 0.5,
                'avg_score': sum(fake_scores) / len(fake_scores)
            }
        return {}


class VectorDB(models.Model):
    """
    Stores frame embeddings and vector data for video analysis
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_analysis = models.OneToOneField(VideoAnalysis, on_delete=models.CASCADE, related_name='vector_data')
    frame_embeddings = models.JSONField(default=dict)  # Requires Django 3.1+
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"Vector data for {self.video_analysis}"

    class Meta:
        verbose_name = "Vector Database"
        verbose_name_plural = "Vector Databases"

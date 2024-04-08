from django.db import models

class ProcessedVideo(models.Model):
    video_file = models.FileField(upload_to='videos/')
    video_title = models.CharField(max_length=100)
    duration = models.FloatField()  # Duration in seconds
    emotional_tone = models.CharField(max_length=100)
    engagement_score = models.FloatField()
    time_stamp = models.DateTimeField()
from django.db import models
from django.contrib.auth.models import AbstractUser
from datetime import datetime

# CUSTOM USER MODEL
class User(AbstractUser):
    """Custom User model extending Django's AbstractUser"""
    full_name = models.CharField(max_length=255, blank=True, null=True) 
    gender = models.CharField(max_length=20, default="Not set")
    date_of_birth = models.DateField(default=datetime(2000, 1, 1).date())
    account_type = models.CharField(max_length=20, default="user")
    #profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.username


# VITALSTATS MODEL
class VitalStats(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='vital_stats')
    date = models.DateField(auto_now_add=True)
    time = models.CharField(max_length=10)
    bp_systolic = models.IntegerField()
    bp_diastolic = models.IntegerField()
    sugar_level = models.FloatField()
    notes = models.TextField(blank=True)
    
    class Meta:
        db_table = 'vital_stats'
        unique_together = ['user', 'date']  # Ensure only one record per user per day
        
    def __str__(self):
        return f"{self.user.username} - {self.date}"


# NOTIFICATION MODEL
class Notification(models.Model):
    TYPE_CHOICES = [
        ('reminder', 'Reminder'),
        ('insight', 'Insight'),
        ('alert', 'Alert'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='notifications')
    message = models.TextField()
    type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='reminder')
    sent_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)
    
    class Meta:
        db_table = 'notifications'
        ordering = ['-sent_at']
        
    def __str__(self):
        return f"{self.user.username} - {self.message[:50]}"


# HEALTH RECORD MODEL
class HealthRecord(models.Model):
    RECORD_TYPE_CHOICES = [
        ('vitals', 'Vitals'),
        ('diagnosis', 'Diagnosis'),
        ('report', 'Report'),
        ('symptoms', 'Symptoms'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='health_records')
    record_type = models.CharField(max_length=20, choices=RECORD_TYPE_CHOICES)
    description = models.TextField(blank=True)
    file_path = models.FileField(upload_to='health_records/', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'health_records'
        ordering = ['-created_at']
        
    def __str__(self):
        return f"{self.user.username} - {self.record_type} - {self.created_at.date()}"
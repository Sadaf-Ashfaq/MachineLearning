from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import VitalStats, Notification
from datetime import datetime
from django.utils import timezone

# Implementation of observer pattern
@receiver(post_save, sender=VitalStats)
def notify_on_vitals(sender, instance, created, **kwargs):
    if created:  # Only create notification for new records, not updates
        Notification.objects.create(
            user=instance.user,  # Using ForeignKey relationship instead of user_id
            message="Your vitals were recorded successfully.",
            type="info",
            sent_at=timezone.now()  # Using Django's timezone-aware datetime
        )
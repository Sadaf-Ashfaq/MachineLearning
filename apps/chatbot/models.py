from django.db import models

class ChatMessage(models.Model):
    user_id = models.CharField(max_length=255)
    chat_type = models.CharField(max_length=10, choices=[("mental", "mental"), ("health", "health")])
    role = models.CharField(max_length=10, choices=[("user", "user"), ("bot", "bot")])
    text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

from django.urls import path
from . import views

urlpatterns = [
    path("chat/<str:chat_type>/", views.chat_view, name="chat_view"),
    path('chat/<str:chat_type>/clear/', views.clear_chat_view, name='clear_chat'),
]


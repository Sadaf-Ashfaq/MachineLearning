from django.urls import path
from . import views

urlpatterns=[
    path('register/',views.register_view,name='register'),
    path('login/', views.login_view, name='login'), 
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('edit-profile/', views.edit_profile_view, name='edit_profile'),
    path('notifications/', views.notification_list_view, name='notification_list_view'),
    path('notifications/clear/', views.clear_notifications_view, name='clear_notifications'),
    path('health-records/', views.health_records_view, name='health_records'),
]
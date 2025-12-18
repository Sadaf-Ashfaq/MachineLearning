from django.urls import path
from . import views

app_name = 'disease_prediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('stroke-prediction/', views.stroke_prediction, name='stroke_prediction'),
    path('diabetes-risk/', views.diabetes_risk, name='diabetes_risk'),
    path('heart-disease/', views.heart_disease, name='heart_disease'),
    path('liver-disease/', views.liver_disease, name='liver_disease'),
    path('obesity-prediction/', views.obesity_prediction, name='obesity-prediction'),
    path('lifestyle-score/', views.lifestyle_score, name='lifestyle_score'),
 
]
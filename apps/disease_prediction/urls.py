from django.urls import path
from . import views

app_name = 'disease_prediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('disease-prediction/', views.disease_prediction, name='disease_prediction'),
    path('diabetes-risk/', views.diabetes_risk, name='diabetes_risk'),
   path('heart-disease/', views.heart_prediction_view, name='heart_disease'),

    path('liver-disease/', views.liver_disease, name='liver_disease'),
    path('nutrient-deficiency/', views.nutrient_deficiency, name='nutrient_deficiency'),
    path('lifestyle-score/', views.lifestyle_score, name='lifestyle_score'),
    path('symptom-severity/', views.symptom_severity, name='symptom_severity'),
]
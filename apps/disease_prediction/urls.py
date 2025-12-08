from django.urls import path
from . import views

app_name = 'disease_prediction'

urlpatterns = [
    path('', views.index, name='index'),
    path('liver-disease/', views.liver_disease, name='liver_disease'),
    path('nutrient-deficiency/', views.nutrient_deficiency, name='nutrient_deficiency'),
]



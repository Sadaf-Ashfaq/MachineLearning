from django.shortcuts import render
from .ml_utils import predict_liver_disease

def index(request):
    return render(request, 'disease_prediction/disease_prediction.html')

def liver_disease(request):
    result = None
    
    if request.method == 'POST':
        data = {
            'age': float(request.POST.get('age')),
            'gender': request.POST.get('gender'),
            'total_bilirubin': float(request.POST.get('total_bilirubin')),
            'direct_bilirubin': float(request.POST.get('direct_bilirubin')),
            'alkaline_phosphotase': float(request.POST.get('alkaline_phosphotase')),
            'alamine_aminotransferase': float(request.POST.get('alamine_aminotransferase')),
            'aspartate_aminotransferase': float(request.POST.get('aspartate_aminotransferase')),
            'total_proteins': float(request.POST.get('total_proteins')),
            'albumin': float(request.POST.get('albumin')),
            'ag_ratio': float(request.POST.get('ag_ratio'))
        }
        
        result = predict_liver_disease(data)
    
    return render(request, 'disease_prediction/liver_disease.html', {'result': result})

def nutrient_deficiency(request):
    return render(request, 'disease_prediction/nutrient_deficiency.html')
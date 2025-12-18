from django.shortcuts import render
from .ml_utils import predict_liver_disease
from .ml_utils import predict_heart_disease




def index(request):
    return render(request, 'disease_prediction/disease_prediction.html')

# 1. Disease Prediction (General)
def disease_prediction(request):
    return render(request, 'disease_prediction/disease_prediction_form.html')

# 2. Diabetes Risk
def diabetes_risk(request):
    return render(request, 'disease_prediction/diabetes_risk.html')

# 3. Heart Disease
def heart_prediction_view(request):
    result = None

    if request.method == "POST":
        try:
            input_data = {
                "Age": int(request.POST.get("Age")),
                "Sex": request.POST.get("Sex"),
                "ChestPainType": request.POST.get("ChestPainType"),
                "RestingBP": int(request.POST.get("RestingBP")),
                "Cholesterol": int(request.POST.get("Cholesterol")),
                "FastingBS": int(request.POST.get("FastingBS")),
                "RestingECG": request.POST.get("RestingECG"),
                "MaxHR": int(request.POST.get("MaxHR")),
                "ExerciseAngina": request.POST.get("ExerciseAngina"),
                "Oldpeak": float(request.POST.get("Oldpeak")),
                "ST_Slope": request.POST.get("ST_Slope"),
            }

            result = predict_heart_disease(input_data)

        except Exception as e:
            result = {
                "status": "error",
                "error": str(e)
            }

    return render(
        request,
        "disease_prediction/heart_disease.html",
        {"result": result}
    )

#
# 5. Liver Disease (Already working with ML model)
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

# 6. Nutrient Deficiency
def nutrient_deficiency(request):
    return render(request, 'disease_prediction/nutrient_deficiency.html')

# 7. Lifestyle Score
def lifestyle_score(request):
    return render(request, 'disease_prediction/lifestyle_score.html')

# 8. Symptom Severity
def symptom_severity(request):
    return render(request, 'disease_prediction/symptom_severity.html')



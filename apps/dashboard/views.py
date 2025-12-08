from django.shortcuts import render
from django.contrib.auth.decorators import login_required

@login_required
def home_view(request):
    return render(request, 'dashboard/home.html')

def about_view(request):
    return render(request, 'dashboard/about.html')


from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from .models import User, VitalStats, Notification, HealthRecord
from django.contrib import messages
from datetime import datetime, date
from .factories import ChartDataFactory
from .strategies import LineChart, BarChart
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import os
import uuid
from django.conf import settings
from django.db import IntegrityError


@login_required
def health_records_view(request):
    user = request.user

    # Handle POST upload
    if request.method == 'POST':
        record_type = request.POST.get('record_type')
        description = request.POST.get('description')
        file = request.FILES.get('file_path') 

        if record_type:
            HealthRecord.objects.create(
                user=user,
                record_type=record_type,
                description=description,
                file_path=file
            )

    # Fetch all records
    records = HealthRecord.objects.filter(user=user).order_by('-created_at')

    return render(request, 'user_accounts/health_records.html', {
        'records': records
    })


def register_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Check if username already exists
        if User.objects.filter(username=username).exists():  
            messages.error(request, "Username already exists.")
            return redirect('register')

        if User.objects.filter(email=email).exists():  
            messages.error(request, "Email is already in use.")
            return redirect('register')

        user = User.objects.create_user(username=username, email=email, password=password)
        user.save()

        messages.success(request, "Account created successfully.")
        return redirect('login')

    return render(request, 'user_accounts/register.html')


def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  
        else:
            messages.error(request, "Invalid credentials.")
            return redirect('login')
    return render(request, 'user_accounts/login.html')


def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def profile_view(request):
    user = request.user
    today = date.today()

    error_message = None
    success_message = None 

    if request.method == 'POST' and 'bp_sys' in request.POST:
        bp_sys = request.POST.get('bp_sys')
        bp_dia = request.POST.get('bp_dia')
        sugar = request.POST.get('sugar')

        try:
            bp_sys = int(bp_sys)
            bp_dia = int(bp_dia)
            sugar = float(sugar)

            # Range validation
            if not (50 <= bp_sys <= 250):
                error_message = "🩸 Systolic BP must be between 50 and 250"
            elif not (30 <= bp_dia <= 150):
                error_message = "🩸 Diastolic BP must be between 30 and 150"
            elif not (30 <= sugar <= 500):
                error_message = "🍬 Sugar level must be between 30 and 500"
            elif bp_sys <= bp_dia:  # Add logical validation
                error_message = "🩸 Systolic BP must be higher than Diastolic BP"
            else:
                # Check if data already exists for today
                existing_record = VitalStats.objects.filter(user=user, date=today).first()
                if existing_record:
                    error_message = "⚠️ You have already recorded your vitals for today. You can only add vitals once per day."
                else:
                    # Save the data
                    VitalStats.objects.create(
                        user=user,
                        time=datetime.now().strftime("%H:%M"),
                        bp_systolic=bp_sys,
                        bp_diastolic=bp_dia,
                        sugar_level=sugar,
                        notes=""
                    )
                    success_message = "✅ Your vitals have been recorded successfully!"

        except ValueError:
            error_message = "❌ Please enter valid numeric values!"

    # Chart data logic remains the same
    chart_data = None
    chart_type = request.GET.get("chart")
    if chart_type:
        factory = ChartDataFactory(str(user.id))
        labels, data = factory.get_data(chart_type)

        if chart_type == 'bp':
            strategy = LineChart()
            color = 'red'
        elif chart_type == 'sugar':
            strategy = BarChart()
            color = 'green'
        else:
            strategy = LineChart()
            color = 'gray'

        chart_data = strategy.render(labels or [], data or [], chart_type.upper(), color)

    # Handle notifications (optional - remove the notification logic or keep it simple)
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    # Clear today's reminder notifications if they exist
    Notification.objects.filter(
        user=user,
        message="Don't forget to update your vitals!",
        sent_at__gte=start_of_day,
        sent_at__lte=end_of_day
    ).delete()
    
    unread_count = Notification.objects.filter(user=user, is_read=False).count()

    return render(request, 'user_accounts/profile.html', {
        'profile': user,  # Now using the User model directly
        'chart_data': chart_data,
        'unread_count': unread_count,
        'error_message': error_message,
        'success_message': success_message
    })


@login_required
def edit_profile_view(request):
    user = request.user

    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        gender = request.POST.get('gender')
        dob_str = request.POST.get('dob')

       
        if full_name:
            user.full_name = full_name
            
        if gender:
            user.gender = gender
            
        if dob_str:
            try:
                user.date_of_birth = datetime.strptime(dob_str, "%Y-%m-%d").date()
            except ValueError:
                pass       

        user.save()
        return redirect('profile')
        
    return render(request, 'user_accounts/profile.html', {
        'profile': user
    })


@login_required
def notification_list_view(request):
    user = request.user
    notifications = Notification.objects.filter(user=user).order_by('-sent_at')
    
    # Mark unread notifications as read
    unread_notifications = notifications.filter(is_read=False)
    unread_notifications.update(is_read=True)

    return render(request, 'user_accounts/notifications.html', {
        'notifications': notifications
    })


@login_required
def clear_notifications_view(request):
    if request.method == "POST":
        Notification.objects.filter(user=request.user).delete()
    return redirect('notification_list_view')



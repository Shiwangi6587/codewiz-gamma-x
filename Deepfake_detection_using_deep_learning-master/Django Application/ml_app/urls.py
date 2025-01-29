"""project_settings URL Configuration
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from .views import upload_video, predict_page, cuda_full

app_name = 'ml_app'
handler404 = views.handler404

urlpatterns = [
    path('', views.upload_video, name='home'),
    path('fake_news/', views.fake_news, name='fake_news'),
    path('deepfake/', views.deepfake, name='deepfake'),
    path('dashboard/', views.dashboard, name='dashboard'),  # Add dashboard route
    path('upload_video/', views.upload_video, name='upload_video'),
    path('predict/', views.predict_page, name='predict'),
    path('cuda_full/', views.cuda_full, name='cuda_full'),
    path('verify_news/', views.verify_news, name='verify_news'),
    path('get_csrf_token/', views.get_csrf_token, name='get_csrf_token'),
]
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict_view, name='predict'),
    path('get-thresholds/', views.get_thresholds, name='get_thresholds'),
]
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home),
    path('run_test/',views.run_test),
    path('ppt/', views.ppt)
]
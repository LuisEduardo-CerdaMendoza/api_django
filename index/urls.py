from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('code14/', include('code14.urls')),
    path('code15/', include('code15.urls')),
    path('code15/', include('code16.urls')),
]

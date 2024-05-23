from django.urls import path
from .views import index

app_name = 'cod18'
urlpatterns = [
    path('', index, name='home'),
]

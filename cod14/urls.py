from django.urls import path
from .views import index

app_name = 'cod14'
urlpatterns = [
    path('', index, name='home'),
]

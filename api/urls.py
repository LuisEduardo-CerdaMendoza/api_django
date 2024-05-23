from django.urls import path
from index import views as index_views
from cod14 import views as cod14_views
from cod15 import views as cod15_views
from cod16 import views as cod16_views
from cod17 import views as cod17_views
from cod18 import views as cod18_views

urlpatterns = [
    path('', index_views.index, name='index'),
    path('index/', index_views.index, name='index_index'),
    path('cod14/', cod14_views.index, name='cod14_index'),
    path('cod15/', cod15_views.index, name='cod15_index'),
    path('cod16/', cod16_views.index, name='cod16_index'),
    path('cod17/', cod17_views.index, name='cod17_index'),
    path('cod18/', cod18_views.index, name='cod18_index'),
    # Otras URLs de tu proyecto...
]
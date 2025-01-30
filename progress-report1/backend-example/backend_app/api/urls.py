from django.urls import path
from .views import (
    serve_map, make_prediction
)

urlpatterns = [
    path('map/', serve_map, name='serve_map'),
    path('predict/', make_prediction, name='make_prediction'),
]

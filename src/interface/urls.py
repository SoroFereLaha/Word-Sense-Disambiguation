from django.urls import path
from . import views



urlpatterns = [

    path('',views.home, name="home"),
    path('lesk/',views.lesk_method, name="lesk"),
    path('knn/',views.knn, name="knn"),
    path('bilstm/',views.bilstm, name="bilstm"),  
    path('index/', views.index, name='index'),
]

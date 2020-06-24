from django.urls import path
from . import views

urlpatterns =[

    path('', views.dashboard, name='dashboard'),

    path('analysisProduct/', views.analysisProduct, name='analysisProduct'),
    path('analysisProduct/add', views.add, name='da'),

    path('analysisOutlet/', views.analysisOutlet, name='analysisOutlet'),
    path('analysisOutlet/sendOutlet', views.analysisOutlet2, name='analysisOutlet'),
    path('analysisOutlet/showChart', views.showOutletChart, name='analysisOutlet'),

    path('analysisSupplier/', views.analysisSupplier, name='analysisSupplier'),
    path('analysisSupplier/sendSupplier', views.analysisSupplier2, name='analysisSupplier'),
    path('analysisSupplier/showChart', views.showSupplierChart, name='analysisSupplier'),

    path('salesPrediction/', views.salesPrediction, name='salesPrediction'),
    path('salesPrediction/predict', views.predict, name='predict'),

    path('aboutUs/', views.aboutUs, name='aboutUs'),

    path('debug/', views.debug, name='debug'),

    path('empty/', views.empty, name='empty'),
]
from django.urls import path
from . import views

urlpatterns = [

    # Home Page
    path('', views.home, name='home'),

    # Resume Upload
    path('upload/', views.upload_pdf, name='upload_pdf'),

    # Resume Score Dashboard
    path('score/', views.score_resume, name='score_resume'),

    # AI Chat API
    path('chat/', views.chat_api, name='chat_api'),

]

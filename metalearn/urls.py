from django.urls import path
from django.contrib import admin
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),

    path('getEpisodeNoisyExecution/', views.getEpisodeNoisyExecution, name='getEpisodeNoisyExecution'),
    path('getEpisodeNoisyExecution/<str:preferedEpisodeIds>', views.getEpisodeNoisyExecution, name='getEpisodeNoisyExecution'),

    path('getEpisodeWeightNoise/<str:episodeId>', views.getEpisodeWeightNoise, name='getEpisodeWeightNoise'),
    path('putResult/<int:episodeNoisyExecutionId>/<str:lock>', views.putResult, name='putResult'),

    path('stats/<str:settings>', views.stats, name='stats'),
    path('admin/', admin.site.urls),
]

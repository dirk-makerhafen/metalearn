from django.urls import path, include
from django.contrib import admin
from django.conf import settings
from cruds_adminlte.urls import crud_for_app

from . import views


urlpatterns = [
    path('', views.dashboard, name='dashboard'),

    path('getEpisodeNoisyExecution/', views.getEpisodeNoisyExecution, name='getEpisodeNoisyExecution'),
    path('getEpisodeNoisyExecution/<str:preferedEpisodeIds>', views.getEpisodeNoisyExecution, name='getEpisodeNoisyExecution'),

    path('getEpisodeWeightNoise/<str:episodeId>', views.getEpisodeWeightNoise, name='getEpisodeWeightNoise'),
    path('putResult/<int:episodeNoisyExecutionId>/<str:lock>', views.putResult, name='putResult'),

    path('stats/<str:settings>', views.stats, name='stats'),
    path('admin/', admin.site.urls),

    path('', include(views.EnvironmentView().get_urls())),
    path('', include(views.ArchitectureView().get_urls())),
    path('', include(views.OptimiserView().get_urls())),
    path('', include(views.ExperimentSetView().get_urls())),
    path('', include(views.ExperimentView().get_urls())),
    path('', include(views.EpisodeView().get_urls())),
    path('', include(views.EpisodeNoisyExecutionView().get_urls())),

]
if settings.DEBUG:
    import debug_toolbar
    urlpatterns += [
        path('__debub__/', include(debug_toolbar.urls)),
    ]


#urlpatterns += crud_for_app('metalearn')

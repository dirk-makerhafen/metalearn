from django.contrib import admin

# Register your models here.

from .models import Environment
from .models import Architecture
from .models import Optimiser

from .models import ExperimentSet
from .models import ExperimentSetToEnvironment
from .models import ExperimentSetToArchitecture
from .models import ExperimentSetToOptimiser

#from .models import OptimiserInstance
from .models import Experiment
from .models import Episode
from .models import EpisodeNoisyExecution

def action_ExperimentNoisyExecution_setIdleUnlocked(modeladmin, request, queryset):
    queryset.update(status = 'idle', lock = "", client = "")

action_ExperimentNoisyExecution_setIdleUnlocked.short_description = "Set selected Items to Idle/Unlocked"


class ExperimentSetToEnvironment_Inline(admin.TabularInline):
    model = ExperimentSetToEnvironment
    extra = 1 # how many rows to show
class ExperimentSetToArchitecture_Inline(admin.TabularInline):
    model = ExperimentSetToArchitecture
    extra = 1 # how many rows to show
class ExperimentSetToOptimiser_Inline(admin.TabularInline):
    model = ExperimentSetToOptimiser
    extra = 1 # how many rows to show

class ExperimentSet_Admin(admin.ModelAdmin):
    inlines = (ExperimentSetToEnvironment_Inline, ExperimentSetToArchitecture_Inline, ExperimentSetToOptimiser_Inline)
    list_display = ('id', "created", 'name', "status" )
    list_filter = ('status',)

class Experiment_Admin(admin.ModelAdmin):
    list_display = ('id', "created", 'status', "timespend", "fitness_min", "fitness_max", "fitness_avg", "fitness_median", "fitness_top10", "environment", "architecture", "optimiser", "experimentSet" )
    list_filter = ('status',)

class Episode_Admin(admin.ModelAdmin):
    list_display = ('id', "created", 'status', "version", "timespend", "fitness_min", "fitness_max", "fitness_avg", "fitness_median", "fitness_top10", "environment", "architecture", "optimiser", "experimentSet", "experiment", )
    list_filter = ('status',)

class ExperimentNoisyExecution_Admin(admin.ModelAdmin):
    list_display = ('id', "created", 'status', "timespend", "noiseseed",  "fitness", "fitness_rank", "fitness_calc_key","fitness_calc_value", "client", "environment", "architecture", "optimiser", "experimentSet", "experiment", "episode" )
    list_filter = ('status',)
    actions = [action_ExperimentNoisyExecution_setIdleUnlocked]




admin.site.register(Environment)
admin.site.register(Architecture)
admin.site.register(Optimiser)

admin.site.register(ExperimentSet, ExperimentSet_Admin)
#admin.site.register(ExperimentSetToEnvironment)
#admin.site.register(ExperimentSetToArchitecture)
#dmin.site.register(ExperimentSetToOptimiser)

admin.site.register(Experiment, Experiment_Admin)
admin.site.register(Episode   , Episode_Admin)
admin.site.register(EpisodeNoisyExecution , ExperimentNoisyExecution_Admin)







from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django import forms
from collections import OrderedDict
from django.utils import timezone
from django.urls import reverse
from django.http import JsonResponse
from django.db.models import Avg, Count, Min, Sum

import random
import redis
import json
import time
import numpy
from django.db import connection
import datetime
import uuid
from django.conf.urls import url, include
from cruds_adminlte import utils


# Create your views here.

from django.http import HttpResponse

from . import models
from . import tasks

from cruds_adminlte.crud import CRUDView
from cruds_adminlte.inline_crud import InlineAjaxCRUD


# Monkey patch because django-crud-adminlte because git 0.0.13 != pip 0.0.13
import cruds_adminlte.utils
def get_fields(model, include=None):
    """
    Returns ordered dict in format 'field': 'verbose_name'
    """
    fields = OrderedDict()
    info = model._meta
    if include:  # self.model._meta.get_field(fsm_field_name)
        selected = {}
        for name in include:
            if '__' in name:
                related_model, field_name = name.split('__', 1)
                try:
                    selected[name] = \
                        info.get_field_by_name(related_model)[0].\
                        related_model._meta.get_field_by_name(name)[0]
                except:
                    selected[name] = info.get_field(related_model).\
                            related_model._meta.get_field(field_name)
            else:
                try:
                    selected[name] = info.get_field_by_name(name)[0]
                except:
                    selected[name] = info.get_field(name)
    else:
        try:
            selected = {field.name: field for field in info.fields
                        if field.editable}
        except:
            # Python < 2.7
            selected = dict((field.name, field) for field in info.fields
                            if field.editable)
    for name, field in selected.items():
        if field.__class__.__name__ == 'ManyToOneRel':
            field.verbose_name = field.related_name
        fields[name] = [
            field.verbose_name.title(),
            field.get_internal_type]
    if include:
        fields = OrderedDict((key, fields[key]) for key in include)
    return fields
cruds_adminlte.utils.get_fields = get_fields



class EnvironmentView(CRUDView):
    model = models.Environment
    template_name_base = "Environment"
    search_fields = ['classname__contains', 'classargs__contains', 'name__contains', 'description__contains']
    split_space_search = ' '
    list_filter = ['id','classname', 'name']
    list_fields = ['id', 'name', 'classname','classargs','description', 'broken']


class ArchitectureView(CRUDView):
    model = models.Architecture
    template_name_base = "Architecture"
    search_fields = ['classname__contains', 'classargs__contains', 'name__contains', 'description__contains']
    split_space_search = ' '
    list_filter = ['id','classname', 'name']
    list_fields = ['id', 'name', 'classname','classargs','description', 'broken']


class OptimiserView(CRUDView):
    model = models.Optimiser
    template_name_base = "Optimiser"
    search_fields = ['classname__contains', 'classargs__contains', 'name__contains', 'description__contains']
    split_space_search = ' '
    list_filter = ['id','classname', 'name']
    list_fields = ['id', 'name', 'classname','classargs','description', 'broken']


class ExperimentSetView(CRUDView):
    model = models.ExperimentSet
    template_name_base = "ExperimentSet"

    list_filter = [ 
        'id', 
        'status',
    ]
    list_fields = [ 
        'id', 
        'name', 'description', 

        'subsettings_Experiments_max',
        'subsettings_Episodes_max',
        #'subsettings_EpisodeNoisyExecutions_min',
        #'subsettings_EpisodeNoisyExecutions_max',
        #'subsettings_EpisodeNoisyExecutions_min_steps',
        #'subsettings_EpisodeNoisyExecutions_max_steps',
        #'subsettings_EpisodeNoisyExecutions_min_timespend',
        #'subsettings_EpisodeNoisyExecutions_max_timespend',

        'public', 'status', 'timespend', 
        'on_created_executed', 'on_done_executed'
    ]

    
    def get_urls(self):
        pre = ""
        try:
            if self.cruds_url:
                pre = "%s/" % self.cruds_url
        except AttributeError:
            pre = ""
        base_name = "%s%s/%s" % (pre, self.model._meta.app_label, self.model.__name__.lower())

        urls = CRUDView.get_urls(self)
        urls.append(
            url(r"^%s/(?P<pk>[^/]+)/trigger_on_done$" % (base_name,),
            self.trigger_on_done,
            name=utils.crud_url_name(self.model, 'trigger_on_done', prefix=self.urlprefix))
        )   
        return urls

    def trigger_on_done(self, request, pk):
        tasks.on_ExperimentSet_done.delay(pk)
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


class ExperimentView(CRUDView):
    model = models.Experiment
    template_name_base = "Experiment"
    paginate_by = 200

    list_filter = [ 
        'id', 
        'environment', 'architecture', 'optimiser', 'experimentSet', 
        'status',
    ]
    list_fields = [ 
        'environment', 'architecture', 'optimiser', 'experimentSet',  
        'public', 'status', 'timespend', 
        'fitness_min', 'fitness_max', 'fitness_avg', 'fitness_median', 'fitness_top10' ,
        'on_created_executed', 'on_done_executed'
    ]

    def get_urls(self):
        pre = ""
        try:
            if self.cruds_url:
                pre = "%s/" % self.cruds_url
        except AttributeError:
            pre = ""
        base_name = "%s%s/%s" % (pre, self.model._meta.app_label, self.model.__name__.lower())

        urls = CRUDView.get_urls(self)
        urls.append(
            url(r"^%s/(?P<pk>[^/]+)/trigger_on_done$" % (base_name,),
            self.trigger_on_done,
            name=utils.crud_url_name(self.model, 'trigger_on_done', prefix=self.urlprefix))
        )   
        return urls

    def trigger_on_done(self, request, pk):
        ex = models.Experiment.objects.get(id=pk)
        tasks.on_Experiment_done.delay(ex.id, ex.experimentSet_id)
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


class EpisodeView(CRUDView):
    model = models.Episode
    template_name_base = "Episode"
    paginate_by = 200

    list_filter = [ 
        'id', 
        'environment', 'architecture', 'optimiser', 'experimentSet', 'experiment', 
        'status',
    ]
    list_fields = [ 
        'environment', 'architecture', 'optimiser', 'experimentSet', 'experiment', 'public',
        'version', 'status',   'timespend', 
        'subsettings_EpisodeNoisyExecutions_max', 'subsettings_EpisodeNoisyExecutions_max_steps', 'subsettings_EpisodeNoisyExecutions_max_steps_unrewarded', 'subsettings_EpisodeNoisyExecutions_max_timespend',
        'fitness_min', 'fitness_max', 'fitness_avg', 'fitness_median', 'fitness_top10' ,
        'on_created_executed', 'on_done_executed'
    ]

    def get_urls(self):
        pre = ""
        try:
            if self.cruds_url:
                pre = "%s/" % self.cruds_url
        except AttributeError:
            pre = ""
        base_name = "%s%s/%s" % (pre, self.model._meta.app_label, self.model.__name__.lower())

        urls = CRUDView.get_urls(self)
        urls.append(
            url(r"^%s/(?P<pk>[^/]+)/trigger_on_done$" % (base_name,),
            self.trigger_on_done,
            name=utils.crud_url_name(self.model, 'trigger_on_done', prefix=self.urlprefix))
        )   
        return urls

    def trigger_on_done(self, request, pk):
        ep = models.Episode.objects.get(id=pk)
        tasks.on_Episode_done.delay(ep.id, ep.experiment_id, ep.experimentSet_id)
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))


class EpisodeNoisyExecutionView(CRUDView):
    model = models.EpisodeNoisyExecution
    template_name_base = "EpisodeNoisyExecution"
    paginate_by = 200

    list_filter = [ 
        'id', 
        'environment', 'architecture', 'optimiser', 'experimentSet', 'experiment', 'episode', 
        'status', 'client' 
    ]
    list_fields = [ 
        'environment', 'architecture', 'optimiser', 'experimentSet', 'experiment', 'episode', 
        'number', 'status', 'steps', 'first_rewarded_step', 'timespend',  
        'fitness', 'fitness_scaled', 'fitness_rank', 'fitness_norm', 'fitness_norm_scaled', 
        'client', 
        'on_created_executed', 'on_done_executed'
    ]


    def get_urls(self):
        pre = ""
        try:
            if self.cruds_url:
                pre = "%s/" % self.cruds_url
        except AttributeError:
            pre = ""
        base_name = "%s%s/%s" % (pre, self.model._meta.app_label, self.model.__name__.lower())

        urls = CRUDView.get_urls(self)
        urls.append(
            url(r"^%s/(?P<pk>[^/]+)/clearlock$" % (base_name,),
            self.clearlock,
            name=utils.crud_url_name(self.model, 'clearlock', prefix=self.urlprefix))
        )   
        return urls

    def clearlock(self, request, pk):
        models.EpisodeNoisyExecution.objects.filter(id=pk).update(status = 'idle', lock = "", client = "")
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))



def dashboard(request):
    template = loader.get_template('dashboard.html')
    return HttpResponse(template.render({}, request))

def workerstats(request):
    time_threshold = timezone.now() - datetime.timedelta(minutes=10)

    clients = list(models.EpisodeNoisyExecution.objects.filter(status="locked", experiment__public=True).values_list("client",flat=True))
    done_last_10_min = models.EpisodeNoisyExecution.objects.filter(status="done", updated__gte = time_threshold, experiment__public=True).count()
    steps_last_10_min = models.EpisodeNoisyExecution.objects.filter(status="done", updated__gte = time_threshold, experiment__public=True).aggregate(Sum('steps'))
    idlecnt = models.EpisodeNoisyExecution.objects.filter(status="idle", experiment__public=True).count()
    permin = done_last_10_min / 10.0 

    stepsmin = 0
    stepspersecpercore = 0
    if steps_last_10_min["steps__sum"] != None:
        stepsmin = steps_last_10_min["steps__sum"] / 10.0 
        if len(clients) > 0:
            stepspersecpercore = stepsmin / 60.0 / len(clients)
        
    return JsonResponse({
        "threads": len(clients) ,
        "clients": len(set(clients)),
        "taskspermin": permin,
        "tasksidle": idlecnt,
        "stepspermin": stepsmin,
        "stepspersecpercore": stepspersecpercore,
    },safe=False)   


# client can request preferedEpisodeIds if he has those episodes in cache so no additional download of weights is needed
def getEpisodeNoisyExecution(request, preferedEpisodeIds = ''):
    if preferedEpisodeIds != "":
        preferedEpisodeIds = [int(x) for x in preferedEpisodeIds.split(",")]
    client_ip = request.META.get('REMOTE_ADDR')

    episodeNoisyExecution, lock = models.EpisodeNoisyExecution.getOneIdleLocked(client_ip, episodeIds = preferedEpisodeIds )
    if episodeNoisyExecution == None and len(preferedEpisodeIds) > 0:
        episodeNoisyExecution, lock = models.EpisodeNoisyExecution.getOneIdleLocked(client_ip)

    if episodeNoisyExecution != None:
        episodeNoisyExecutionJson = {
            "lock"              : lock,
            "id"                : episodeNoisyExecution.id,
            "noiseseed"         : episodeNoisyExecution.noiseseed,
            "episode.id"        : episodeNoisyExecution.episode.id,
            "max_timespend"     : episodeNoisyExecution.episode.subsettings_EpisodeNoisyExecutions_max_timespend,
            "max_steps"         : episodeNoisyExecution.episode.subsettings_EpisodeNoisyExecutions_max_steps,
            "max_steps_unrewarded"   : episodeNoisyExecution.episode.subsettings_EpisodeNoisyExecutions_max_steps_unrewarded,
            "environment.classname"  : episodeNoisyExecution.environment.classname,
            "environment.classargs"  : episodeNoisyExecution.environment.classargs,
            "architecture.classname" : episodeNoisyExecution.architecture.classname,
            "architecture.classargs" : episodeNoisyExecution.architecture.classargs,
        }
        return JsonResponse(episodeNoisyExecutionJson,safe=False)   
    return JsonResponse(None,safe=False)   

def getEpisodeWeightNoise(request, episodeId):
    episode = models.Episode.objects.get(id=episodeId)
    data = open(episode.weightsNoiseFile,"rb").read()
    return HttpResponse(data, content_type='application/octet-stream')
  
  
@csrf_exempt
def putResult(request,episodeNoisyExecutionId, lock):
    if request.method != 'POST':
        return
    episodeNoisyExecution = models.EpisodeNoisyExecution.objects.get(id=int(episodeNoisyExecutionId), lock = lock)
    data = json.loads(request.body.decode("UTF-8"))
    episodeNoisyExecution.setResult(data)
    return HttpResponse()


available_names = [
    "Environment.created",
    "Environment.name",
    "Environment.groupname",
    "Environment.classname",
    "Environment.classargs",
    "Architecture.created",
    "Architecture.name",
    "Architecture.groupname",
    "Architecture.classname",
    "Architecture.classargs",
    "Optimiser.created",
    "Optimiser.name",
    "Optimiser.groupname",
    "Optimiser.classname",
    "Optimiser.classargs",
    "ExperimentSet.id",
    "ExperimentSet.created",
    "ExperimentSet.name",
    "ExperimentSet.status",
    "ExperimentSet.max_Episodes",
    "ExperimentSet.episodeNoisyExecutions_count_min",
    "ExperimentSet.episodeNoisyExecutions_count_max",
    "Experiment.id",
    "Experiment.created",
    "Experiment.status",
    "Experiment.timespend",
    "Episode.id",
    "Episode.created",
    "Episode.status",
    "Episode.version",
    "Episode.max_NoisyExecutions",
    "Episode.timespend",
    "Episode.fitness_min",
    "Episode.fitness_max",
    "Episode.fitness_avg",
    "Episode.fitness_mean",
    "EpisodeNoisyExecution.id",
    "EpisodeNoisyExecution.number",
    "EpisodeNoisyExecution.created",
    "EpisodeNoisyExecution.status",
    "EpisodeNoisyExecution.timespend",
    "EpisodeNoisyExecution.fitness",
]
def stats(request, settings):
    r = {
            "axis" : {
                "x" : {
                    "fields": [ "ExperimentSet.name", "ExperimentSet.id" ],
                },
                "y" : {
                    "fields": [ "Experiment.id" ],
                },
                "z" : {
                    "fields": [ "EpisodeNoisyExecution.id" ],
                },
                "color" : {
                    "field": "Episode.id",
                },
            },
            "aggregation": {
                "axis" : "color",
                "type" : "avg"
            }
        }
    r = json.loads(settings)
    #print(r)
    with connection.cursor() as cursor:
        prefix = "metalearn_"
        selectelements = []
        fromelements = []
        groupelements = []
        joinelements = []
        tmpjoinelements = []

        if r["aggregation"]["axis"] not in ["x", "y", "z", "color"]:    
            print("unknown aggregation axis: %s" % r["aggregation"]["axis"])
            return HttpResponse("unknown aggregation axis: %s" % r["aggregation"]["axis"])
        if r["aggregation"]["type"] not in ["min", "max", "mean", "avg", "sum", "count"]:
            print("unknown aggregation type: %s" % r["aggregation"]["type"])
            return HttpResponse("unknown aggregation type: %s" % r["aggregation"]["type"])

        for axisName in ["x", "y", "z"]:
            if r["axis"][axisName] == None:
                continue
            for field in r["axis"][axisName]["fields"]:
                if field not in available_names:
                    return HttpResponse("unknown: %s" % field)

            if axisName == r["aggregation"]["axis"]:
                if len(r["axis"][axisName]["fields"]) > 1:
                    print("no aggregation for multi data axis: %s" % axisName)
                    return HttpResponse("no aggregation for multi data axis: %s" % axisName)
                selectelements.append(r["aggregation"]["type"] + "(" + prefix + r["axis"][axisName]["fields"][0] + ")" + " as " + axisName )
            else:
                sq = " "
                sqfields = []
                for field in r["axis"][axisName]["fields"]:
                    sqfields.append(prefix + field)
                sq += " || '|' || ".join(sqfields)
                sq += " as " + axisName
                selectelements.append(sq)
                groupelements.append( axisName )

            for field in r["axis"][axisName]["fields"]:
                fromelement = prefix + field.split(".")[0]
                if fromelement not in fromelements:
                    fromelements.append(fromelement)
                tmpjoinelements.append(field.split(".")[0])


        for axisName in ["color"]:
            if r["axis"][axisName] == None:
                continue
            if r["axis"][axisName]["field"] not in available_names:
                print("unknown fieldname: %s" % r["axis"][axisName]["field"])
                return HttpResponse("unknown: %s" % r["axis"][axisName]["field"])

            if axisName == r["aggregation"]["axis"]:
                selectelements.append(r["aggregation"]["type"] + "(" + prefix + r["axis"][axisName]["field"] + ")" + " as " + axisName )
            else:
                selectelements.append(prefix + r["axis"][axisName]["field"] + " as " + axisName )
                groupelements.append( axisName )

            fromelement = prefix + r["axis"][axisName]["field"].split(".")[0]
            if fromelement not in fromelements:
                fromelements.append(fromelement)
            tmpjoinelements.append(r["axis"][axisName]["field"].split(".")[0])




        joinorderElements = ["EpisodeNoisyExecution", "Episode", "Experiment", "ExperimentSet", "Optimiser", "Architecture", "Environment"]
        while len(tmpjoinelements) > 0:
            for joinorderElement in joinorderElements:
                if joinorderElement in tmpjoinelements:
                    for e in [t for t in tmpjoinelements if t != joinorderElement] :
                        subitems = joinorderElements[joinorderElements.index(joinorderElement)+1:]
                        if e in ["Episode", "Experiment", "ExperimentSet", "Optimiser", "Architecture", "Environment"]:
                            joinelements.append(prefix + joinorderElement + "."  + e + "_id" +  " = " + prefix + e + ".id")
                            tmpjoinelements.remove(e)
                    tmpjoinelements.remove(joinorderElement)
                    break
                
        query = 'SELECT %s FROM %s WHERE %s GROUP BY %s' % ( 
            ", ".join(selectelements), 
            ", ".join(fromelements), 
            " AND ".join(joinelements),
            ", ".join(groupelements),
        )
        #print(query)
        cursor.execute(query)
        rows = cursor.fetchall()

        j = {
            "x" : [],
            "y" : [],
            "z" : [],
            "color" : [],
        }

        active_axis = [a for a in ["x","y","z","color"] if r["axis"][a] != None ]
        for row in rows:            
            for index, axis in enumerate(active_axis):
                j[axis].append(row[index])
            
        return JsonResponse(j)


    
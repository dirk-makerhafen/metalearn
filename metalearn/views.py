from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from django.urls import reverse
from django.http import JsonResponse
import random
import redis
import json
import time
import numpy
from django.db import connection
import datetime
import uuid
# Create your views here.

from django.http import HttpResponse

from . import models


def dashboard(request):
    template = loader.get_template('dashboard.html')
    return HttpResponse(template.render({}, request))


# client can request preferedEpisodeIds if he has those episodes in cache so no additional download of weights is needed
def getEpisodeNoisyExecution(request, preferedEpisodeIds = ''):
    if preferedEpisodeIds != "":
        preferedEpisodeIds = [int(x) for x in preferedEpisodeIds.split(",")]
    
    lock = "%s" % uuid.uuid4()

    locked = 0

    episodeNoisyExecutions = []
    if len(preferedEpisodeIds) > 0:
        episodeNoisyExecutions = models.EpisodeNoisyExecution.objects           \
            .filter(status = "idle")                            \
            .filter(episode__status = "active")                \
            .filter(episode__id__in = preferedEpisodeIds)[:3]

    if len(episodeNoisyExecutions) == 0:
        episodeNoisyExecutions = models.EpisodeNoisyExecution.objects           \
            .filter(status = "idle")                            \
            .filter(episode__status = "active")[:3]

    episodeNoisyExecutions = list(episodeNoisyExecutions)

    if len(episodeNoisyExecutions) > 0:
        random.shuffle( episodeNoisyExecutions )
        for episodeNoisyExecution in episodeNoisyExecutions:
            locked = models.EpisodeNoisyExecution.objects                   \
                .filter( id = episodeNoisyExecution.id, status = "idle", )   \
                .update(status = "locked", lock = lock, client =  "some client")
            if locked == 1:
                episodeNoisyExecutionJson = {
                    "id"                : episodeNoisyExecution.id,
                    "lock"              : lock,
                    "noiseseed"         : episodeNoisyExecution.noiseseed,
                    "episode.id"        : episodeNoisyExecution.episode.id,
                    "environment.name"  : episodeNoisyExecution.environment.name,
                    "architecture.name" : episodeNoisyExecution.architecture.name,
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
    "Architecture.created",
    "Architecture.name",
    "Optimiser.created",
    "Optimiser.name",
    "ExperimentSet.id",
    "ExperimentSet.created",
    "ExperimentSet.name",
    "ExperimentSet.status",
    "ExperimentSet.max_Episodes",
    "ExperimentSet.min_EpisodeNoisyExecutions",
    "ExperimentSet.max_EpisodeNoisyExecutions",
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


    
import random
import requests
import json
import datetime
import time
import traceback
import binascii
import hashlib
import os
import uuid
import numpy
import redis

from django.db import transaction
from django.db import connection
from django.db import models
from django.db.models import F, Q
from django.utils import timezone

from . import  tasks

from .ml.optimisers import all_optimisers


ExperimentSet_STATUS_ENUM = ( ( "active", "active" ), ( "done", "done"), )
Experiment_STATUS_ENUM    = ( ( "active", "active" ), ( "done", "done"), )
Episode_STATUS_ENUM       = ( ( "active", "active" ), ( "done", "done"), )
EpisodeNoisyExecution_STATUS_ENUM = (("idle" , "idle"),("locked" , "locked"),("done" , "done"),)   


def getNoiseSeed():
    return random.randint(0,2**31)
def getUUID():
    return "%s" % uuid.uuid4()


class Environment(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    name     = models.CharField( max_length=200,default="", unique=True)
    description = models.CharField( max_length=200,default="") 
    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem

    # experimentSets -> ExperimentSet
    # experiments -> Experiment
    # noisyExecutions -> EpisodeNoisyExecution
    # episodes -> Episode

    def __str__(self):
        return "Env '%s' Id:%s" % (self.name, self.id) 


class Architecture(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    name     = models.CharField( max_length=200,default="", unique=True)
    description = models.CharField( max_length=200,default="") 
    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem

    # experimentSets -> ExperimentSet
    # experiments -> Experiment
    # noisyExecutions -> EpisodeNoisyExecution
    # episodes -> Episode

    def __str__(self):
        return "Arch '%s' Id:%s" % (self.name, self.id) 


class Optimiser(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    name     = models.CharField( max_length=200,default="", unique=True)
    description = models.CharField( max_length=200,default="") 
    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem

    # experimentSets -> ExperimentSet

    def __str__(self):
        return "Opt '%s' Id:%s" % (self.name, self.id) 


class ExperimentSet(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  

    public = models.BooleanField(default=True)  # is public via web api for clients to execute

    name     = models.CharField( max_length=200,default="")
    description     = models.CharField( max_length=200,default="")
    
    status = models.CharField(max_length= 200, choices=ExperimentSet_STATUS_ENUM, default="active")
    timespend =  models.FloatField(default = 0) # sum of Experiment.timespend ,  calculated on_ExperimentSet_done

    environments  = models.ManyToManyField(Environment , related_name='experimentSets', through="ExperimentSetToEnvironment")
    architectures = models.ManyToManyField(Architecture, related_name='experimentSets', through="ExperimentSetToArchitecture")
    optimisers    = models.ManyToManyField(Optimiser   , related_name='experimentSets', through="ExperimentSetToOptimiser")

    max_Episodes = models.BigIntegerField(default=10) #max number of episodes per experiment

    # actual values are selected by optimiser between min and max
    episodeNoisyExecutions_count_min = models.BigIntegerField(default=10)  # nr of NoisyExecutions per Episode per Experiment
    episodeNoisyExecutions_count_max = models.BigIntegerField(default=100)  # 
    episodeNoisyExecution_timespend_min = models.BigIntegerField(default=120)  # # time per NoisyExecutions, in seconds
    episodeNoisyExecution_timespend_max = models.BigIntegerField(default=120)  # 
    episodeNoisyExecution_steps_min = models.BigIntegerField(default=10000)  #  steps per NoisyExecutions
    episodeNoisyExecution_steps_max = models.BigIntegerField(default=10000)  #
    
    #experiments -> Experiment
    
    def save(self, *args, **kwargs):
        isNew = self.id == None
        super(ExperimentSet, self).save(*args, **kwargs)      

        transaction.commit()

        if isNew == True:
            tasks.on_ExperimentSet_created.delay(self.id)
        

class ExperimentSetToEnvironment(models.Model):
    matrixexperiment = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='environments_set')
    environment = models.ForeignKey(Environment, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)
class ExperimentSetToArchitecture(models.Model):
    matrixexperiment = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='architectures_set')
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)
class ExperimentSetToOptimiser(models.Model):
    matrixexperiment = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='optimisers_set')
    optimiser = models.ForeignKey(Optimiser, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)


class Experiment(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    
    public = models.BooleanField(default=True)  # is public via web api for clients to execute

    status    = models.CharField(max_length= 200, choices=Experiment_STATUS_ENUM, default="new")
    timespend =  models.FloatField(default = 0) # sum of Episode.timespend ,  calculated on_Experiment_done

    environment  = models.ForeignKey(Environment , on_delete=models.CASCADE, related_name='experiments',db_index=True)
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE, related_name='experiments',db_index=True)
    optimiser    = models.ForeignKey(Optimiser   , on_delete=models.CASCADE, related_name='experiments',db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='experiments',db_index=True)
    
    # episodes -> Episode
    # noisyExecutions -> EpisodeNoisyExecution

    def save(self, *args, **kwargs):
        isNew = self.id == None
        super(Experiment, self).save(*args, **kwargs)

        transaction.commit()

        if isNew == True:
            tasks.on_Experiment_created.delay(self.id, self.experimentSet.id)
        
              
class Episode(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    
    public = models.BooleanField(default=True)  # is public via web api for clients to execute

    status  = models.CharField(max_length= 200, choices=Episode_STATUS_ENUM, default="active")
    version = models.BigIntegerField(default = 1) # set on creation of next episode via on_Episode_done 
    episodeNoisyExecutions_count      =  models.BigIntegerField(default = 0) # actual max number of noisyExecutions for this episode, generated 
    # between experimentSet.episodeNoisyExecutions_count_min and experimentSet.episodeNoisyExecutions_count_max via a factor given by the optimiser on_Experiment_created and on_Episode_done
    episodeNoisyExecution_timespend =  models.BigIntegerField(default = 0)
    episodeNoisyExecution_steps     = models.BigIntegerField(default = 0)

    timespend    =  models.FloatField(default = 0) # sum of noisyExecutions.timespend ,  calculated on_Episode_done
    fitness_min  =  models.FloatField(default = 0) # min fitness of noisyExecutions,  calculated on_Episode_done
    fitness_max  =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_avg  =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_median =  models.FloatField(default = 0) # calculated on_Episode_done
   
    hasFolder = models.BooleanField(default=False)  # does weightNoise and Optimiser data exist on harddisk?

    environment = models.ForeignKey(Environment, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    optimiser    = models.ForeignKey(Optimiser   , on_delete=models.CASCADE, related_name='episodes',db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    experiment  = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='episodes',db_index=True)

    #noisyExecutions -> EpisodeNoisyExecution

    def __str__(self):
        return "Episode Id:%s, version:%s" % (self.id, self.version) 


    @property
    def filepath(self):
        p = "metalearn/ml/data/set_%s/exp_%s/ep_%s/" % (self.experimentSet.id, self.experiment.id, self.version)
        if not os.path.exists(p):
            try: 
                os.makedirs(p)
                self.hasFolder = True
            except:
                pass
        return p

    @property
    def weightsNoiseFile(self):
        return "%s/wn.npy" % self.filepath

    @property
    def optimiserDataFile(self):
        return "%s/od.dat" % self.filepath

    @property
    def weightsNoise(self):
        return numpy.load(self.weightsNoiseFile)

    @weightsNoise.setter
    def weightsNoise(self, weightsNoise):
        numpy.save(self.weightsNoiseFile, weightsNoise)

    @property
    def optimiserData(self):
        return open(self.optimiserDataFile,"rb").read()

    @optimiserData.setter
    def optimiserData(self, data):
        open(self.optimiserDataFile,"wb").write(data)

    def save(self, *args, **kwargs):
        isNew = self.id == None

        super(Episode, self).save(*args, **kwargs)

        transaction.commit()

        if isNew == True:
            tasks.on_Episode_created.delay(self.id, self.experiment.id, self.experimentSet.id)


class EpisodeNoisyExecution(models.Model):
    id        = models.BigAutoField(primary_key=True)
    created   = models.DateTimeField('created',auto_now_add=True)
    updated   = models.DateTimeField('updated',auto_now=True)  

    status    = models.CharField(max_length= 200, choices=EpisodeNoisyExecution_STATUS_ENUM, default="idle")
    lock   = models.CharField(max_length=100, default="", blank=True) # some random hash if this is locked by some client for execution, or "" if not locked
    client = models.CharField(max_length=100, default="", blank=True) # the client owning this lock, or "" if not locked or done

    number = models.BigIntegerField(default = 0) # 0..N   number of execution within Episode, index  

    noiseseed = models.BigIntegerField(default=getNoiseSeed) # some random integer number so noisepattern can be regenerated

    timespend = models.FloatField(default = 0) # number of seconds this task spend running on a client, including weightdownloads/init 
    steps = models.BigIntegerField(default = 0) # number of steps that were executed, aka nr of frames seen, game steps taken, images seen, hotdogs classified and so on
    
    fitness   = models.FloatField(default = 0) # actual reward returned by whatever was executed
    fitness_rank   = models.FloatField(default = 0) # fitness rank within episode, 0..1, 0 worst, 1 best. Calculated on_Episode_done 

    environment      = models.ForeignKey(Environment     , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    architecture     = models.ForeignKey(Architecture    , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    optimiser        = models.ForeignKey(Optimiser       , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    experiment       = models.ForeignKey(Experiment      , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    episode   = models.ForeignKey(Episode  , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)

    def save(self, *args, **kwargs):
        isNew = self.id == None

        super(EpisodeNoisyExecution, self).save(*args, **kwargs)
        
        transaction.commit()

        #if isNew == True:
        #    tasks.on_EpisodeNoisyExecution_created.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id)
        

    def setResult(self, data):
        print("setResult %s" % data)
        self.fitness = data["fitness"]
        self.timespend = data["timespend"]
        self.steps = data["steps"]
        self.status = "done"
        self.save()

        transaction.commit()

        tasks.on_NoisyExecution_done.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id)



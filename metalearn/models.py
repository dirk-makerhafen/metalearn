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
from django.db.transaction import on_commit
from django.utils import timezone

from . import  tasks

import metalearn.ml.environments
import metalearn.ml.architectures
import metalearn.ml.optimisers

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
    name     = models.CharField( max_length=200,default="")
    description = models.CharField( max_length=200,default="") 

    classname = models.CharField( max_length=200,default="") 
    classargs = models.CharField( max_length=2000,default="[]") 

    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem
 
    # experimentSets -> ExperimentSet
    # experiments -> Experiment
    # episodes -> Episode
    # noisyExecutions -> EpisodeNoisyExecution

    @property
    def experimentSets_count(self):
        return self.experimentSets.all().count()
    @property
    def experiments_count(self):
        return self.experiments.all().count()
    @property
    def episodes_count(self):
        return self.episodes.all().count()
    @property
    def noisyExecutions_count(self):
        return self.noisyExecutions.all().count()


    class Meta:
        unique_together = ('classname', 'classargs',)

    def getInstance(self):
        _class = getattr(metalearn.ml.environments, self.classname)
        a = {}
        print(self.classargs)
        for v in json.loads(self.classargs):
            a[v[0]] = v[1]
        return _class(**a)

    def __str__(self):
        return "%s:%s" % (self.id, self.name) 


class Architecture(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    name     = models.CharField( max_length=200,default="")
    description = models.CharField( max_length=200,default="") 

    classname = models.CharField( max_length=200,default="") 
    classargs = models.CharField( max_length=2000,default="[]") 

    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem

    # experimentSets -> ExperimentSet
    # experiments -> Experiment
    # episodes -> Episode
    # noisyExecutions -> EpisodeNoisyExecution

    @property
    def experimentSets_count(self):
        return self.experimentSets.all().count()
    @property
    def experiments_count(self):
        return self.experiments.all().count()
    @property
    def episodes_count(self):
        return self.episodes.all().count()
    @property
    def noisyExecutions_count(self):
        return self.noisyExecutions.all().count()


    class Meta:
        unique_together = ('classname', 'classargs',)

    def getInstance(self):
        _class = getattr(metalearn.ml.architectures, self.classname)
        a = {}
        for v in json.loads(self.classargs):
            a[v[0]] = v[1]
        return _class(**a)

    def __str__(self):
        return "%s:%s" % (self.id, self.name) 


class Optimiser(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    name     = models.CharField( max_length=200,default="")
    description = models.CharField( max_length=200,default="") 

    classname = models.CharField( max_length=200,default="") 
    classargs = models.CharField( max_length=2000,default="[]") 

    broken = models.BooleanField(default=False)  # is set if env is missing on filesystem

    # experimentSets -> ExperimentSet
    # experiments -> Experiment
    # episodes -> Episode
    # noisyExecutions -> EpisodeNoisyExecution

    @property
    def experimentSets_count(self):
        return self.experimentSets.all().count()
    @property
    def experiments_count(self):
        return self.experiments.all().count()
    @property
    def episodes_count(self):
        return self.episodes.all().count()
    @property
    def noisyExecutions_count(self):
        return self.noisyExecutions.all().count()

    class Meta:
        unique_together = ('classname', 'classargs',)

    def getInstance(self):
        _class = getattr(metalearn.ml.optimisers, self.classname)
        a = {}
        for v in json.loads(self.classargs):
            a[v[0]] = v[1]
        return _class(**a)
        

    def __str__(self):
        return "%s:%s" % (self.id, self.name) 


class ExperimentSet(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  

    # Settings for this ExperimentSet
    public = models.BooleanField(default=True)  # is public via web api for clients to execute
    name        = models.CharField( max_length=200,default="")
    description = models.CharField( max_length=200,default="")
    environments  = models.ManyToManyField(Environment , related_name='experimentSets', through="ExperimentSetToEnvironment")
    architectures = models.ManyToManyField(Architecture, related_name='experimentSets', through="ExperimentSetToArchitecture")
    optimisers    = models.ManyToManyField(Optimiser   , related_name='experimentSets', through="ExperimentSetToOptimiser")

    # Settings for Experiments created by this ExperimentSet
    subsettings_Experiments_max = models.BigIntegerField(default=10000) # limit nr of experiments if its larger len(environments) * len(architectures) * len(optimisers)
    
    # Settings for Episodes created by Experiments created by this ExperimentSet
    subsettings_Episodes_max = models.BigIntegerField(default=10) # max number of episodes per experiment

    # Settings for EpisodeNoisyExecution created by Episodes created by Experiments created by this ExperimentSet
    subsettings_EpisodeNoisyExecutions_min = models.BigIntegerField(default=10) # 
    subsettings_EpisodeNoisyExecutions_max = models.BigIntegerField(default=100 ) # nr of EpisodeNoisyExecution per Episode per Experiment
    subsettings_EpisodeNoisyExecutions_min_steps = models.BigIntegerField(default=100)   # 
    subsettings_EpisodeNoisyExecutions_max_steps = models.BigIntegerField(default=10000)  # steps per NoisyExecutions
    subsettings_EpisodeNoisyExecutions_min_timespend = models.BigIntegerField(default=10) # 
    subsettings_EpisodeNoisyExecutions_max_timespend = models.BigIntegerField(default=120)# max time per NoisyExecutions, in seconds
    
    # stats
    status = models.CharField(max_length= 200, choices=ExperimentSet_STATUS_ENUM, default="active")
    timespend =  models.FloatField(default = 0) # sum of Experiment.timespend ,  calculated on_ExperimentSet_done
    on_created_executed = models.BooleanField(default=False) # set after task.on_ExperimentSet_created is done
    on_done_executed = models.BooleanField(default=False)# set after task.on_ExperimentSet_done is done

    @property
    def experiments_count(self):
        return self.experiments.all().count()

    #experiments -> Experiment

    def __str__(self):
        return "%s:%s" % (self.id, self.name) 

    def save(self, *args, **kwargs):
        isNew = self.id == None
        super(ExperimentSet, self).save(*args, **kwargs)      

        if isNew == True:
            on_commit(lambda: tasks.on_ExperimentSet_created.delay(self.id))
            
        
class ExperimentSetToEnvironment(models.Model):
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='environments_set')
    environment = models.ForeignKey(Environment, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)
class ExperimentSetToArchitecture(models.Model):
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='architectures_set')
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)
class ExperimentSetToOptimiser(models.Model):
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='optimisers_set')
    optimiser = models.ForeignKey(Optimiser, on_delete=models.CASCADE)
    nr_of_instances = models.BigIntegerField(default=1)

class Experiment(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  

    # Settings for this Experiment
    public = models.BooleanField(default=True)  # is public via web api for clients to execute
    
    environment  = models.ForeignKey(Environment , on_delete=models.CASCADE, related_name='experiments',db_index=True)
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE, related_name='experiments',db_index=True)
    optimiser    = models.ForeignKey(Optimiser   , on_delete=models.CASCADE, related_name='experiments',db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='experiments',db_index=True)
    
    # Stats
    status    = models.CharField(max_length= 200, choices=Experiment_STATUS_ENUM, default="new")
    timespend =  models.FloatField(default = 0) # sum of Episode.timespend ,  calculated on_Experiment_done
    fitness_min   =  models.FloatField(default = 0) # min fitness of noisyExecutions,  calculated on_Experiment_done
    fitness_max   =  models.FloatField(default = 0) # calculated on_Experiment_done
    fitness_avg   =  models.FloatField(default = 0) # calculated on_Experiment_done
    fitness_median=  models.FloatField(default = 0) # calculated on_Experiment_done
    fitness_top10 =  models.FloatField(default = 0) # calculated on_Experiment_done
    on_created_executed = models.BooleanField(default=False) # set after task.on_Experiment_created is done
    on_done_executed = models.BooleanField(default=False)# set after task.on_Experiment_done is done

    # episodes -> Episode
    # noisyExecutions -> EpisodeNoisyExecution

    @property
    def episodes_count(self):
        return self.episodes.all().count()
    @property
    def noisyExecutions_count(self):
        return self.noisyExecutions.all().count()


    def __str__(self):
        return "Id:%s" % (self.id) 

    def save(self, *args, **kwargs):
        isNew = self.id == None
        super(Experiment, self).save(*args, **kwargs)

        if isNew == True:
            on_commit(lambda: tasks.on_Experiment_created.delay(self.id, self.experimentSet.id))
        
              
class Episode(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    
    # Settings for this Episode
    version = models.BigIntegerField(default = 1) # set on creation of next episode via on_Episode_done 
    
    environment = models.ForeignKey(Environment, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    architecture = models.ForeignKey(Architecture, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    optimiser    = models.ForeignKey(Optimiser   , on_delete=models.CASCADE, related_name='episodes',db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    experiment  = models.ForeignKey(Experiment, on_delete=models.CASCADE, related_name='episodes',db_index=True)
    
    # Settings for EpisodeNoisyExecutions created by this Episode
    subsettings_EpisodeNoisyExecutions_max = models.BigIntegerField(default=100 ) # nr of EpisodeNoisyExecution per Episode per Experiment  #actual max number of noisyExecutions for this episode, generated, between experimentSet.episodeNoisyExecutions_count_min and experimentSet.episodeNoisyExecutions_count_max via a factor given by the optimiser on_Experiment_created and on_Episode_done
    subsettings_EpisodeNoisyExecutions_max_steps = models.BigIntegerField(default=10000)  # steps per NoisyExecutions
    subsettings_EpisodeNoisyExecutions_max_timespend = models.BigIntegerField(default=120)# max time per NoisyExecutions, in seconds
    
    # Stats
    public = models.BooleanField(default=True)  # is public via web api for clients to execute
    status  = models.CharField(max_length= 200, choices=Episode_STATUS_ENUM, default="active")
    hasFolder = models.BooleanField(default=False)  # does weightNoise and Optimiser data exist on harddisk?

    timespend     =  models.FloatField(default = 0) # sum of noisyExecutions.timespend ,  calculated on_Episode_done
    fitness_min   =  models.FloatField(default = 0) # min fitness of noisyExecutions,  calculated on_Episode_done
    fitness_max   =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_avg   =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_median=  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_top10 =  models.FloatField(default = 0) # calculated on_Episode_done
    on_created_executed = models.BooleanField(default=False) # set after task.on_Episode_created is done
    on_done_executed = models.BooleanField(default=False)# set after task.on_Episode_done is done


    class Meta():
        ordering = ['version']


    #noisyExecutions -> EpisodeNoisyExecution

    @property
    def noisyExecutions_count(self):
        return self.noisyExecutions.all().count()


    def __str__(self):
        return "Id:%s Version:%s" % (self.id, self.version) 


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
    def optimiserMetaDataFile(self):
        return "%s/od.meta.dat" % self.filepath

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

    @property
    def optimiserMetaData(self):
        return open(self.optimiserMetaDataFile,"rb").read()

    @optimiserMetaData.setter
    def optimiserMetaData(self, data):
        open(self.optimiserMetaDataFile,"wb").write(data)


    def save(self, *args, **kwargs):
        isNew = self.id == None

        super(Episode, self).save(*args, **kwargs)

        if isNew == True:
            on_commit(lambda: tasks.on_Episode_created.delay(self.id, self.experiment.id, self.experimentSet.id))


class EpisodeNoisyExecution(models.Model):
    id        = models.BigAutoField(primary_key=True)
    created   = models.DateTimeField('created',auto_now_add=True)
    updated   = models.DateTimeField('updated',auto_now=True)  

    # Settings for this EpisodeNoisyExecution
    number = models.BigIntegerField(default = 0) # 0..N   number of execution within Episode, index  
    noiseseed = models.BigIntegerField(default=getNoiseSeed) # some random integer number so noisepattern can be regenerated

    environment      = models.ForeignKey(Environment     , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    architecture     = models.ForeignKey(Architecture    , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    optimiser        = models.ForeignKey(Optimiser       , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    experimentSet    = models.ForeignKey(ExperimentSet   , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    experiment       = models.ForeignKey(Experiment      , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    episode   = models.ForeignKey(Episode  , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)


    # Lock
    lock   = models.CharField(max_length=100, default="", blank=True) # some random hash if this is locked by some client for execution, or "" if not locked
    client = models.CharField(max_length=100, default="", blank=True) # the client owning this lock, or "" if not locked or done

    # Stats
    status = models.CharField(max_length= 200, choices=EpisodeNoisyExecution_STATUS_ENUM, default="idle")
    timespend = models.FloatField(default = 0) # number of seconds this task spend running on a client, including weightdownloads/init 
    steps = models.BigIntegerField(default = 0) # number of steps that were executed, aka nr of frames seen, game steps taken, images seen, hotdogs classified and so on
    fitness   = models.FloatField(default = 0) # actual reward returned by whatever was executed
    fitness_rank   = models.FloatField(default = 0) # fitness rank within episode, 0..1, 0 worst, 1 best. Calculated on_Episode_done 
    fitness_calc_key = models.CharField(max_length=100, default="", blank=True) # 
    fitness_calc_value   = models.FloatField(default = 0)
    # if fitness_calc_key and value are given, fitness and fitness_rank are calculated on_Episode_done as the center (-1..1) rank of this fitness_calc_value in comparision 
    # to ALL other EpisodeNoisyExecution that have the same fitness_calc_key. this value is outdated right after calculating it, so use only one or recalc. 
    on_created_executed = models.BooleanField(default=False) # set after task.on_EpisodeNoisyExecution_created is done
    on_done_executed = models.BooleanField(default=False)# set after task.on_EpisodeNoisyExecution_done is done


    def save(self, *args, **kwargs):
        isNew = self.id == None

        super(EpisodeNoisyExecution, self).save(*args, **kwargs)
        
        #if isNew == True:
        #    on_commit(lambda: tasks.on_EpisodeNoisyExecution_created.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id))
        

    def setResult(self, data):
        if self.status != "done":
            print("setResult %s" % data)
            self.fitness = data["fitness"]
            self.timespend = data["timespend"]
            self.steps = data["steps"]
            if "fitness_calc_key" in data:
                self.fitness_calc_key = data["fitness_calc_key"]
            if "fitness_calc_value" in data:
                self.fitness_calc_value = data["fitness_calc_value"]
            self.status = "done"
            self.save()

            on_commit(lambda: tasks.on_NoisyExecution_done.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id))

    @staticmethod
    def getOneIdleLocked(client_id, public=True, experimentSetIds = [], experimentIds = [], episodeIds = []):
        lock = "%s" % uuid.uuid4()

        locked = 0

        episodeNoisyExecutions = []

        q = EpisodeNoisyExecution.objects \
            .filter(status = "idle")                                  \
            .filter(episode__status = "active")                       \
            .filter(episode__public = public)

        if len(experimentSetIds) > 0:
            q = q.filter(experimentSet_id__in = experimentSetIds)
        if len(experimentIds) > 0:
            q = q.filter(experiment_id__in = experimentIds)
        if len(episodeIds) > 0:
            q = q.filter(episode_id__in = episodeIds)

        episodeNoisyExecutions = list(q.order_by("number")[:5])

        if len(episodeNoisyExecutions) > 0:
            random.shuffle( episodeNoisyExecutions )
            for episodeNoisyExecution in episodeNoisyExecutions:
                locked = EpisodeNoisyExecution.objects                   \
                    .filter( id = episodeNoisyExecution.id, status = "idle", )   \
                    .update(status = "locked", lock = lock, client =  client_id)
                if locked == 1:
                    return episodeNoisyExecution, lock
        
        return None, None

'''

class OptimiserTraining(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    
    # Settings for this OptimiserTraining
    name     = models.CharField( max_length=200,default="")
    description     = models.CharField( max_length=200,default="")    
    optimiser    = models.ForeignKey(Optimiser)
    environments  = models.ManyToManyField(Environment , related_name='experimentSets', through="ExperimentSetToEnvironment")
    architectures = models.ManyToManyField(Architecture, related_name='experimentSets', through="ExperimentSetToArchitecture")

    # Settings for OptimiserTrainingEpisode created by this OptimiserTraining
    subsettings_OptimiserTrainingEpisodes_parallel_environments = models.BigIntegerField(default=5)
    subsettings_OptimiserTrainingEpisodes_parallel_environments_delta = models.BigIntegerField(default=1)# every episode replace x random selected envs with y others
    subsettings_OptimiserTrainingEpisodes_parallel_architectures = models.BigIntegerField(default=5)
    subsettings_OptimiserTrainingEpisodes_parallel_architectures_delta = models.BigIntegerField(default=1)  
    subsettings_OptimiserTrainingEpisodes_max = models.BigIntegerField(default=10) # max number of episodes per experiment

    # Settings for OptimiserTrainingEpisodeNoisyExecution created by OptimiserTrainingEpisode created by this OptimiserTraining
    subsettings_OptimiserTrainingEpisodeNoisyExecutions_min = models.BigIntegerField(default=100 ) # nr of EpisodeNoisyExecution per Episode per Experiment  #actual max number of noisyExecutions for this episode, generated, between experimentSet.episodeNoisyExecutions_count_min and experimentSet.episodeNoisyExecutions_count_max via a factor given by the optimiser on_Experiment_created and on_Episode_done
    subsettings_OptimiserTrainingEpisodeNoisyExecutions_max = models.BigIntegerField(default=100 ) # nr of EpisodeNoisyExecution per Episode per Experiment  #actual max number of noisyExecutions for this episode, generated, between experimentSet.episodeNoisyExecutions_count_min and experimentSet.episodeNoisyExecutions_count_max via a factor given by the optimiser on_Experiment_created and on_Episode_done

    
    # Settings for ExperimentSets created as Episode by this OptimiserTraining
    
    # Settings for Experiments created by ExperimentSets created as Episode by this OptimiserTraining

    # Settings for Episodes created by Experiments created by ExperimentSets created as Episode by this OptimiserTraining
    subexperimentsettings_Episodes_max = models.BigIntegerField(default=10) # max number of episodes per experiment
    subexperimentsettings_Episodes_min = models.BigIntegerField(default=10) # 

    # Settings for EpisodeNoisyExecution created by Episodes created by Experiments created by ExperimentSets created as Episode by this OptimiserTraining 
    subexperimentsettings_EpisodeNoisyExecutions_max = models.BigIntegerField(default=10 ) # nr of EpisodeNoisyExecution per Episode per Experiment
    subexperimentsettings_EpisodeNoisyExecutions_min = models.BigIntegerField(default=100) # 
    subexperimentsettings_EpisodeNoisyExecutions_max_steps = models.BigIntegerField(default=10000)  # steps per NoisyExecutions
    subexperimentsettings_EpisodeNoisyExecutions_min_steps = models.BigIntegerField(default=10000)   # 
    subexperimentsettings_EpisodeNoisyExecutions_max_timespend = models.BigIntegerField(default=120)# max time per NoisyExecutions, in seconds
    subexperimentsettings_EpisodeNoisyExecutions_min_timespend = models.BigIntegerField(default=120) # 

    # Stats
    status = models.CharField(max_length= 200, choices=ExperimentSet_STATUS_ENUM, default="active")
    


    def save(self, *args, **kwargs):
        isNew = self.id == None
        super(OptimiserTraining, self).save(*args, **kwargs)      

        if isNew == True:
            on_commit(lambda: tasks.on_OptimiserTraining_created.delay(self.id))
  
 
class OptimiserTrainingEpisode(models.Model):
    id       = models.BigAutoField(primary_key=True)
    created  = models.DateTimeField('created',auto_now_add=True)
    updated  = models.DateTimeField('updated',auto_now=True)  
    
    # Settings for OptimiserTrainingEpisode
    public = models.BooleanField(default=True)  # is public via web api for clients to execute
    version = models.BigIntegerField(default = 1) # set on creation of next episode via on_Episode_done 
    optimiserTraining = models.ForeignKey(OptimiserTraining , on_delete=models.CASCADE, related_name='optimiserTrainingEpisodes', db_index=True)
    experimentSet     = models.OneToOne(ExperimentSet     , on_delete=models.CASCADE, related_name='optimiserTrainingEpisode', db_index=True)

    # Settings for OptimiserTrainingEpisodeNoisyExecutions
    subsettings_OptimiserTrainingEpisodeNoisyExecutions_max = models.BigIntegerField(default=100 ) # actual max number of noisyExecutions for this episode, generated 
    # between experimentSet.episodeNoisyExecutions_count_min and experimentSet.episodeNoisyExecutions_count_max via a factor given by the optimiser on_Experiment_created and on_Episode_done


    # Stats
    status  = models.CharField(max_length= 200, choices=Episode_STATUS_ENUM, default="active")
    hasFolder = models.BooleanField(default=False)  # does weightNoise and Optimiser data exist on harddisk?
    timespend    =  models.FloatField(default = 0) # sum of noisyExecutions.timespend ,  calculated on_Episode_done
    fitness_min  =  models.FloatField(default = 0) # min fitness of noisyExecutions,  calculated on_Episode_done
    fitness_max  =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_avg  =  models.FloatField(default = 0) # calculated on_Episode_done
    fitness_median =  models.FloatField(default = 0) # calculated on_Episode_done   


class OptimiserTrainingEpisodeNoisyExecution(models.Model):
    id        = models.BigAutoField(primary_key=True)
    created   = models.DateTimeField('created',auto_now_add=True)
    updated   = models.DateTimeField('updated',auto_now=True)  

    # Settings for OptimiserTrainingEpisodeNoisyExecution
    number = models.BigIntegerField(default = 0) # 0..N   number of execution within Episode, index  
    noiseseed = models.BigIntegerField(default=getNoiseSeed) # some random integer number so noisepattern can be regenerated

    optimiserTraining        = models.ForeignKey(OptimiserTraining        , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    optimiserTrainingEpisode = models.ForeignKey(OptimiserTrainingEpisode , on_delete=models.CASCADE, related_name='noisyExecutions', db_index=True)
    experiment    = models.OneToOne(Experiment      , on_delete=models.CASCADE, related_name='optimiserTrainingEpisodeNoisyExecution' , db_index=True)
    experimentSet = models.ForeignKey(ExperimentSet , on_delete=models.CASCADE, related_name='optimiserTrainingEpisodeNoisyExecutions', db_index=True)

    # Lock
    lock   = models.CharField(max_length=100, default="", blank=True) # some random hash if this is locked by some client for execution, or "" if not locked
    client = models.CharField(max_length=100, default="", blank=True) # the client owning this lock, or "" if not locked or done

    # Stats
    status    = models.CharField(max_length= 200, choices=EpisodeNoisyExecution_STATUS_ENUM, default="idle")
    timespend = models.FloatField(default = 0) # number of seconds this task spend running on a client, including weightdownloads/init 
    steps = models.BigIntegerField(default = 0) # number of steps that were executed, aka nr of frames seen, game steps taken, images seen, hotdogs classified and so on
    fitness   = models.FloatField(default = 0) # actual reward returned by whatever was executed
    fitness_rank   = models.FloatField(default = 0) # fitness rank within episode, 0..1, 0 worst, 1 best. Calculated on_Episode_done 

    def save(self, *args, **kwargs):
        isNew = self.id == None

        super(EpisodeNoisyExecution, self).save(*args, **kwargs)
        
        #if isNew == True:
        #    on_commit(lambda: tasks.on_EpisodeNoisyExecution_created.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id))
        

    def setResult(self, data):
        if self.status != "done":
            print("setResult %s" % data)
            self.fitness = data["fitness"]
            self.timespend = data["timespend"]
            self.steps = data["steps"]
            self.status = "done"
            self.save()

            on_commit(lambda: tasks.on_NoisyExecution_done.delay(self.id, self.episode.id, self.experiment.id, self.experimentSet.id))

    @classmethod
    def getOneIdleLocked(client_id, public=True, experimentSetIds = [], experimentIds = [], episodeIds = []):
        lock = "%s" % uuid.uuid4()

        locked = 0

        episodeNoisyExecutions = []

        q = models.EpisodeNoisyExecution.objects \
            .filter(status = "idle")                                  \
            .filter(episode__status = "active")                       \
            .filter(episode__public = public)

        if len(experimentSetIds) > 0:
            q = q.filter(experimentSet_id__in = experimentSetIds)
        if len(experimentIds) > 0:
            q = q.filter(experiment_id__in = experimentIds)
        if len(episodeIds) > 0:
            q = q.filter(episode_id__in = episodeIds)

        episodeNoisyExecutions = list(q.order_by("number")[:5])

        if len(episodeNoisyExecutions) > 0:
            random.shuffle( episodeNoisyExecutions )
            for episodeNoisyExecution in episodeNoisyExecutions:
                locked = models.EpisodeNoisyExecution.objects                   \
                    .filter( id = episodeNoisyExecution.id, status = "idle", )   \
                    .update(status = "locked", lock = lock, client =  client_id)
                if locked == 1:
                    return episodeNoisyExecution, lock
        
        return None, None

'''

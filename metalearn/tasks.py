
from celery import shared_task
from django.db.models import Q
import numpy
import time
import random
import pickle
from datetime import datetime, timedelta
from scipy.stats import rankdata
from django.db.transaction import on_commit
from django.db.transaction import commit


def rank(numbers):
    uniq_values = len(set(numbers))
    ranks = rankdata(numbers, method="dense")
    if uniq_values > 1:
        ranks0to1 = ( ranks - 1 ) / ( uniq_values - 1)
    else:
        ranks0to1 = ranks / 2
    return ranks0to1

def rank_and_center(numbers):
    ranks0to1 = rank(numbers)
    ranksMinus1To1 = ( ranks0to1 * 2 ) - 1
    return ranksMinus1To1



# ExperimentSet

@shared_task
def on_ExperimentSet_created(experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment

    experimentSet = ExperimentSet.objects.get(id=experimentSet_id) # because autocommit..
    
    combinations = []
    for environment_set in experimentSet.environments_set.all():
        for _ in range(0,environment_set.nr_of_instances):

            for architecture_set in experimentSet.architectures_set.all():
                for _ in range(0,architecture_set.nr_of_instances):

                    for optimiser_set in experimentSet.optimisers_set.all():
                        for _ in range(0,optimiser_set.nr_of_instances):

                            combinations.append( [ environment_set, architecture_set, optimiser_set])


    if len(combinations) > experimentSet.subsettings_Experiments_max:
        combinations = random.sample(combinations, experimentSet.subsettings_Experiments_max)

    for combination in combinations:
        environment_set, architecture_set, optimiser_set = combination
        experiment = Experiment()
        experiment.status = "active"
        experiment.public = experimentSet.public
        experiment.experimentSet = experimentSet
        experiment.environment  = environment_set.environment
        experiment.architecture = architecture_set.architecture
        experiment.optimiser    = optimiser_set.optimiser
        experiment.save()


@shared_task
def on_ExperimentSet_done(experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment
    from .models import EpisodeNoisyExecution

    experimentSet = ExperimentSet.objects.get(id=experimentSet_id)
    
    timespend = 0.0
    experiments = experimentSet.experiments.all()
    for experiment in experiments:  
        timespend += experiment.timespend

    experimentSet.timespend = timespend
    experimentSet.save()

    combinations_fitnesses = {}
    for experiment in experiments:  
        combinationid = "%s_%s" % ( experiment.environment.id, experiment.architecture.id )
        if combinationid not in combinations_fitnesses:
            combinations_fitnesses[combinationid] = []

        combinations_fitnesses[combinationid].append(experiment.fitness_top10)

    indexes = {}
    for key in combinations_fitnesses:
        indexes[key] = 0
        combinations_fitnesses[key] = rank_and_center(combinations_fitnesses[key])
        
    for experiment in experiments:  
        combinationid = "%s_%s" % ( experiment.environment.id, experiment.architecture.id ) 
        some_episode = experiment.episodes.all()[0]
        optimiserInstance = some_episode.optimiser.getInstance()
        optimiserInstance.reward(some_episode, combinations_fitnesses[combinationid][indexes[combinationid]])
        indexes[combinationid] += 1



# Experiment

@shared_task
def on_Experiment_created(experiment_id, experimentSet_id):
    from .models import Experiment
    from .models import Episode

    experiment = Experiment.objects.get(id=experiment_id)

    # init optimiser
    optimiserInstance = experiment.optimiser.getInstance()
    result = optimiserInstance.initialize(experiment.environment, experiment.architecture)  
    if result == "delay":
        print("Delaying Experiment create because no optimiser is available at the moment")
        on_Experiment_created(experiment_id, experimentSet_id).apply_async(countdown=60)
        return 


    # create first episode
    episode = Episode()
    episode.environment = experiment.environment
    episode.architecture = experiment.architecture
    episode.optimiser = experiment.optimiser
    episode.experimentSet = experiment.experimentSet
    episode.experiment = experiment
    episode.version = 1
    episode.public = experiment.public

    episode.weightsNoise, episode.optimiserMetaData, episode.optimiserData, count_factor, timespend_factor, steps_factor = result


    eset = episode.experimentSet          
    
    #factors are -1 .. 1 
    h = ( eset.subsettings_EpisodeNoisyExecutions_max           - eset.subsettings_EpisodeNoisyExecutions_min           ) / 2.0
    episode.subsettings_EpisodeNoisyExecutions_max           = eset.subsettings_EpisodeNoisyExecutions_min            +  h + ( h  * count_factor      ) 

    h = ( eset.subsettings_EpisodeNoisyExecutions_max_steps     - eset.subsettings_EpisodeNoisyExecutions_min_steps     ) / 2.0
    episode.subsettings_EpisodeNoisyExecutions_max_steps     = eset.subsettings_EpisodeNoisyExecutions_min_steps      +  h + ( h * steps_factor      )

    h = ( eset.subsettings_EpisodeNoisyExecutions_max_timespend - eset.subsettings_EpisodeNoisyExecutions_min_timespend ) / 2.0
    episode.subsettings_EpisodeNoisyExecutions_max_timespend = eset.subsettings_EpisodeNoisyExecutions_min_timespend  +  h + ( h * timespend_factor  )
    episode.save()


@shared_task
def on_Experiment_done(experiment_id, experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment
    from .models import Episode
    from .models import EpisodeNoisyExecution

    ids_fitnesses_timespend = EpisodeNoisyExecution.objects.filter(experiment_id = experiment_id).values_list('id',"fitness","timespend").distinct()
    fitnesses = [x[1] for x in ids_fitnesses_timespend]
    timesspend  = [x[2] for x in ids_fitnesses_timespend]

    experiment = Experiment.objects.get(id=experiment_id)
    # calc episode average/sum values
    experiment.timespend   = sum(timesspend)
    experiment.fitness_min   = min(fitnesses)
    experiment.fitness_max   = max(fitnesses)
    experiment.fitness_avg   = numpy.mean(fitnesses)
    experiment.fitness_median =  numpy.median(fitnesses)
    experiment.fitness_top10 =  numpy.mean(sorted(fitnesses, reverse=True)[0:10])
    experiment.save()

    experiments_to_go = Experiment.objects.filter(experimentSet_id = experimentSet_id).filter(~Q(status = "done")).count()
    if experiments_to_go > 0:
        return  

    experimentSet_done = ExperimentSet.objects.filter(id = experimentSet_id).filter(~Q(status = "done")).update(status="done")
    if experimentSet_done == 1:
        on_commit(lambda: on_ExperimentSet_done.delay(experimentSet_id))

    

# Episode

@shared_task
def on_Episode_created(episode_id, experiment_id, experimentSet_id):
    from .models import Episode
    from .models import EpisodeNoisyExecution

    episode = Episode.objects.get(id=episode_id)

    count = episode.noisyExecutions.count()

    for number in range(count,episode.subsettings_EpisodeNoisyExecutions_max):
        episodeNoisyExecution = EpisodeNoisyExecution()
        episodeNoisyExecution.environment = episode.environment
        episodeNoisyExecution.architecture = episode.architecture
        episodeNoisyExecution.optimiser = episode.optimiser
        episodeNoisyExecution.experimentSet = episode.experimentSet
        episodeNoisyExecution.experiment = episode.experiment
        episodeNoisyExecution.episode = episode
        episodeNoisyExecution.number = number
        episodeNoisyExecution.save()

@shared_task
def on_Episode_done(episode_id, experiment_id, experimentSet_id):
    from .models import Episode
    from .models import Experiment
    from .models import ExperimentSet
    from .models import EpisodeNoisyExecution

    current_episode = Episode.objects.get(id=episode_id)

    # calc fitness via fitness_calc_key if needed
    ids = [x[0] for x in current_episode.noisyExecutions.all().values_list('id').distinct()]
    fitness_calc_keys = current_episode.noisyExecutions.filter(~Q(fitness_calc_key = "")).values_list("fitness_calc_key").distinct()
    print(fitness_calc_keys)
    for fitness_calc_key in fitness_calc_keys:
        nes = EpisodeNoisyExecution.filter(fitness_calc_key = fitness_calc_key).values_list("id", "fitness_calc_value").distinct()
        fnes = [x[1] for x in nes]
        ranks = rank_and_center(fnes)
        for i in range(0,len(fnes)):
            if nes[i][0] in ids: # update only fitness of noisyExecutions of this episode
                EpisodeNoisyExecution.objects.filter(id=nes[i][0]).update(fitness = ranks[i])
    commit()

    # calc ranks
    ids_fitnesses_timespend = current_episode.noisyExecutions.all().values_list('id',"fitness","timespend").distinct()
    fitnesses = [x[1] for x in ids_fitnesses_timespend]
    timesspend  = [x[2] for x in ids_fitnesses_timespend]

    ranks = rank(fitnesses)
    for i in range(0,len(fitnesses)):
        EpisodeNoisyExecution.objects.filter(id=ids_fitnesses_timespend[i][0]).update(fitness_rank = ranks[i])

    # calc episode average/sum values
    current_episode.timespend   = sum(timesspend)
    current_episode.fitness_min   = min(fitnesses)
    current_episode.fitness_max   = max(fitnesses)
    current_episode.fitness_avg   = numpy.mean(fitnesses)
    current_episode.fitness_median =  numpy.median(fitnesses)
    current_episode.fitness_top10 =  numpy.mean(sorted(fitnesses,reverse=True)[0:10])
    current_episode.save()

    # check if episodes are finished
    episodes_finished = Episode.objects.filter(experiment_id = experiment_id).filter(status = "done").count()
    max_Episodes = ExperimentSet.objects.get(id=experimentSet_id).subsettings_Episodes_max
    if episodes_finished >= max_Episodes:
        experiment_done = Experiment.objects.filter(id = experiment_id).filter(~Q(status = "done")).update(status="done")
        if experiment_done == 1:
            on_commit(lambda: on_Experiment_done.delay(experiment_id, experimentSet_id))

        # clean hd space
        #current_episode.weightsNoise = numpy.array([])
        current_episode.optimiserData = pickle.dumps({})
        return

    # create next episode
    next_episode = Episode()
    next_episode.environment = current_episode.environment
    next_episode.architecture = current_episode.architecture
    next_episode.optimiser = current_episode.optimiser
    next_episode.experimentSet = current_episode.experimentSet
    next_episode.experiment = current_episode.experiment
    next_episode.version = current_episode.version + 1
    next_episode.public = current_episode.public

    # run optimiser
    optimiserInstance = next_episode.optimiser.getInstance()
    next_episode.weightsNoise, next_episode.optimiserMetaData, next_episode.optimiserData, count_factor, timespend_factor, steps_factor = optimiserInstance.optimise(current_episode)

    # clean hd space
    current_episode.weightsNoise = numpy.array([])
    current_episode.optimiserData = pickle.dumps({})

    eset = next_episode.experimentSet          
    
    #factors are -1 .. 1 
    h = ( eset.subsettings_EpisodeNoisyExecutions_max           - eset.subsettings_EpisodeNoisyExecutions_min           ) / 2.0
    next_episode.subsettings_EpisodeNoisyExecutions_max           = eset.subsettings_EpisodeNoisyExecutions_min            +  h + ( h  * count_factor      ) 

    h = ( eset.subsettings_EpisodeNoisyExecutions_max_steps     - eset.subsettings_EpisodeNoisyExecutions_min_steps     ) / 2.0
    next_episode.subsettings_EpisodeNoisyExecutions_max_steps     = eset.subsettings_EpisodeNoisyExecutions_min_steps      +  h + ( h * steps_factor      )

    h = ( eset.subsettings_EpisodeNoisyExecutions_max_timespend - eset.subsettings_EpisodeNoisyExecutions_min_timespend ) / 2.0
    next_episode.subsettings_EpisodeNoisyExecutions_max_timespend = eset.subsettings_EpisodeNoisyExecutions_min_timespend  +  h + ( h * timespend_factor  )
    next_episode.save()



# EpisodeNoisyExecution
@shared_task
def on_NoisyExecution_created(noisyExecution_id, episode_id, experiment_id, experimentSet_id):
    pass

@shared_task
def on_NoisyExecution_done(noisyExecution_id, episode_id, experiment_id, experimentSet_id):
    from .models import EpisodeNoisyExecution
    from .models import Episode
    
    episode = Episode.objects.get(id=episode_id)
    noisyExecutions_done = EpisodeNoisyExecution.objects.filter(episode_id = episode_id).filter(status = "done").count()
    if noisyExecutions_done < episode.subsettings_EpisodeNoisyExecutions_max:
        return

    episode_done = Episode.objects.filter(id = episode_id).filter(~Q(status = "done")).update(status="done")
    if episode_done == 1:
        on_Episode_done.delay(episode_id, experiment_id, experimentSet_id)
    


# CRON

@shared_task
def cron_clean_locked_hanging():
    from .models import EpisodeNoisyExecution
    
    tdiff = datetime.now() - timedelta(minutes=5)
    noisyExecution_hangs = EpisodeNoisyExecution.objects.filter(public = True, status = "locked", updated__lt = tdiff)
    for noisyExecution_hang in noisyExecution_hangs:
        noisyExecution_hang.locked = False
        noisyExecution_hang.client = ""
        noisyExecution_hang.lock = ""
        noisyExecution_hang.status = "idle"
        noisyExecution_hang.save()
    

'''
@shared_task
def on_OptimiserTraining_created(optimiserTraining_id):
    from .models import OptimiserTraining
    from .models import ExperimentSet
    from .models import Experiment

    optimiserTraining = OptimiserTraining.objects.get(id=optimiserTraining_id) 
    
    envs = []
    archs = []
    for environment_set in optimiserTraining.environments_set.all():
        for _ in range(0,environment_set.nr_of_instances):
            envs.append(environment_set.environment)
            for architecture_set in optimiserTraining.architectures_set.all():
                for _ in range(0,architecture_set.nr_of_instances):
                    archs.append(architecture_set.architecture)

    if len(envs) > optimiserTraining.max_parallel_envs_per_episode:
        envs = random.sample(envs, optimiserTraining.max_parallel_envs_per_episode)
    if len(archs) > optimiserTraining.max_parallel_archs_per_episode:
        archs = random.sample(archs, optimiserTraining.max_parallel_archs_per_episode)

    experimentSet = ExperimentSet()

    experimentSet.public = True
    experimentSet.name = ""
    experimentSet.description = ""
    experimentSet.status = "active"

    experimentSet.optimiserTraining = optimiserTraining 
    experimentSet.optimiserTraining_episodeNr = 1

    experimentSet.environments = envs 
    experimentSet.architectures = archs
    experimentSet.optimisers = optimiserTraining.optimiser

    experimentSet.max_Episodes = 2
    experimentSet.max_Experiments = optimiserTraining.episodeNoisyExecutions_count_max

    experimentSet.episodeNoisyExecutions_count_min = 5
    experimentSet.episodeNoisyExecutions_count_max = 500
    experimentSet.episodeNoisyExecution_timespend_min = 10
    experimentSet.episodeNoisyExecution_timespend_max = 120
    experimentSet.episodeNoisyExecution_steps_min = 10
    experimentSet.episodeNoisyExecution_steps_max = 10000
    experimentSet.save()


def on_OptimiserTraining_EpisodeDone(experimentSet_id):
    from .models import OptimiserTraining
    from .models import ExperimentSet
    from .models import Experiment

    last_experimentSet = ExperimentSet.objects.get(id=experimentSet_id) 
    optimiserTraining = last_experimentSet.optimiserTraining

    archs = [ x.architecture for x in last_experimentSet.architectures.all() ]    
    ac = optimiserTraining.max_parallel_archs_change_per_episode
    while len(archs) > 0 and ac > 0:
        del archs[random.randint(0,len(archs)-1)]
        ac -= 1

    envs  = [ x.environment  for x in last_experimentSet.environments.all()  ]
    ec = optimiserTraining.max_parallel_envs_change_per_episode
    while len(envs) > 0 and ec > 0:
        del envs[random.randint(0,len(envs)-1)]
        ec -= 1

    n_envs = []
    n_archs = []
    for environment_set in optimiserTraining.environments_set.all():
        for _ in range(0,environment_set.nr_of_instances):
            n_envs.append(environment_set.environment)
            for architecture_set in optimiserTraining.architectures_set.all():
                for _ in range(0,architecture_set.nr_of_instances):
                    n_archs.append(architecture_set.architecture)

    diff = optimiserTraining.max_parallel_envs_per_episode - len(envs)
    while diff > 0:
        envs.append(random.choice(n_envs))
        diff -= 1 

    diff = optimiserTraining.max_parallel_envs_per_episode - len(archs)
    while diff > 0:
        archs.append(random.choice(n_archs))
        diff -= 1 

    experimentSet = ExperimentSet()

    experimentSet.public = True
    experimentSet.name = ""
    experimentSet.description = ""
    experimentSet.status = "active"

    experimentSet.optimiserTraining = optimiserTraining 
    experimentSet.optimiserTraining_episodeNr = last_experimentSet.optimiserTraining_episodeNr + 1 

    experimentSet.environments = envs 
    experimentSet.architectures = archs
    experimentSet.optimisers = optimiserTraining.optimiser

    experimentSet.max_Episodes = 2
    experimentSet.max_Experiments = optimiserTraining.episodeNoisyExecutions_count_max

    experimentSet.episodeNoisyExecutions_count_min = 5
    experimentSet.episodeNoisyExecutions_count_max = 500
    experimentSet.episodeNoisyExecution_timespend_min = 10
    experimentSet.episodeNoisyExecution_timespend_max = 120
    experimentSet.episodeNoisyExecution_steps_min = 10
    experimentSet.episodeNoisyExecution_steps_max = 10000
    experimentSet.save()


@shared_task
def on_OptimiserTraining_done(experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment
    timespend = 0.0
    for ep in Experiment.objects.filter(experimentSet_id = experimentSet_id):
        timespend += ep.timespend
    experiment = ExperimentSet.objects.get(id=experimentSet_id)
    experiment.timespend = timespend
    experiment.save()



'''
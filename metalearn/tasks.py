
from celery import shared_task
from django.db.models import Q
import numpy
from scipy.stats import rankdata
from .ml.optimisers import all_optimisers


# ExperimentSet

@shared_task
def on_ExperimentSet_created(experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment

    experimentSet = ExperimentSet.objects.get(id=experimentSet_id)

    for environment_set in experimentSet.environments_set.all():
        for _ in range(0,environment_set.nr_of_instances):

            for architecture_set in experimentSet.architectures_set.all():
                for _ in range(0,architecture_set.nr_of_instances):

                    for optimiser_set in experimentSet.optimisers_set.all():
                        for _ in range(0,optimiser_set.nr_of_instances):

                            experiment = Experiment()
                            experiment.status = "active"
                            experiment.experimentSet = experimentSet
                            experiment.environment  = environment_set.environment
                            experiment.architecture = architecture_set.architecture
                            experiment.optimiser    = optimiser_set.optimiser
                            experiment.save()

@shared_task
def on_ExperimentSet_done(experimentSet_id):
    print("onExperimentSet_done(%s)" % experimentSet_id)
    timespend = 0.0
    for ep in Experiment.objects.filter(experimentSet_id = experimentSet_id):
        timespend += ep.timespend
    experiment = ExperimentSet.objects.get(id=experimentSet_id)
    experiment.timespend = timespend
    experiment.save()


# Experiment

@shared_task
def on_Experiment_created(experiment_id, experimentSet_id):
    from .models import Experiment
    from .models import Episode

    experiment = Experiment.objects.get(id=experiment_id)

    # create first episode

    episode = Episode()
    episode.environment = experiment.environment
    episode.architecture = experiment.architecture
    episode.optimiser = experiment.optimiser
    episode.experimentSet = experiment.experimentSet
    episode.experiment = experiment
    episode.version = 1

    # init optimiser
    optimiserInstance = all_optimisers[episode.optimiser.name ]["class"]()
    episode.weightsNoise, episode.optimiserData, count_factor, timespend_factor, steps_factor = optimiserInstance.initialize(episode.environment, episode.architecture)  
    eset = episode.experimentSet          
    episode.episodeNoisyExecution_steps     = eset.episodeNoisyExecution_steps_min     + ( ( eset.episodeNoisyExecution_steps_max     - eset.episodeNoisyExecution_steps_min     ) * steps_factor )
    episode.episodeNoisyExecution_timespend = eset.episodeNoisyExecution_timespend_min + ( ( eset.episodeNoisyExecution_timespend_max - eset.episodeNoisyExecution_timespend_min ) * timespend_factor )
    episode.episodeNoisyExecutions_count    = eset.episodeNoisyExecutions_count_min    + ( ( eset.episodeNoisyExecutions_count_max    - eset.episodeNoisyExecutions_count_min    ) * count_factor )



    episode.save()

@shared_task
def on_Experiment_done(experiment_id, experimentSet_id):
    from .models import ExperimentSet
    from .models import Experiment
    from .models import Episode

    experiments_to_go = Experiment.objects.filter(experimentSet_id = experimentSet_id).filter(~Q(status = "done")).count()
    if experiments_to_go > 0:
        return  
    
    timespends = Episode.objects.filter(experiment_id = experiment_id).values_list("timespend")
    
    experiment = Experiment.objects.get(id=experimentSet_id)
    experiment.timespend = sum(timespends)
    experiment.save()

    experimentSet_done = ExperimentSet.objects.filter(id = experimentSet_id).filter(~Q(status = "done")).update(status="done")
    if experimentSet_done == 1:
        on_ExperimentSet_done.delay(experimentSet_id)



# Episode

@shared_task
def on_Episode_created(episode_id, experiment_id, experimentSet_id):
    from .models import Episode
    from .models import EpisodeNoisyExecution

    episode = Episode.objects.get(id=episode_id)
    count = episode.noisyExecutions.count()

    for number in range(count,episode.episodeNoisyExecutions_count):
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


    # calc fitness ranks for episode.NoisyExecution
    ids_fitnesses_timespend = current_episode.noisyExecutions.all().values_list('id',"fitness","timespend").distinct()
    fitnesses = [x[1] for x in ids_fitnesses_timespend]
    timesspend  = [x[2] for x in ids_fitnesses_timespend]

    l = ( len(set(fitnesses)) - 1 )
    if l == 0:
        l = 1
    ranks = ( rankdata(fitnesses, method="dense") - 1 ) / l
    for i in range(0,len(fitnesses)):
        EpisodeNoisyExecution.objects.filter(id=ids_fitnesses_timespend[i][0]).update(fitness_rank = ranks[i])

    current_episode.timespend   = sum(timesspend)
    current_episode.fitness_min   = min(fitnesses)
    current_episode.fitness_max   = max(fitnesses)
    current_episode.fitness_avg   = numpy.mean(fitnesses)
    current_episode.fitness_median =  numpy.median(fitnesses)


    episodes_finished = Episode.objects.filter(experiment_id = experiment_id).filter(status = "done").count()
    max_Episodes = ExperimentSet.objects.get(id=experimentSet_id).max_Episodes

    if episodes_finished >= max_Episodes:
        experiment_done = Experiment.objects.filter(id = experiment_id).filter(~Q(status = "done")).update(status="done")
        if experiment_done == 1:
            on_Experiment_done.delay(experiment_id, experimentSet_id)
        return

    # create next episode

    next_episode = Episode()

    next_episode.environment = current_episode.environment
    next_episode.architecture = current_episode.architecture
    next_episode.optimiser = current_episode.optimiser
    next_episode.experimentSet = current_episode.experimentSet
    next_episode.experiment = current_episode.experiment
    next_episode.version = current_episode.version + 1

    # run optimiser
    optimiserInstance = all_optimisers[current_episode.optimiser.name ]["class"]()
    next_episode.weightsNoise, next_episode.optimiserData, count_factor, timespend_factor, steps_factor = optimiserInstance.optimise(current_episode)

    eset = next_episode.experimentSet          
    next_episode.episodeNoisyExecution_steps     = eset.episodeNoisyExecution_steps_min     + ( ( eset.episodeNoisyExecution_steps_max     - eset.episodeNoisyExecution_steps_min     ) * steps_factor )
    next_episode.episodeNoisyExecution_timespend = eset.episodeNoisyExecution_timespend_min + ( ( eset.episodeNoisyExecution_timespend_max - eset.episodeNoisyExecution_timespend_min ) * timespend_factor )
    next_episode.episodeNoisyExecutions_count    = eset.episodeNoisyExecutions_count_min    + ( ( eset.episodeNoisyExecutions_count_max    - eset.episodeNoisyExecutions_count_min    ) * count_factor )

    next_episode.save()



# EpisodeNoisyExecution
@shared_task
def on_NoisyExecution_created(noisyExecution_id, episode_id, experiment_id, experimentSet_id):
    pass

@shared_task
def on_NoisyExecution_done(noisyExecution_id, episode_id, experiment_id, experimentSet_id):
    from .models import EpisodeNoisyExecution
    from .models import Episode

    noisyExecutions_to_go = EpisodeNoisyExecution.objects.filter(episode_id = episode_id).filter(~Q(status = "done")).count()
    if noisyExecutions_to_go > 0:
        return

    episode_done = Episode.objects.filter(id = episode_id).filter(~Q(status = "done")).update(status="done")
    if episode_done == 1:
        on_Episode_done.delay(episode_id, experiment_id, experimentSet_id)
    

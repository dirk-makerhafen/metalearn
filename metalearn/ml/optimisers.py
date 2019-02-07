import numpy as np
import json
import redis
import pickle
import time
from gym import spaces
from django.db.transaction import commit

try:
    redisconnection = redis.StrictRedis(unix_socket_path='/var/run/redis/redis.sock', db=8)
    redisconnection.get("__test")
except:
    redisconnection = redis.StrictRedis(db=8)


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks

def compute_centered_ranks(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y

def compute_weight_decay(weight_decay, list_of_ind_weights):
  return - weight_decay * np.mean(numpy.square(list_of_ind_weights), axis=1)

def createNoise(seed, width):
    r = np.random.RandomState(seed)
    return r.randn(width).astype(np.float32)

def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)

def batched_weighted_sum(weights, vecs, batch_size=500):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed

class AdamOptimizer(object):
    def __init__(self,num_params, learning_rate, beta1=0.99, beta2=0.999, epsilon=1e-08):
        self.dim = num_params

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def update(self, weights, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / (np.linalg.norm(weights) + self.epsilon)
        new_weights = weights + step
        return new_weights, ratio

    def _compute_step(self, globalg):
        if self.m is None:
            self.m = np.zeros(self.dim, dtype=np.float32)
        if self.v is None:
            self.v = np.zeros(self.dim, dtype=np.float32)

        a = self.learning_rate * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = -a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step

    def toDict(self):
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "dim": self.dim,
            "m": self.m,
            "v": self.v,
            "t": self.t,
        }

    def fromDict(self, dataDict):
        self.learning_rate = dataDict["learning_rate"]
        self.beta1 = dataDict["beta1"]
        self.beta2 = dataDict["beta2"]
        self.epsilon = dataDict["epsilon"]
        self.dim = dataDict["dim"]
        self.t = dataDict["t"]
        self.m = dataDict["m"]
        self.v = dataDict["v"]



class BaseOptimiser():
    def __init__(self):
        self.parameters = {
            "num_params" : -1,              # number of model parameters
        }

    def initialize(self, environment, architecture):
        self.parameters["num_params"] = self.getNrOfTrainableParameters(environment, architecture)

        weightsNoise = np.array([
            np.zeros(self.parameters["num_params"], dtype=np.float32),  # parameter 0 -> Weights
            [ 1, ],            # parameter 1 -> Noiselevels
        ])

        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        optimiserData =  pickle.dumps({},2)

        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1
    
        return  [ weightsNoise, optimiserMetaData , optimiserData, count_factor, timespend_factor, steps_factor]

    def optimise(self, episode):
        weightsNoise  = np.array( [ [], [] ] ,dtype=np.float32)
        optimiserMetaData = ""
        optimiserData = ""
        steps_factor = 1        
        count_factor = 1
        timespend_factor = 1
        return  [ weightsNoise, optimiserData, optimiserMetaData, count_factor, timespend_factor, steps_factor ]
    
    # in case of meta optimisers, overwrite this function
    def reward(self, episode, fitness):
        pass

    def getNrOfTrainableParameters(self, environment, architecture):
        num_params = 0
        cache_key = "%s_%s.num_params" % (environment.name, architecture.name)
        c = redisconnection.get(cache_key)
        if c != None:
            num_params = int(c)
        else:
            env = environment.getInstance()
            env.initialize()

            arch = architecture.getInstance()
            arch.initialize(env.observation_space, env.action_space)
        
            num_params = arch.num_params
            env.close()
            arch.close()

        if num_params < 1:
            raise Exception("Failed to get number of trainable parameters for arch '%s'  env '%s'" % (architecture.name, environment.name))

        redisconnection.set(cache_key, num_params)
        redisconnection.expire(cache_key,30)
        print("getNrOfTrainableParameters for %s  -  %s  : %s" % (environment, architecture, num_params))
        return num_params
    def close(self):
        pass

# https://github.com/hardmaru/estool/blob/master/es.py
class OptimiserOpenES(BaseOptimiser):
    def __init__(self):
        self.parameters = {
            "num_params" : -1,              # number of model parameters
            "sigma" : 0.1,                    # current standard deviation
            "sigma_ini" : 0.1,                # initial standard deviation
            "sigma_decay" : 0.999,            # anneal standard deviation
            "sigma_limit" : 0.01,             # stop annealing if less than this
            "learning_rate" : 0.01,           # learning rate for standard deviation
            "learning_rate_decay" : 0.9999, # annealing the learning rate
            "learning_rate_limit" : 0.001,  # stop annealing learning rate
            "weight_decay" : 0.01,            # weight decay coefficient
            "rank_fitness" : True,            # use rank rather than fitness numbers
            "subOptimizerData" : None,
        }

    def initialize(self, environment, architecture):
        self.parameters["num_params"] = self.getNrOfTrainableParameters(environment, architecture)

        weightsNoise = np.array([
            np.zeros(self.parameters["num_params"], dtype=np.float32),  # parameter 0 -> Weights
            [ self.parameters["sigma"], ],            # parameter 1 -> Noiselevels
        ])

        subOptimizer = AdamOptimizer( self.parameters["num_params"], self.parameters["learning_rate"])
        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        optimiserData =  pickle.dumps({
            "subOptimizerData" : subOptimizer.toDict(),
        },2)

        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1
    
        return  [ weightsNoise, optimiserMetaData, optimiserData, count_factor, timespend_factor, steps_factor]

    def optimise(self, episode):
        optimiserMetaData = pickle.loads(episode.optimiserMetaData)
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserMetaData["parameters"]

        # collect rewards from episoed noisyExecutions
        noisyExecutions = list(episode.noisyExecutions.all())
        reward = np.array([n.fitness for n in noisyExecutions], dtype=np.float32)

        if self.parameters["rank_fitness"]:
            reward = compute_centered_ranks(reward)

        weightsNoise = episode.weightsNoise
        weights = weightsNoise[0]
        noiselevels = weightsNoise[1]

        #if self.parameters["weight_decay"] > 0: 
        #    used_weights = weights + noisyExecutions_noise * noiselevels
        #    l2_decay = compute_weight_decay( self.parameters["weight_decay"], used_weights)
        #    used_weights = None
        #    reward += l2_decay

        # main bit:

        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)

        # reward * noise
        g, _ = batched_weighted_sum(
            normalized_reward,
            (createNoise(n.noiseseed, self.parameters["num_params"]) for n in noisyExecutions)
        )
        g /= len(noisyExecutions)
        g /= self.parameters["sigma"] # ?? this is in the online example, but may be wrong. 

        # load saved sub optimiser
        subOptimizer = AdamOptimizer(self.parameters["num_params"], self.parameters["learning_rate"])
        subOptimizer.fromDict(optimiserData["subOptimizerData"])
        subOptimizer.stepsize = self.parameters["learning_rate"]

        # update weights
        weights, update_ratio = subOptimizer.update(weights, -g )

        # main bit done.

        # adjust sigma according to the adaptive sigma calculation
        if (self.parameters["sigma"] > self.parameters["sigma_limit"]):
            self.parameters["sigma"] *= self.parameters["sigma_decay"]

        if (self.parameters["learning_rate"] > self.parameters["learning_rate_limit"]):
            self.parameters["learning_rate"] *= self.parameters["learning_rate_decay"]

        weightsNoise = np.array([
            weights,                        # parameter 0 -> Weights
            [ self.parameters["sigma"], ],  # parameter 1 -> Noiselevels
        ])

        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        optimiserData =  pickle.dumps({
            "subOptimizerData" : subOptimizer.toDict(),
        },2)    


        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserMetaData, optimiserData, count_factor, timespend_factor, steps_factor]


class OptimiserOpenES_Bugfixed(OptimiserOpenES):
    def __init__(self, *args, **kwargs):
        super(OptimiserOpenES_Bugfixed, self).__init__(*args, **kwargs)

    def optimise(self, episode):
        optimiserMetaData = pickle.loads(episode.optimiserMetaData)
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserMetaData["parameters"]

        # collect rewards from episoed noisyExecutions
        noisyExecutions = list(episode.noisyExecutions.all())
        reward = np.array([n.fitness for n in noisyExecutions], dtype=np.float32)

        if self.parameters["rank_fitness"]:
            reward = compute_centered_ranks(reward)

        weightsNoise = episode.weightsNoise
        weights = weightsNoise[0]
        noiselevels = weightsNoise[1]

        # main bit:

        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)

        # reward * noise
        g, _ = batched_weighted_sum(
            normalized_reward,
            (createNoise(n.noiseseed, self.parameters["num_params"]) for n in noisyExecutions)
        )
        g /= len(noisyExecutions)
        g *= self.parameters["sigma"] # BUG FIXED?

        # load saved sub optimiser
        subOptimizer = AdamOptimizer(self.parameters["num_params"], self.parameters["learning_rate"])
        subOptimizer.fromDict(optimiserData["subOptimizerData"])
        subOptimizer.stepsize = self.parameters["learning_rate"]

        # update weights
        weights, update_ratio = subOptimizer.update(weights, -g )

        # main bit done.

        # adjust sigma according to the adaptive sigma calculation
        if (self.parameters["sigma"] > self.parameters["sigma_limit"]):
            self.parameters["sigma"] *= self.parameters["sigma_decay"]

        if (self.parameters["learning_rate"] > self.parameters["learning_rate_limit"]):
            self.parameters["learning_rate"] *= self.parameters["learning_rate_decay"]

        weightsNoise = np.array([
            weights,                        # parameter 0 -> Weights
            [ self.parameters["sigma"], ],  # parameter 1 -> Noiselevels
        ])
        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        optimiserData =  pickle.dumps({
            "subOptimizerData" : subOptimizer.toDict(),
        },2)      


        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserMetaData, optimiserData, count_factor, timespend_factor, steps_factor]


class OptimiserESUeber(OptimiserOpenES):

    def __init__(self, *args, **kwargs):
        super(OptimiserESUeber, self).__init__(*args, **kwargs)

    def optimise(self, episode):
        optimiserMetaData = pickle.loads(episode.optimiserMetaData)
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserMetaData["parameters"]

        # collect rewards from episoed noisyExecutions
        noisyExecutions = list(episode.noisyExecutions.all())
        reward = np.array([n.fitness for n in noisyExecutions], dtype=np.float32)

        if self.parameters["rank_fitness"]:
            reward = compute_centered_ranks(reward)

        weightsNoise = episode.weightsNoise
        weights = weightsNoise[0]
        noiselevels = weightsNoise[1]

        # main bit:

        # reward * noise
        g, _ = batched_weighted_sum(
            reward,
            (createNoise(n.noiseseed, self.parameters["num_params"]) for n in noisyExecutions)
        )
        g /= len(noisyExecutions)

        # load saved sub optimiser
        subOptimizer = AdamOptimizer(self.parameters["num_params"], self.parameters["learning_rate"])
        subOptimizer.fromDict(optimiserData["subOptimizerData"])
        subOptimizer.stepsize = self.parameters["learning_rate"]

        # update weights
        l2coeff = 0.005
        weights, update_ratio = subOptimizer.update(weights, -g + l2coeff * weights) # openai/ueber

        # main bit done.

        # adjust sigma according to the adaptive sigma calculation
        if (self.parameters["sigma"] > self.parameters["sigma_limit"]):
            self.parameters["sigma"] *= self.parameters["sigma_decay"]

        if (self.parameters["learning_rate"] > self.parameters["learning_rate_limit"]):
            self.parameters["learning_rate"] *= self.parameters["learning_rate_decay"]

        weightsNoise = np.array([
            weights,                        # parameter 0 -> Weights
            [ self.parameters["sigma"], ],  # parameter 1 -> Noiselevels
        ])
        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        optimiserData =  pickle.dumps({
            "subOptimizerData" : subOptimizer.toDict(),
        },2)  
    
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserMetaData, optimiserData, count_factor, timespend_factor, steps_factor]


class OptimiserMetaES(BaseOptimiser):
    def __init__(self, nr_of_embeddings_per_weight, nr_of_embeddings):
        from ..models import Environment
        from ..models import Architecture
        from ..models import Optimiser
        from ..models import ExperimentSet
        from ..models import ExperimentSetToEnvironment
        from ..models import ExperimentSetToArchitecture
        from ..models import ExperimentSetToOptimiser

        print("OptimiserMetaES init")
        self.nr_of_embeddings_per_weight = nr_of_embeddings_per_weight
        self.nr_of_embeddings = nr_of_embeddings
        self.experimentSet_name = "OptimiserMetaES Emb:%s/%s" % (self.nr_of_embeddings_per_weight, self.nr_of_embeddings)  # this optimiser creates an experimentSet for its own evolution

        param = [["nr_of_embeddings_per_weight",self.nr_of_embeddings_per_weight], ["nr_of_embeddings",self.nr_of_embeddings]]

        try:
            self.optimiserenvironment = Environment.objects.get(classname="OptimiserMetaESEnvironment", classargs=json.dumps(param))
        except:
            print("Must generate Environment Model instance")
            self.optimiserenvironment = Environment()
            self.optimiserenvironment.name = "OptimiserMetaESEnvironment"
            self.optimiserenvironment.description = "Autogenerated by OptimiserMetaES"
            self.optimiserenvironment.classname = "OptimiserMetaESEnvironment"
            self.optimiserenvironment.classargs = json.dumps(param)
            self.optimiserenvironment.save()

        try:
            self.optimiserarchitecture = Architecture.objects.get(classname="Architecture_MetaES", classargs=json.dumps(param))
        except:
            print("Must generate Architecture Model instance")
            self.optimiserarchitecture = Architecture()
            self.optimiserarchitecture.name = "Architecture_MetaES"
            self.optimiserarchitecture.description = "Autogenerated by OptimiserMetaES"
            self.optimiserarchitecture.classname = "Architecture_MetaES"
            self.optimiserarchitecture.classargs = json.dumps(param)
            self.optimiserarchitecture.save()

        try:
            self.optimiseroptimiser = Optimiser.objects.get(classname="OptimiserOpenES_Bugfixed", classargs=json.dumps([]))
        except:
            print("Must generate Optimiser Model instance")
            self.optimiseroptimiser = Architecture()
            self.optimiseroptimiser.name = "OptimiserOpenES_Bugfixed"
            self.optimiseroptimiser.description = "Autogenerated by OptimiserMetaES"
            self.optimiseroptimiser.classname = "OptimiserOpenES_Bugfixed"
            self.optimiseroptimiser.classargs = json.dumps([])
            self.optimiseroptimiser.save()

        try:
            self.experimentSet = ExperimentSet.objects.get(name=self.experimentSet_name)
        except:
            self.experimentSet = ExperimentSet()
            self.experimentSet.public = False
            self.experimentSet.name = self.experimentSet_name
            self.experimentSet.description = "Autogenerated by OptimiserMetaES"
            self.experimentSet.subsettings_Experiments_max = 1 
            self.experimentSet.subsettings_Episodes_max = 1000
            self.experimentSet.subsettings_EpisodeNoisyExecutions_min = 10 
            self.experimentSet.subsettings_EpisodeNoisyExecutions_max = 100
            self.experimentSet.subsettings_EpisodeNoisyExecutions_min_steps = 2   # aka nr of episodes for sub experiments
            self.experimentSet.subsettings_EpisodeNoisyExecutions_max_steps = 300
            self.experimentSet.subsettings_EpisodeNoisyExecutions_min_timespend = 1 # ignored for opti meta training
            self.experimentSet.subsettings_EpisodeNoisyExecutions_max_timespend = 2

            self.experimentSet.save()

            e1 = ExperimentSetToEnvironment(experimentSet=self.experimentSet, environment = self.optimiserenvironment, nr_of_instances = 1)
            e2 = ExperimentSetToArchitecture(experimentSet=self.experimentSet, architecture = self.optimiserarchitecture, nr_of_instances = 1)
            e3 = ExperimentSetToOptimiser(experimentSet=self.experimentSet, optimiser = self.optimiseroptimiser, nr_of_instances = 1)
            e1.save()
            e2.save()
            e3.save()
            
    

        self.parameters = {
            "num_params" : -1,              # number of model parameters
            "nr_of_embeddings_per_weight" : self.nr_of_embeddings_per_weight,  
            "nr_of_embeddings" : self.nr_of_embeddings,  
            "experimentSet_name" : self.experimentSet_name, 
        }


    def initialize(self, environment, architecture):
        print("OptimiserMetaES initialize")
        from ..models import ExperimentSet
        from ..models import EpisodeNoisyExecution
        start_t = time.time()

        self.parameters["num_params"] = self.getNrOfTrainableParameters(environment, architecture)

        opti_noisyExecution = None
        for i in range(0,10):
            experimentSet_id = ExperimentSet.objects.get(name=self.experimentSet_name).id
            opti_noisyExecution, lock = EpisodeNoisyExecution.getOneIdleLocked("OptimiserMetaES", public=False, experimentSetIds = [experimentSet_id ,])
            if opti_noisyExecution != None:
                break            
            print("delay")
            time.sleep(2)
            commit()

        if opti_noisyExecution == None:
            return "delay"


        # load optimiser 
        opti_weightNoise = opti_noisyExecution.episode.weightsNoise
        opti_weights = opti_weightNoise[0] + (opti_weightNoise[1] * createNoise(opti_noisyExecution.noiseseed, len(opti_weightNoise[0] ) ) )
        optimiser_Arch = opti_noisyExecution.architecture.getInstance()
        input_space, output_space = self._getInputOutputSpaces()
        optimiser_Arch.initialize(input_space, output_space, opti_weights)
        print(optimiser_Arch)

        data = np.array([
            np.zeros( [ self.parameters["num_params"], self.parameters["nr_of_embeddings_per_weight"] ]).astype(np.float32),    # per_weight_embeddings
            np.random.randn( self.parameters["num_params"]).astype(np.float32),        # used weights            
            np.zeros( [ self.parameters["nr_of_embeddings"]]).astype(np.float32),       # embeddings          
            np.array([0.0]), # episode nr 
            np.array([0.0]), # fitness
            np.array([0.0]), # rank
            np.array([0.0]), # steps
            np.array([0.0]), # nr_of_noisy execution per expisode
        ])
        r = optimiser_Arch.run(data)
        optimiser_Arch.close()
        print(r)
        new_embeddings_per_weight = r[0]
        new_weights = r[1]
        new_noise = r[2]
        new_embeddings = r[3]
        new_count_factor = r[4]
        new_timespend_factor = r[5]
        new_steps_factor = r[6]

        print("new_weights")
        print(new_weights)
        print("new_noise")
        print(new_noise)
        print("new_embeddings")
        print(new_embeddings)
        print("new_count_factor")
        print(new_count_factor)
        print("new_timespend_factor")
        print(new_timespend_factor)
        print("new_steps_factor")
        print(new_steps_factor)

        weightsNoise = np.array([
            new_weights, # parameter 0 -> Weights
            new_noise,   # parameter 1 -> Noiselevels
        ])

        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
            "noisyExecution_id" : opti_noisyExecution.id,
            "steps" : 1,
            "timespend" : time.time() - start_t,
        },2)

        print("optimiserMetaData")
        print(optimiserMetaData)

        optimiserData =  pickle.dumps({
            "embeddings_per_weight" : new_embeddings_per_weight,
            "embeddings" : new_embeddings,
        },2)  

        return  [ weightsNoise, optimiserMetaData, optimiserData, new_count_factor, new_timespend_factor, new_steps_factor]

    def optimise(self, episode):
        from ..models import ExperimentSet
        from ..models import EpisodeNoisyExecution
        start_t = time.time()
        print("OptimiserMetaES.optimise")
        print(episode)

        optimiserMetaData = pickle.loads(episode.optimiserMetaData)
        optimiserData = pickle.loads(episode.optimiserData)

        self.parameters = optimiserMetaData["parameters"]


        # load optimiser 
        optimiser_noisyExecution = EpisodeNoisyExecution.objects.get(id = optimiserMetaData["noisyExecution_id"])
        optimiser_Arch = optimiser_noisyExecution.architecture.getInstance()
        opti_weightNoise = optimiser_noisyExecution.episode.weightsNoise
        print("optimiser opti_weightNoise")
        print(optimiser_noisyExecution)
        print(opti_weightNoise)
        opti_weights = opti_weightNoise[0] + (opti_weightNoise[1] * createNoise(optimiser_noisyExecution.noiseseed, len(opti_weightNoise[0] ) ) )
        input_space, output_space = self._getInputOutputSpaces()
        optimiser_Arch.initialize(input_space, output_space, opti_weights )

        
        print(optimiser_Arch)

        new_embeddings_per_weight = optimiserData["embeddings_per_weight"]
        new_weights = None
        new_noise = None
        new_embeddings = optimiserData["embeddings"]
        new_count_factor = 1
        new_timespend_factor = 1
        new_steps_factor = 1

        weightNoise = episode.weightsNoise
        
        noisyExecutions = [x for x in episode.noisyExecutions.all()]
        for noisyExecution in noisyExecutions:
            print("running meta optimiser")
            print(weightNoise)
            print(len(weightNoise[0]))
            print(len(weightNoise[1]))
            print(len(weightNoise))
            n = createNoise(noisyExecution.noiseseed, len(weightNoise[0] ))
            print(len(n))
            weights_used = weightNoise[0] + (weightNoise[1] * n )

            data = np.array([
                new_embeddings_per_weight,      # last per_weight_embeddings
                weights_used,        # used weights            
                new_embeddings,        # last embedding            
                np.array([episode.version]), # 
                np.array([noisyExecution.fitness]), # fitness
                np.array([noisyExecution.fitness_rank]), # rank
                np.array([ noisyExecution.steps ]), # steps of this noisy episode 
                np.array([len(noisyExecutions) ]), # nr_of_noisy execution per expisode
            ])

            print(data)
            r = optimiser_Arch.run(data)  # Run Optimiser NN
            
            print("r[0]")
            print(r[0])
            print("r[1]")
            print(r[1])
            print("r[2]")
            print(r[2])
            print("r[3]")
            print(r[3])
            
            new_embeddings_per_weight = r[0]
            new_weights = r[1]
            new_noise = r[2]
            new_embeddings = r[3]
            new_count_factor = r[4]
            new_timespend_factor = r[5]
            new_steps_factor = r[6]


        optimiser_Arch.close()

        weightsNoise = np.array([
            new_weights, # parameter 0 -> Weights
            new_noise,   # parameter 1 -> Noiselevels
        ])
        optimiserMetaData =  pickle.dumps({
            "parameters": self.parameters,
            "noisyExecution_id" : optimiser_noisyExecution.id,
            "steps" : optimiserMetaData["steps"] + 1,
            "timespend" : optimiserMetaData["timespend"] + (time.time() - start_t),
        },2)
        optimiserData =  pickle.dumps({
            "embeddings_per_weight" : new_embeddings_per_weight,
            "embeddings" : new_embeddings,
        },2)  
    
        return  [ weightsNoise, optimiserMetaData, optimiserData, new_count_factor, new_timespend_factor, new_steps_factor]

    def reward(self, episode, fitness):
        print("reward")
        from ..models import EpisodeNoisyExecution

        optimiserMetaData = pickle.loads(episode.optimiserMetaData)

        noisyExecution = EpisodeNoisyExecution.objects.get(id = optimiserMetaData["noisyExecution_id"])
        data = {
            "fitness": 0 ,  # actual reward is calculated on_Episode_done of this Optimisers Experiment
            "timespend": optimiserMetaData["timespend"], 
            "steps": optimiserMetaData["steps"], 
            "fitness_calc_key"   : "%s_%s_%s" % (episode.architecture.id, episode.environment.id, episode.version),
            "fitness_calc_value" : fitness,
        }
        noisyExecution.setResult(data)


    def _getInputOutputSpaces(self):

        input_space = spaces.Tuple((  
            spaces.Box(low=0,    high=100, shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings_per_weight"] ]),   # Last per_weight_embeddings
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"] ]),  # Used Weights#
            spaces.Box(low=-180, high=180, shape=[ self.parameters["nr_of_embeddings"] ]),  # embedding
            spaces.Box(low=-180, high=180, shape=[ 1 ]), # episode nr   
            spaces.Box(low=-180, high=180, shape=[ 1 ]), # fitness
            spaces.Box(low=0,    high=1,   shape=[ 1 ]), # rank
            spaces.Box(low=-180, high=180, shape=[ 1 ]), # steps
            spaces.Box(low=0,    high=1,   shape=[ 1 ]), # nr_of_noisy execution per expisode
        ))

        output_space = spaces.Tuple((
            spaces.Box(low=0,    high=100, shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings_per_weight"] ]), # new per_weight_embeddings
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"]  ]),  # new Weights
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"]  ]),  # new noise
            spaces.Box(low=0, high=1, shape=[ self.parameters["nr_of_embeddings"] ]),  # embedding
            spaces.Box(low=0, high=1, shape=[ 1 ]),  # count_factor
            spaces.Box(low=0, high=1, shape=[ 1 ]),  # timespend_factor
            spaces.Box(low=0, high=1, shape=[ 1 ]),  # steps_factor
        ))   
        return input_space, output_space



# create these if you db is empty
default_models = [
        {
            "name":"OptimiserOpenES",
            "description": "",
            "classname":"OptimiserOpenES",
            "classargs":[],
        },{
            "name":"OptimiserOpenES_Bugfixed",
            "description":"",
            "classname":"OptimiserOpenES_Bugfixed",
            "classargs":[],
        },{
            "name":"OptimiserESUeber",
            "description":"",
            "classname":"OptimiserESUeber",
            "classargs":[],
        },{
            "name":"OptimiserMetaES",
            "description":"",
            "classname":"OptimiserMetaES",
            "classargs":[["nr_of_embeddings_per_weight",5] , [ "nr_of_embeddings", 20 ] ],
        },
    ]
    


import numpy as np
import json
import redis
import pickle

from .environments import all_environments
from .architectures import all_architectures

redisconnection = redis.StrictRedis(unix_socket_path='/var/run/redis/redis.sock', db=8)



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

# batched to save ram, weights by weights.
def batched_weighted_sum(weights, vecs, batch_size=100):
    total = 0.
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size), itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float32), np.asarray(batch_vecs, dtype=np.float32))
        num_items_summed += len(batch_weights)
    return total, num_items_summed


class MetaOpti():
    def run(input):
        pass
'''
input 
    fitness
    fitness_rank
    steps
    steps_rank
    used_weights
    last_embedding
   
output
    weights 
    noiselevels
    factor_count
    factor_steps
    factor_time
    embedding

'''
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

        optimiserData =  pickle.dumps({
            "parameters": self.parameters,
        },2)
        
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1
    
        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]

    def optimise(self, episode):
        weightsNoise  = np.array( [ [], [] ] ,dtype=np.float32)
        optimiserData = ""
        steps_factor = 1        
        count_factor = 1
        timespend_factor = 1
        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor ]

    def getNrOfTrainableParameters(self, environment, architecture):
        num_params = 0
        cache_key = "%s_%s.num_params" % (environment.name, architecture.name)
        c = redisconnection.get(cache_key)
        if c != None:
            num_params = int(c)
        else:
            env = all_environments[environment.name]["class"]()
            env.initialize()

            arch = all_architectures[architecture.name]["class"]()
            arch.initialize(env.observation_space, env.action_space)
        
            num_params = arch.num_params
            env.close()
            arch.close()

        if num_params < 1:
            raise Exception("Failed to get number of trainable parameters for arch '%s'  env '%s'" % (architecture.name, environment.name))

        redisconnection.set(cache_key, num_params)
        redisconnection.expire(cache_key,30)
        return num_params


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
        optimiserData =  pickle.dumps({
            "parameters": self.parameters,
            "subOptimizerData" : subOptimizer.toDict(),
        },2)
        
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1
    
        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]

    def optimise(self, episode):
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserData["parameters"]

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


        # l2coeff = 0.001
        #weights, update_ratio = subOptimizer.update(weights, -g + l2coeff * weights) # openai/ueber

        # adjust sigma according to the adaptive sigma calculation
        if (self.parameters["sigma"] > self.parameters["sigma_limit"]):
            self.parameters["sigma"] *= self.parameters["sigma_decay"]

        if (self.parameters["learning_rate"] > self.parameters["learning_rate_limit"]):
            self.parameters["learning_rate"] *= self.parameters["learning_rate_decay"]

        weightsNoise = np.array([
            weights,                        # parameter 0 -> Weights
            [ self.parameters["sigma"], ],  # parameter 1 -> Noiselevels
        ])
        optimiserData = pickle.dumps({
            "parameters": self.parameters,
            "subOptimizerData" : subOptimizer.toDict(),
        },2)
    
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]


class OptimiserOpenES_Bugfixed(OptimiserOpenES):
    def __init__(self, *args, **kwargs):
        super(OptimiserOpenES_Bugfixed, self).__init__(*args, **kwargs)

    def optimise(self, episode):
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserData["parameters"]

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
        optimiserData = pickle.dumps({
            "parameters": self.parameters,
            "subOptimizerData" : subOptimizer.toDict(),
        },2)
    
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]


class OptimiserESUeber(OptimiserOpenES):

    def __init__(self, *args, **kwargs):
        super(OptimiserESUeber, self).__init__(*args, **kwargs)

    def optimise(self, episode):
        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserData["parameters"]

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
        optimiserData = pickle.dumps({
            "parameters": self.parameters,
            "subOptimizerData" : subOptimizer.toDict(),
        },2)
    
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]


'''
class OptimiserMetaES(BaseOptimiser):
    def __init__(self, nr_of_embeddings):
        
        self.parameters = {
            "num_params" : -1,              # number of model parameters
            "nr_of_embeddings" : nr_of_embeddings,  
        }
        # create ExperimentSet "OptimiserMetaES $nr_of_embeddings" if not exists

    def _getInputOutputSpaces(self ):
        episodeparameters = spaces.Tuple((   # metaparameters
                spaces.Box(low=-180, high=180, shape=1), # fitness
                spaces.Box(low=-180, high=180, shape=1), # rank
                spaces.Box(low=-180, high=180, shape=1), # steps
                spaces.Box(low=-180, high=180, shape=1), # foo
            ))

        input_space = spaces.Tuple((
            spaces.Box(low=0,    high=100, shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings"] ]),   # Last embedding
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"], 1 ]),  # Used Weights#
            episodeparameters,
        ))

        output_space = spaces.Tuple((
            spaces.Box(low=0,    high=100, shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings"] ]),   #new embedding
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"], 1 ]),  # new Weights
            spaces.Box(low=-180, high=180, shape=[ self.parameters["num_params"], 1 ]),  # new noise
        ))   
        return input_space, output_space

    def initialize(self, environment, architecture):
        self.parameters["num_params"] = self.getNrOfTrainableParameters(environment, architecture)

        input_space, output_space = self.getInputOutputSpaces()


        # noiseExecution = Get ExperimentSet Optimiser NoiseExecution
        optiArch = getInstance( noiseExecution.architecture )

        weightNoise = noiseExecution.episode.wn
        optiArch.initialize(input_space, output_space, weights)

        data = (
            np.random.randn(shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings"] ]),      # last embedding
            np.random.randn(shape=[ self.parameters["num_params"], 1 ]),        # used weights
            (
                0, # fitness
                0, # rank
                0, # steps
                0, # foo
            )
        )

        r = optiArch.run(data)
       
        new_weights = r[0]
        new_noise = r[1]
        new_embeddings = r[2]

        optimiserData =  pickle.dumps({
            "parameters": self.parameters,
            "embeddings" : embeddings,
            "NoiseExecution" : noiseExecution.id,
        },2)
        
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1
    
        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]

    def optimise(self, episode):
        input_space, output_space = self.getInputOutputSpaces()

        # noiseExecution = Get ExperimentSet Optimiser NoiseExecution via optimiserData["NoiseExecution"]
        optiArch = getInstance( noiseExecution.architecture )
        weight = []
        optiArch.initialize(input_space, output_space, weight)
        data = (
            np.random.randn(shape=[ self.parameters["num_params"], self.parameters["nr_of_embeddings"] ]),      # last embedding
            np.random.randn(shape=[ self.parameters["num_params"], 1 ]),        # used weights
            (
                0, # fitness
                0, # rank
                0, # steps
                0, # foo
            )
        )

        r = optiArch.run(data)
       
        new_weights = r[0]
        new_noise = r[1]
        new_embeddings = r[2]


        emb = optimiserData["embeddings"]
        for ne in episode.noisyExecutions.all():
            data = (
                emb,
                ne.used_weights
            )
            new_data = ne_arch.run(data)
            new_weights = new_data[0]
            new_noise = new_data[1]
            emb = new_data[3]

        optimiserData = pickle.loads(episode.optimiserData)
        self.parameters = optimiserData["parameters"]
        subOptimizer = AdamOptimizer(self.parameters["num_params"], self.parameters["learning_rate"])
        subOptimizer.fromDict(optimiserData["subOptimizerData"])

        noisyExecutions = list(episode.noisyExecutions.all())
        reward = np.array([n.fitness for n in noisyExecutions], dtype=np.float32)

        noisyExecutions_noise = np.array([createNoise(n.noiseseed, self.parameters["num_params"]) for n in noisyExecutions], dtype=np.float32)

        if self.parameters["rank_fitness"]:
            reward = compute_centered_ranks(reward)

        weightsNoise = episode.weightsNoise
        weights = weightsNoise[0]
        noiselevels = weightsNoise[1]

        #if self.parameters["weight_decay"] > 0: 
        #    used_weights = weights + noisyExecutions_noise * noiselevels
        #    l2_decay = compute_weight_decay( self.parameters["weight_decay"], used_weights)
        #    reward += l2_decay

        idx = np.argsort(reward)[::-1]

        # main bit:
        # standardize the rewards to have a gaussian distribution
        normalized_reward = (reward - np.mean(reward)) / np.std(reward)
        change_mu = 1./(len(noisyExecutions)*self.parameters["sigma"])*np.dot(noisyExecutions_noise.T, normalized_reward)
        noisyExecutions_noise = None

        subOptimizer.stepsize = self.parameters["learning_rate"]
        weights, update_ratio = subOptimizer.update(weights, -change_mu)

        # adjust sigma according to the adaptive sigma calculation
        if (self.parameters["sigma"] > self.parameters["sigma_limit"]):
            self.parameters["sigma"] *= self.parameters["sigma_decay"]

        if (self.parameters["learning_rate"] > self.parameters["learning_rate_limit"]):
            self.parameters["learning_rate"] *= self.parameters["learning_rate_decay"]

        weightsNoise = np.array([
            weights,                        # parameter 0 -> Weights
            [ self.parameters["sigma"], ],  # parameter 1 -> Noiselevels
        ])
        optimiserData = pickle.dumps({
            "parameters": self.parameters,
            "subOptimizerData" : subOptimizer.toDict(),
        },2)
    
        # Other optimisers may changed this
        count_factor = 1
        timespend_factor = 1
        steps_factor = 1

        return  [ weightsNoise, optimiserData, count_factor, timespend_factor, steps_factor]

'''


all_optimisers = {
    "OpenES" : {
        "description" : "deep-neuroevolution/ES",
        "class"       : OptimiserOpenES, 
    },
    "OpenES_Bugfixed" : {
        "description" : "deep-neuroevolution/ES Bugfixed",
        "class"       : OptimiserOpenES_Bugfixed, 
    },
    "OpenES_Ueber" : {
        "description" : "deep-neuroevolution/ES OpenAi/Ueber",
        "class"       : OptimiserESUeber, 
    }
}


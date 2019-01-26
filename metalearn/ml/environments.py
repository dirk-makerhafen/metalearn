
import gym
import tensorflow as tf
#from nn import tf_util



class EnvironmentInstance():
    def initialize(self):
        pass

    def reset(self):
        # reset to a state as if initialize was just run
        pass
    
    def hasNextObservation(self):
        pass

    def getNextObservation(self):
        pass

    def runAction(self, action):
        pass

    def close(self):
        pass


'''
class MetaTrainingEnvironmentInstance():
    def __init__(self, environment_name, arch_name):
        self.trainenv = environment_name # some env
        self.trainarch = arch_name # some arch_name

        self.episodes = 100
        self.episodes_done = 0

        self.exec_per_episode = 300
        self.exec_per_episode_done = 0

        num_params = getNumberOfParameter(self.trainenv, self.trainarch)
        nr_of_embeddings_optimiser = 4
        
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0,    high=100, shape=[ num_params, nr_of_embeddings_optimiser ]),   # Last embedding
            spaces.Box(low=-180, high=180, shape=[ num_params, 1 ]),  # Used Weights#
            spaces.Tuple((   # metaparameters
                spaces.Box(low=-180, high=180, shape=1), # fitness
                spaces.Box(low=-180, high=180, shape=1), # rank
                spaces.Box(low=-180, high=180, shape=1), # steps
                spaces.Box(low=-180, high=180, shape=1), # foo
            )),
        ))

        self.action_space = spaces.Tuple((
            spaces.Box(low=0,    high=100, shape=[ num_params, nr_of_embeddings_optimiser ]),   #new embedding
            spaces.Box(low=-180, high=180, shape=[ num_params, 1 ]),  # new Weights
            spaces.Box(low=-180, high=180, shape=[ num_params, 1 ]),  # new noise
        ))

        self.next_observation = (
            np.random.randn(shape=[ num_params, nr_of_embeddings_optimiser ]),      # last embedding
            np.random.randn(shape=[ num_params, 1 ]),        # used weights
            (
                0, # fitness
                0, # rank
                0, # steps
                0, # foo
            )
        )
        firststepdone = False
        self.fitnesses = []
        currentEpisode = None

    def initialize(self):
        pass

    def reset(self):
        pass
    
    def hasNextObservation(self):
        return self.next_observation != None:
            
    def getNextObservation(self):
        return self.next_observation
        
    def runAction(self, action):
        if currentEpisode == None:
            self.currentEpisode = Episode()
            return 0

        if currentEpisode.receivedAll == True:
            if more_to_go :
                self.currentEpisode = Episode()
                set weights from action
            return 0

        if self.currentEpisode.rundone != True:
            self.currentEpisode.runall()

        for item in currentEpisode.noiseexecutions:
            if item.received == False:  
                item.received = True
                self.next_observation = (
                    action.emb,      # last embedding
                    item.used_weights,        # used weights
                    (
                        item.fitness, # fitness
                        item.rank, # rank
                        item.steps, # steps
                        item.foo, # foo
                    )
                )
                return item.fitness
        self.next_observation = None
        return 0

        emb ,_, _ = action
        if firststepdone == False:  
            firststepdone = True
            emb ,weights, noise = action    
            self.fitnesses = []

            for j in range(self.exec_per_episode_done,self.exec_per_episode):
                rseed = 23
                weights_new = inputweights + np.random.randn(23)
                fitness = run_one_turn(self, inputweights )
                self.exec_per_episode_done += 1
                self.fitnesses.append(fitness)
                if self.exec_per_episode_done == self.exec_per_episode:
                    s = sum(fitnesses)
                    self.fitnesses = []
                    return s
                return 0
            self.episodes_to_do += 1
            self.exec_per_episode_done = 0
            emb ,weights, noise = action    


        return None

    def run_one_turn(self, inputweights ):
        
        self.trainarch.initialize(environment.observation_space, environment.action_space, weights_new)
        fitness = 0
        while self.trainenv.hasNextObservation():
            observation = self.trainenv.getNextObservation()
            action = self.trainarch.run(observation)
            fitness += self.trainenv.runAction(action) 
            #env.env.render()
            steps += 1
            if steps >= noisyExecution["max_steps"]:
                break
            if int(time.time() - start) >= noisyExecution["max_timespend"]:
                break
        return fitness


    def close(self):
        pass
{ 
    'MetaTraining - Atari Frostbite-v0 - GAAtariPolicy elu' : { 
        "description": "OpenAI Default Env",
        "class": factory(MetaTrainingEnvironmentInstance, environment_name = 'Atari Frostbite-v0', arch_name = "GAAtariPolicy elu"),
}

class DummyEnvironmentInstance():
    def __init__(self):
        pass

    def initialize(self):
        pass

    def reset(self):
        pass
    
    def hasNextObservation(self):
        return False

    def getNextObservation(self):
        return None

    def runAction(self, action):
        return None

    def close(self):
        pass

'''

class OpenAiGymEnvironmentInstance(EnvironmentInstance):
    def __init__(self, name):
        self.name = name
        self.gym = None

    def initialize(self):
        self.gym = gym.make(self.name)
        self.observation_space = self.gym.observation_space
        self.action_space = self.gym.action_space
        self.need_firststep = True
        self.observation = None
        self.reward = None
        self.done = False
        self.info = None

    def reset(self):
        self.observation_space = self.gym.observation_space
        self.action_space = self.gym.action_space
        self.need_firststep = True
        self.observation = None
        self.reward = None
        self.done = False
        self.info = None
        self.gym.reset()

    def hasNextObservation(self):
        if self.need_firststep == True:
            self.firststep()
        if self.done == False and self.observation  is not  None:
            return True
        return False

    def getNextObservation(self):
        if self.need_firststep == True:
            self.firststep()
        if self.done == False and self.observation is not  None:
            return self.observation
        return None

    def runAction(self, action):
        if self.need_firststep == True:
            self.firststep()
        self.observation, self.reward, self.done, self.info = self.gym.step(action)
        return self.reward

    def firststep(self):
        self.need_firststep = False
        self.observation = self.gym.reset()
        self.observation, self.reward, self.done, self.info = self.gym.step([1])
    
    def close(self):
        if self.gym != None:    
            self.gym.close()


def factory(_class, **args):
    def f():
        return _class(**args)
    return f

all_environments = {
    'Copy-v0'                 : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'Copy-v0'),
    },
    'DuplicatedInput-v0'      : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'DuplicatedInput-v0'), 
    },
    'RepeatCopy-v0'           : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'RepeatCopy-v0'), 
    } ,
    'Reverse-v0'              : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'Reverse-v0'), 
    } ,
    'ReversedAddition-v0'     : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'ReversedAddition-v0'), 
    },
    'ReversedAddition3-v0'    : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'ReversedAddition3-v0'), 
    },
    'Acrobot-v1'              : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'Acrobot-v1'), 
    },
    'CartPole-v0'             : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'CartPole-v0'), 
    },
    'MountainCar-v0'          : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'MountainCar-v0'), 
    },
    'MountainCarContinuous-v0': { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'MountainCarContinuous-v0'), 
    },
    'Pendulum-v0'             : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Pendulum-v0'), 
    },
    'Blackjack-v0'            : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Blackjack-v0'), 
    },
    'FrozenLake-v0'           : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'FrozenLake-v0'), 
    },
    'FrozenLake8x8-v0'        : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'FrozenLake8x8-v0'), 
    },
    'GuessingGame-v0'         : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'GuessingGame-v0'), 
    },
    'HotterColder-v0'         : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'HotterColder-v0'), 
    },
    'NChain-v0'               : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name = 'NChain-v0'), 
    },
    'Roulette-v0'             : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Roulette-v0'), 
    },
    'Taxi-v2'                 : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Taxi-v2'), 
    },
    'BipedalWalker-v2'        : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'BipedalWalker-v2'), 
    },
    'BipedalWalkerHardcore-v2': { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'BipedalWalkerHardcore-v2'), 
    },
    'LunarLander-v2'          : { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'LunarLander-v2'), 
    },
    'LunarLanderContinuous-v2': { 
        "description": "OpenAI Default Env",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'LunarLanderContinuous-v2'), 
    }, 
}


all_environments = {
    'Atari Frostbite-v0'                 : { 
        "description": "Gym Atari Frostbite-v0",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Frostbite-v0'),
    },
    'Atari FreewayNoFrameskip-v4'                 : { 
        "description": "Gym Atari FreewayNoFrameskip-v4",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'FreewayNoFrameskip-v4'),
    },
    'Atari Alien-v0'                 : { 
        "description": "Gym Atari Alien-v0",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Alien-v0'),
    },
    'Atari Pong-v0'                 : { 
        "description": "Gym Atari Pong-v0",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Pong-v0'),
    },
    'Atari Gravitar-v0'                 : { 
        "description": "Gym Atari Gravitar-v0",
        "class": factory(OpenAiGymEnvironmentInstance, name =  'Gravitar-v0'),
    },
}



'''
class aaOpenAiGymEnvironmentInstance():
    def __init__(self, name):
        self.name = name

    def init(self, architectureClass):
        env = gym.make(exp['env_id'])
        #if exp['policy']['type'] == "ESAtariPolicy":
        #    from .atari_wrappers import wrap_deepmind
        #    env = wrap_deepmind(env)
        sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))
        policy = architectureClass()._initialize(env.observation_space, env.action_space, **exp['policy']['args'])
        tf_util.initialize()
        config = Config(**exp['config'])
        self.config = config
        self.env = env
        self.sess = sess
        self.policy = policy
        return config, env, sess, policy

    def run(self, weights, noiseLevels, noiseseed):
        weights = weights + (noiseLevels * createNoise(noiseseed,len(weights)))

        policy.set_trainable_flat(weights)
        eval_rews, eval_length, _ = policy.rollout(env, timestep_limit=task_data.timestep_limit)
        eval_return = eval_rews.sum()
        logger.info('Eval result: task={} return={:.3f} length={}'.format(task_id, eval_return, eval_length))
'''

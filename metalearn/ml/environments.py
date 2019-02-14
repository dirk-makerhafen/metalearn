import numpy
import gym
from gym import spaces

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
        pass # return reward

    def close(self):
        pass




class EnvironmentLens():
    def __init__(self, max_x, max_y):

        self.max_x = 10 # in px
        self.max_y = 10

        self.observation_space = spaces.Tuple((
            spaces.Box(low=-1,   high=1, shape=[ 2 ]), # center  (x,y) # -1,-1 = bottom left
            spaces.Box(low= 0,   high=1, shape=[ 2 ]), # dimensions  (x,y) #1=full input image, 0=max_x/max_y pixels of inputimg
            spaces.Box(low=-1,   high=1, shape=[ 2 ]), # scaling  (x,y) # 0: linear, 1=more pixels at center, -1=more pixels at edge
            spaces.Box(low= 0,   high=1, shape=[ max_x, max_y, 3 ]), # image pixels
        ))

        self.action_space = spaces.Tuple((
            spaces.Box(low=-1,   high=1, shape=[ 2 ]), # center  (x,y)
            spaces.Box(low= 0,   high=1, shape=[ 2 ]), # dimensions  (x,y)
            spaces.Box(low=-1,   high=1, shape=[ 2 ]), # scaling  (x,y)   
        )) 



class EnvironmentMnist(EnvironmentInstance):
    def __init__(self,nr_of_embeddings_per_weight, nr_of_embeddings):
        self.dataset = []
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0,    high=255, shape=[ 28, 28 ]),
        ))

        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=9, shape=[ 1 ]),  
        )) 

        self.max_observations = 1000
        self.obs_nr = 0

    def initialize(self):
        self.dataset = []
        for line in random.sample(open("metalearn/ml/datasets/mnist_train.csv","rb").read().split("\n"), self.max_observations):
            linebits = line.split(",")
            imarray = numpy.asfarray(linebits[1:]).reshape((28,28))
            value = float(linebits[0])
            self.dataset.append([value, imarray])

    def reset(self):
        self.obs_nr = 0
        self.initialize()

    def hasNextObservation(self):
        if self.obs_nr < self.max_observations:
            return True

    def getNextObservation(self):
        if not self.hasNextObservation():
            return None
        value, img = self.dataset[self.obs_nr]
        self.obs_nr += 1
        return img

    def runAction(self, action):
        if self.obs_nr == 0: 
            return 0
        print(self.dataset[ self.obs_nr - 1 ] , action)
        if int(self.dataset[ self.obs_nr - 1 ][0]) == int(action):
            return 1
        return 0

    def close(self):
        pass



#this is used by tasks.py on_Experiment_created when the optimiser of the optimiser experiment is initialized
class OptimiserMetaESEnvironment(EnvironmentInstance):
    def __init__(self,nr_of_embeddings_per_weight, nr_of_embeddings):
        nr_of_weights = 10 # ! this must not influence the network size in term of lernable parameters, its just a placeholder.
        
        self.observation_space = spaces.Tuple((
            # meta data embedding
            spaces.Box(low=-1, high=1, shape=[ nr_of_weights, nr_of_embeddings_per_weight ]),# per_weight_embeddings
            spaces.Box(low=-1, high=1, shape=[ nr_of_embeddings ]),  # embedding

            # noisyExecution data
            spaces.Box(low=-180, high=180, shape=[ nr_of_weights ]),  # used_weights

            spaces.Box(low=-1, high=1, shape=[ 8 ]), # noisyExecution meta data 
            #spaces.Box(low=-1, high=1, shape=[ 1 ]), # fitness                via fitness/abs(fitness) * tanh( log( 1 + log( 1 + abs(fitness)) ) )  # sign * loglog scale
            #spaces.Box(low=-1, high=1, shape=[ 1 ]), # fitness_scaled
            #spaces.Box(low= 0, high=1, shape=[ 1 ]), # fitness_rank
            #spaces.Box(low=-1, high=1, shape=[ 1 ]), # fitness_norm           via fitness/abs(fitness) * tanh( log( 1 + log( 1 + abs(fitness)) ) )  # sign * loglog scale 
            #spaces.Box(low=-1, high=1, shape=[ 1 ]), # fitness_norm_scaled
            #spaces.Box(low= 0, high=1, shape=[ 1 ]), # steps                  via tanh( log( 1 + log( 1 + steps) ) )
            #spaces.Box(low= 0, high=1, shape=[ 1 ]), # first_rewarded_step    via tanh( log( 1 + log( 1 + first_rewarded_step) ) )
            #spaces.Box(low= 0, high=1, shape=[ 1 ]), # timespend              via tanh( log( 1 + log( 1 + timespend) ) )

            # episode data
            spaces.Box(low=0,  high=1, shape=[ 8 ]),  # episode meta data
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # episode_nr             via tanh( log( 1 + log( 1 + episode_nr) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # noisyExecutions_min    via tanh( log( 1 + log( 1 + noisyExecutions_min) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # noisyExecutions_max    via tanh( log( 1 + log( 1 + noisyExecutions_max) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # noisyExecutions_actual via tanh( log( 1 + log( 1 + noisyExecutions_actual) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # steps_min              via tanh( log( 1 + log( 1 + steps_min) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # steps_max              via tanh( log( 1 + log( 1 + steps_max) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # timespend_min          via tanh( log( 1 + log( 1 + timespend_min) ) )
            #spaces.Box(low=0,  high=1, shape=[ 1 ]), # timespend_max          via tanh( log( 1 + log( 1 + timespend_max) ) )
        ))
                
        self.action_space = spaces.Tuple((
            spaces.Box(low=0, high=1, shape=[ nr_of_weights, nr_of_embeddings_per_weight ]),   #per_weight_embeddings
            spaces.Box(low=0, high=1, shape=[ nr_of_embeddings ]),  # embedding
            spaces.Box(low=-180, high=180, shape=[ nr_of_weights ]),  # new Weights
            spaces.Box(low=-180, high=180, shape=[ nr_of_weights ]),  # new noise
            spaces.Box(low=0, high=1, shape=[ 4 ]),  # out_factors
            #spaces.Box(low=0, high=1, shape=[ 1 ]),  # noisyExecutions_max_factor
            #spaces.Box(low=0, high=1, shape=[ 1 ]),  # timespend_factor
            #spaces.Box(low=0, high=1, shape=[ 1 ]),  # steps_factor
            #spaces.Box(low=0, high=1, shape=[ 1 ]),  # steps_unrewarded_factor
        ))   

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


class OpenAiGym(EnvironmentInstance):
    def __init__(self, gymname):
        self.gymname = gymname
        self.gym = None

    def initialize(self):
        self.gym = gym.make(self.gymname)
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


# create these if you db is empty
default_models = [
        {
            "name":"Frostbite-v0",
            "groupname":"Atari",
            "description": "",
            "classname":"OpenAiGym",
            "classargs":[[ "gymname", "Frostbite-v0"]],
        },{
            "name":"FreewayNoFrameskip-v4",
            "groupname":"Atari",
            "description": "",
            "classname":"OpenAiGym",
            "classargs":[[ "gymname", "FreewayNoFrameskip-v4"]],
        },{
            "name":"Alien-v0",
            "groupname":"Atari",
            "description": "",
            "classname":"OpenAiGym",
            "classargs":[[ "gymname", "Alien-v0"]],
        },{
            "name":"Pong-v0",
            "groupname":"Atari",
            "description": "",
            "classname":"OpenAiGym",
            "classargs":[[ "gymname", "Pong-v0"]],
        },{
            "name":"Gravitar-v0",
            "groupname":"Atari",
            "description": "",
            "classname":"OpenAiGym",
            "classargs":[[ "gymname", "Gravitar-v0"]],
        },{
            "name":"OptimiserMetaESEnvironment",
            "groupname":"Optimiser",
            "description": "",
            "classname":"OptimiserMetaESEnvironment",
            "classargs":[["nr_of_embeddings_per_weight",5] , [ "nr_of_embeddings", 20 ] ],
        },
    ]



'''
class aaOpenAiGym():
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

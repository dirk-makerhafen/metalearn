import logging
import pickle
import time

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import gc
from . import tf_util

logger = logging.getLogger(__name__)

def createNoise(seed, width):
    r = np.random.RandomState(seed)
    return r.randn(width).astype(np.float32)


class Architecture():
    def initialize(self, input_space, output_space, weights = None):
        print("initialize.initialize")
        self.input_space = input_space
        self.output_space = output_space
        self.inputs = []
        self.outputs = []

        self.graph = tf.Graph()
        self.session = None

        with self.graph.as_default():
        
            self.scope = self._initialize()

            self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)

            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
            self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
            self._setfromflat = tf_util.SetFromFlat(self.trainable_variables)
            self._getflat = tf_util.GetFlat(self.trainable_variables)

            self.session = tf.Session()
            self.session.run(tf.initialize_variables(tf.all_variables()))

            if weights is not None:
                self.set_weights(weights)

            # Debug Print
            logger.info('Trainable variables ({} parameters)'.format(self.num_params))
            for v in self.trainable_variables:
                shp = v.get_shape().as_list()
                logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
            logger.info('All variables')
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        print("initialize.initialize done")

    def reset(self, weights):
        self.set_weights(weights)
        # reset to a state as if initialize was just run

    def run(self, inputvals):
        assert len(inputvals) == len(self.inputs)
        feed_dict = dict(zip(self.inputs, inputvals))
        results = self.session.run(self.outputs, feed_dict=feed_dict)
        return results

    def set_weights(self, weights):
        self._setfromflat(weights)
    
    def get_weights(self):
        return self._getflat()
    
    def close(self):
        tf.reset_default_graph()
        self.session.close()
        self.session = None
        
        self.graph = None
        gc.collect()


class Architecture_GAAtariPolicy(Architecture):
    def __init__(self, nonlin_type):
        self.nonlin_type = nonlin_type

    def _initialize(self):
        print("Architecture_GAAtariPolicy._initialize")
        self.ob_space_shape = self.input_space.shape
        self.ac_space = self.output_space
        self.ac_init_std = 0.1
        self.num_actions = self.ac_space.n
        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': tf_util.lrelu, 'elu': tf.nn.elu}[self.nonlin_type]


        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))

            x = o
            x = self.nonlin(tf_util.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4, std=1.0))
            x = self.nonlin(tf_util.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2, std=1.0))

            x = tf_util.flattenallbut0(x)
            x = self.nonlin(tf_util.dense(x, 256, 'fc', tf_util.normc_initializer(1.0), std=1.0))

            a = tf_util.dense(x, self.num_actions, 'out', tf_util.normc_initializer(self.ac_init_std), std=self.ac_init_std)

            a = tf.argmax(a,1)

            self.inputs = [o]
            self.outputs = [a]
            return scope


class Architecture_ESAtariPolicy(Architecture):
    def _initialize(self):
        self.ob_space_shape = self.input_space.shape
        self.ac_space = self.output_space
        self.num_actions = self.ac_space.n

        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))

            x = layers.convolution2d(o, num_outputs=16, kernel_size=8, stride=4, activation_fn=None, scope='conv1')
            x = layers.batch_norm(x, scale=True, is_training=False, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)
            x = layers.convolution2d(x, num_outputs=32, kernel_size=4, stride=2, activation_fn=None, scope='conv2')
            x = layers.batch_norm(x, scale=True, is_training=False, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)

            x = layers.flatten(x)
            x = layers.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc')
            x = layers.batch_norm(x, scale=True, is_training=False, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)
            x = layers.fully_connected(x, num_outputs=self.num_actions, activation_fn=None, scope='out')
            a = tf.argmax(x,1)

            self._run = tf_util.function([o] , a)
            return scope


class Architecture_MujocoPolicy(Architecture):
    def __init__(self, nonlin_type, hidden_dims):
        self.nonlin_type = nonlin_type    
        self.hidden_dims = hidden_dims

    def _initialize(self ):
        self.ob_space_shape = self.input_space.shape
        self.ac_space = self.output_space
        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': tf_util.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        assert len(self.ob_space_shape) == len(self.ac_space.shape) == 1
        assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)),'Action bounds required'

        with tf.variable_scope(type(self).__name__) as scope:
            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))
            o = tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0)

            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(tf_util.dense(x, hd, 'l{}'.format(ilayer), tf_util.normc_initializer(1.0)))

            adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low # Map to action

            a = tf_util.dense(x, adim, 'out', tf_util.normc_initializer(0.01))

            self.inputs = [o]
            self.outputs = [a]
            return scope


class Architecture_MetaES(Architecture):
    def __init__(self, nr_of_embeddings_per_weight, nr_of_embeddings):
        self.nr_of_embeddings_per_weight = nr_of_embeddings_per_weight    
        self.nr_of_embeddings = nr_of_embeddings    

    def _initialize(self ):
        with tf.variable_scope(type(self).__name__) as scope:
            nr_of_weights  = self.input_space.spaces[0].shape[0]
            assert self.nr_of_embeddings_per_weight == self.input_space.spaces[0].shape[1]
            assert self.nr_of_embeddings == self.input_space.spaces[2].shape[0]


            in_embeddings_per_weight = tf.placeholder(tf.float32, [1, nr_of_weights, self.nr_of_embeddings_per_weight ])
            in_weights               = tf.placeholder(tf.float32, [1, nr_of_weights ] )
            in_embeddings            = tf.placeholder(tf.float32, [1, self.nr_of_embeddings ] )
            in_episode_nr            = tf.placeholder(tf.float32, [1, 1 ]) 
            in_fitness               = tf.placeholder(tf.float32, [1, 1 ])
            in_rank                  = tf.placeholder(tf.float32, [1, 1 ])
            in_steps                 = tf.placeholder(tf.float32, [1, 1 ])
            in_count                 = tf.placeholder(tf.float32, [1, 1 ])

            self.inputs = [ in_embeddings_per_weight, in_weights, in_embeddings, in_episode_nr, in_fitness, in_rank, in_steps, in_count ]

            # Meta data embedding stuff
            meta = tf.concat(values=[ in_embeddings, in_episode_nr, in_fitness, in_rank, in_steps, in_count],axis=1)
            meta = layers.fully_connected(meta, num_outputs=self.nr_of_embeddings*4, activation_fn=tf.tanh)
            meta = layers.fully_connected(meta, num_outputs=self.nr_of_embeddings*4, activation_fn=tf.tanh)
            meta = layers.fully_connected(meta, num_outputs=self.nr_of_embeddings*3, activation_fn=tf.tanh)

            tmp_embeddings = layers.fully_connected(meta, num_outputs=self.nr_of_embeddings_per_weight  , activation_fn=tf.tanh)     

            # weights related operations

            # expand in_weights to have same rank as in_embeddings_per_weight
            in_weights = tf.expand_dims(in_weights, axis=2) # [1, nr_of_weights] -> [1, nr_of_weights, 1]
            print("in_weights", in_weights)

            # expand and tile tmp_embeddings to have same rank as in_embeddings_per_weight
            tmp_embeddings = tf.expand_dims(tmp_embeddings, axis=1)  # [1, nr_of_embeddings_per_weight] -> [1, 1, nr_of_embeddings_per_weight]
            print("tmp_embeddings", tmp_embeddings)
            tmp_embeddings = tf.tile(tmp_embeddings, multiples=[1, nr_of_weights, 1]) # [1, 1, nr_of_embeddings_per_weight]  -> [1, nr_of_weights, nr_of_embeddings_per_weight]
            print("tmp_embeddings", tmp_embeddings)

            # concat in_weights and in_embeddings_per_weight and tmp_embeddings
            in_data_per_weight = tf.concat(values = [ in_weights,  in_embeddings_per_weight, tmp_embeddings ], axis = 2 ) # [1, nr_of_weights, 1] , [1, nr_of_weights, nr_of_embeddings_per_weight] -> [1, nr_of_weights, 1+nr_of_embeddings_per_weight]
            print("in_data_per_weight", in_data_per_weight)

            in_data_per_weight  = layers.fully_connected(in_data_per_weight, num_outputs=self.nr_of_embeddings_per_weight*4, activation_fn=tf.tanh)
            in_data_per_weight  = layers.fully_connected(in_data_per_weight, num_outputs=self.nr_of_embeddings_per_weight*4, activation_fn=tf.tanh)

            out_embeddings_per_weight  = layers.fully_connected(in_data_per_weight, num_outputs=self.nr_of_embeddings_per_weight, activation_fn=tf.tanh)
            out_weights = layers.fully_connected(in_data_per_weight, num_outputs=1, activation_fn=None)
            out_weights = tf.squeeze(out_weights, axis=2) 
            out_noise   = layers.fully_connected(in_data_per_weight, num_outputs=1, activation_fn=None)
            out_noise   = tf.squeeze(out_noise, axis=2) 

            out_embeddings       = layers.fully_connected(meta, num_outputs=self.nr_of_embeddings  , activation_fn=tf.tanh)     
            out_count_factor     = layers.fully_connected(meta, num_outputs=1, activation_fn=tf.tanh)
            out_timespend_factor = layers.fully_connected(meta, num_outputs=1, activation_fn=tf.tanh)
            out_steps_factor     = layers.fully_connected(meta, num_outputs=1, activation_fn=tf.tanh)

            print("out_embeddings_per_weight", out_embeddings_per_weight)
            print("out_weights", out_weights)
            print("out_noise", out_noise)

            self.outputs = [
                out_embeddings_per_weight,
                out_weights,
                out_noise,
                out_embeddings,
                out_count_factor,
                out_timespend_factor,
                out_steps_factor,
            ]
            
            return scope

    def run(self, inputvals):
        assert len(inputvals) == len(self.inputs)

        for i in range(0,len(inputvals)):
            inputvals[i] = np.array([inputvals[i]])

        feed_dict = dict(zip(self.inputs, inputvals))
        try:
            outputs = self.session.run(self.outputs, feed_dict=feed_dict)
        except Exception as e:
            print(e)

        for i in range(0,len(outputs)):
            outputs[i] = outputs[i][0].astype(np.float32)# remove batch dim
        outputs[2] = outputs[2].astype(np.float16)    # noiselevel is float16 

        print(len(outputs))

        return outputs


# create these if your db is empty
default_models = [
        {
            "name":"GAAtariPolicy",
            "description": "",
            "classname":"Architecture_GAAtariPolicy",
            "classargs":[[ "nonlin_type", "tanh"]],
        },{
            "name":"GAAtariPolicy",
            "description":"",
            "classname":"Architecture_GAAtariPolicy",
            "classargs":[[ "nonlin_type", "relu"]],
        },{
            "name":"GAAtariPolicy",
            "description":"",
            "classname":"Architecture_GAAtariPolicy",
            "classargs":[[ "nonlin_type", "lrelu"]],
        },{
            "name":"GAAtariPolicy",
            "description":"",
            "classname":"Architecture_GAAtariPolicy",
            "classargs":[[ "nonlin_type", "elu"]],
        },{
            "name":"ESAtariPolicy",
            "description":"",
            "classname":"Architecture_ESAtariPolicy",
            "classargs":[],
        },{
            "name":"MetaES",
            "description":"",
            "classname":"Architecture_MetaES",
            "classargs":[["nr_of_embeddings_per_weight",5] , [ "nr_of_embeddings", 20 ] ],
        },
    ]






















'''
class GAAtariPolicy(Policy):
    def _initialize(self, ob_space, ac_space, nonlin_type, ac_init_std=0.1):
        self.ob_space_shape = ob_space.shape
        self.ac_space = ac_space
        self.ac_init_std = ac_init_std
        self.num_actions = self.ac_space.n
        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': tf_util.lrelu, 'elu': tf.nn.elu}[nonlin_type]


        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))

            a = self._make_net(o)
            self._act = tf_util.function([o] , a)
        return scope

    def _make_net(self, o):
        x = o
        x = self.nonlin(tf_util.conv(x, name='conv1', num_outputs=16, kernel_size=8, stride=4, std=1.0))
        x = self.nonlin(tf_util.conv(x, name='conv2', num_outputs=32, kernel_size=4, stride=2, std=1.0))

        x = tf_util.flattenallbut0(x)
        x = self.nonlin(tf_util.dense(x, 256, 'fc', tf_util.normc_initializer(1.0), std=1.0))

        a = tf_util.dense(x, self.num_actions, 'out', tf_util.normc_initializer(self.ac_init_std), std=self.ac_init_std)

        return tf.argmax(a,1)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return False

    # Dont add random noise since action space is discrete
    def act(self, train_vars, random_stream=None):
        return self._act(train_vars)

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, worker_stats=None, policy_seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []; novelty_vector = []
        rollout_details = {}
        t = 0

        if save_obs:
            obs = []

        if policy_seed:
            env.seed(policy_seed)
            np.random.seed(policy_seed)
            if random_stream:
                random_stream.seed(policy_seed)

        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act(ob[None], random_stream=random_stream)[0]

            if save_obs:
                obs.append(ob)
            ob, rew, done, info = env.step(ac)
            rews.append(rew)

            t += 1
            if render:
                env.render()
            if done:
                break

        # Copy over final positions to the max timesteps
        rews = np.array(rews, dtype=np.float32)
        novelty_vector = env.unwrapped._get_ram() # extracts RAM state information
        if save_obs:
            return rews, t, np.array(obs), np.array(novelty_vector)
        return rews, t, np.array(novelty_vector)




class ESAtariPolicy(Policy):
    def _initialize(self, ob_space, ac_space):
        self.ob_space_shape = ob_space.shape
        self.ac_space = ac_space
        self.num_actions = ac_space.n

        with tf.variable_scope(type(self).__name__) as scope:
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space_shape))
            is_ref_ph = tf.placeholder(tf.bool, shape=[])

            a = self._make_net(o, is_ref_ph)
            self._act = tf_util.function([o, is_ref_ph] , a)
        return scope

    def _make_net(self, o, is_ref):
        x = o
        x = layers.convolution2d(x, num_outputs=16, kernel_size=8, stride=4, activation_fn=None, scope='conv1')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)
        x = layers.convolution2d(x, num_outputs=32, kernel_size=4, stride=2, activation_fn=None, scope='conv2')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)

        x = layers.flatten(x)
        x = layers.fully_connected(x, num_outputs=256, activation_fn=None, scope='fc')
        x = layers.batch_norm(x, scale=True, is_training=is_ref, decay=0., updates_collections=None, activation_fn=tf.nn.relu, epsilon=1e-3)
        a = layers.fully_connected(x, num_outputs=self.num_actions, activation_fn=None, scope='out')
        return tf.argmax(a,1)

    def set_ref_batch(self, ref_batch):
        self.ref_list = []
        self.ref_list.append(ref_batch)
        self.ref_list.append(True)

    @property
    def needs_ob_stat(self):
        return False

    @property
    def needs_ref_batch(self):
        return True

    def initialize_from(self, filename):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

    def act(self, train_vars, random_stream=None):
        return self._act(*train_vars)


    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, worker_stats=None, policy_seed=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []; novelty_vector = []
        t = 0

        if save_obs:
            obs = []

        if policy_seed:
            env.seed(policy_seed)
            np.random.seed(policy_seed)
            if random_stream:
                random_stream.seed(policy_seed)

        ob = env.reset()
        self.act(self.ref_list, random_stream=random_stream) #passing ref batch through network

        for _ in range(timestep_limit):
            start_time = time.time()
            ac = self.act([ob[None], False], random_stream=random_stream)[0]

            if worker_stats:
                worker_stats.time_comp_act += time.time() - start_time

            start_time = time.time()
            ob, rew, done, info = env.step(ac)
            ram = env.unwrapped._get_ram() # extracts RAM state information

            if save_obs:
               obs.append(ob)
            if worker_stats:
                worker_stats.time_comp_step += time.time() - start_time

            rews.append(rew)
            novelty_vector.append(ram)

            t += 1
            if render:
                env.render()
            if done:
                break

        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs), np.array(novelty_vector)
        return rews, t, np.array(novelty_vector)



class MujocoPolicy(Policy):
    def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type):
        self.ac_space = ac_space
        self.ac_bins = ac_bins
        self.ac_noise_std = ac_noise_std
        self.hidden_dims = hidden_dims
        self.connection_type = connection_type

        assert len(ob_space.shape) == len(self.ac_space.shape) == 1
        assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
            'Action bounds required'

        self.nonlin = {'tanh': tf.tanh, 'relu': tf.nn.relu, 'lrelu': tf_util.lrelu, 'elu': tf.nn.elu}[nonlin_type]

        with tf.variable_scope(type(self).__name__) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, ob_space.shape)
            in_std = tf.placeholder(tf.float32, ob_space.shape)
            self._set_ob_mean_std = tf_util.function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
            a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
            self._act = tf_util.function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.connection_type == 'ff':
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(tf_util.dense(x, hd, 'l{}'.format(ilayer), tf_util.normc_initializer(1.0)))
        else:
            raise NotImplementedError(self.connection_type)

        # Map to action
        adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
        assert isinstance(self.ac_bins, str)
        ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')

        if ac_bin_mode == 'uniform':
            # Uniformly spaced bins, from ac_space.low to ac_space.high
            num_ac_bins = int(ac_bin_arg)
            aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
            ac_range_1a = (ahigh - alow)[None, :]
            a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]

        elif ac_bin_mode == 'custom':
            # Custom bins specified as a list of values from -1 to 1
            # The bins are rescaled to ac_space.low to ac_space.high
            acvals_k = np.array(list(map(float, ac_bin_arg.split(','))), dtype=np.float32)
            logger.info('Custom action values: ' + ' '.join('{:.3f}'.format(x) for x in acvals_k))
            assert acvals_k.ndim == 1 and acvals_k[0] == -1 and acvals_k[-1] == 1
            acvals_ak = (
                (ahigh - alow)[:, None] / (acvals_k[-1] - acvals_k[0]) * (acvals_k - acvals_k[0])[None, :]
                + alow[:, None]
            )

            aidx_na = bins(x, adim, len(acvals_k), 'out')  # values in [0, k-1]
            a = tf.gather_nd(
                acvals_ak,
                tf.concat(2, [
                    tf.tile(np.arange(adim)[None, :, None], [tf.shape(aidx_na)[0], 1, 1]),
                    tf.expand_dims(aidx_na, -1)
                ])  # (n,a,2)
            )  # (n,a)
        elif ac_bin_mode == 'continuous':
            a = tf_util.dense(x, adim, 'out', tf_util.normc_initializer(0.01))
        else:
            raise NotImplementedError(ac_bin_mode)

        return a

    def act(self, ob, random_stream=None):
        a = self._act(ob)
        if random_stream is not None and self.ac_noise_std != 0:
            a += random_stream.randn(*a.shape) * self.ac_noise_std
        return a

    @property
    def needs_ob_stat(self):
        return True

    @property
    def needs_ref_batch(self):
        return False

    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)


    def _get_pos(self, model):
        mass = model.body_mass
        xpos = model.data.xipos
        center = (np.sum(mass * xpos, 0) / np.sum(mass))
        return center[0], center[1], center[2]


    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None, policy_seed=None, bc_choice=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []
        x_traj, y_traj = np.zeros(timestep_limit), np.zeros(timestep_limit)
        t = 0
        if save_obs:
            obs = []

        if policy_seed:
            env.seed(policy_seed)
            np.random.seed(policy_seed)
            if random_stream:
                random_stream.seed(policy_seed)

        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act(ob[None], random_stream=random_stream)[0]
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac)
            x_traj[t], y_traj[t], _ = self._get_pos(env.unwrapped.model)
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break

        x_pos, y_pos, _ = self._get_pos(env.unwrapped.model)
        rews = np.array(rews, dtype=np.float32)
        x_traj[t:] = x_traj[t-1]
        y_traj[t:] = y_traj[t-1]
        if bc_choice and bc_choice == "traj":
            novelty_vector = np.concatenate((x_traj, y_traj), axis=0)
        else:
            novelty_vector = np.array([x_pos, y_pos])
        if save_obs:
            return rews, t, np.array(obs), novelty_vector
        return rews, t, novelty_vector





class Policy:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.scope = self._initialize(*args, **kwargs)
        self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)

        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        self._setfromflat = tf_util.SetFromFlat(self.trainable_variables)
        self._getflat = tf_util.GetFlat(self.trainable_variables)

        logger.info('Trainable variables ({} parameters)'.format(self.num_params))
        for v in self.trainable_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
        logger.info('All variables')
        for v in self.all_variables:
            shp = v.get_shape().as_list()
            logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))

        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        self.set_all_vars = tf_util.function(
            inputs=placeholders,
            outputs=[],
            updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
        )

    def reinitialize(self):
        for v in self.trainable_variables:
            v.reinitialize.eval()

    def _initialize(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, filename):
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w', libver='latest') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    # === Rollouts/training ===

    def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False, random_stream=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        rews = []
        t = 0
        if save_obs:
            obs = []
        ob = env.reset()
        for _ in range(timestep_limit):
            ac = self.act(ob[None], random_stream=random_stream)[0]
            if save_obs:
                obs.append(ob)
            ob, rew, done, _ = env.step(ac)
            rews.append(rew)
            t += 1
            if render:
                env.render()
            if done:
                break
        rews = np.array(rews, dtype=np.float32)
        if save_obs:
            return rews, t, np.array(obs)
        return rews, t

    def act(self, ob, random_stream=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    @property
    def needs_ob_stat(self):
        raise NotImplementedError

    def set_ob_stat(self, ob_mean, ob_std):
        raise NotImplementedError


def bins(x, dim, num_bins, name):
    scores = tf_util.dense(x, dim * num_bins, name, tf_util.normc_initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1




'''
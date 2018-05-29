import tensorflow as tf
import numpy as np
import gym
import ipdb
# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        if (type(env.action_space) is gym.spaces.discrete.Discrete):
            action_size = 1
        else:
            action_size = env.action_space.shape[0]
        self.normalization = normalization
        self.x = tf.placeholder(tf.float64, [None, action_size+env.observation_space.shape[0]])
        self.delta_s = build_mlp(self.x, env.observation_space.shape[0], scope = 'mlp', n_layers=n_layers, size=size, activation=activation, output_activation=output_activation)
        self.delta_s_target = tf.placeholder(tf.float64, [None, env.observation_space.shape[0]])
        self.loss = tf.losses.mean_squared_error(self.delta_s_target, self.delta_s)
        self.update_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess = sess
        self.iterations = iterations
        self.batch_size = batch_size
    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """
        
        """YOUR CODE HERE """
        print('fitting dynamics model')
        s_normalized = self.normalization(data['observations'])
        a_normalized = self.normalization(data['actions'], act=True, delta_s=False)
        s_delta_normalized = self.normalization(data['next_observations'] - data['observations'], delta_s=True)
        #self.mu_delta_states = np.mean(data['next_observations'] - data['observations'], axis=0)
        #self.sig_delta_states = np.std(data['next_observations'] - data['observations'], axis=0)
        #ipdb.set_trace() 
        num_samples = s_normalized.shape[0]
        for itr in range(self.iterations):
            batch_idx = np.random.randint(0,num_samples-1, self.batch_size)
            _, loss = self.sess.run([self.update_op,self.loss], feed_dict = {self.x:np.concatenate([s_normalized[batch_idx], a_normalized[batch_idx]],axis=-1), 
            self.delta_s_target:s_delta_normalized[batch_idx]})
            print('iteration ' + str(itr) + ':' + 'loss ' + str(loss))
        return

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        s_normalized = self.normalization(states, act=False, delta_s=False)
        a_normalized = self.normalization(actions, act=True, delta_s=False)
        s_delta_normalized = self.sess.run(self.delta_s, feed_dict = {self.x:np.concatenate([s_normalized, a_normalized],axis=-1)})
        s_delta = self.normalization(s_delta_normalized, denorm=True)
        return states + s_delta


        


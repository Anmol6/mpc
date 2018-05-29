import numpy as np
from cost_functions import trajectory_cost_fn
import time
import ipdb

class Controller():
	def __init__(self):
            pass
	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
            pass

class RandomController(Controller):
        def __init__(self, env):
            """ YOUR CODE HERE """
            self.env=env
            pass
        def get_action(self, state):
            """ YOUR CODE HERE """
            """ Your code should randomly sample an action uniformly from the action space """
            return self.env.action_space.sample()  	
    
class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, env, dyn_model, horizon=5, cost_fn=None, num_simulated_paths=10):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
            """ Note: be careful to batch your simulations through the model for speed """
            states = np.repeat(np.expand_dims(state,axis=0),self.num_simulated_paths, axis=0)
            def get_rand_actions():
                actions = []
                for i in range(self.num_simulated_paths):
                    actions.append(self.env.action_space.sample())
                actions = np.array(actions)
                return actions
            actions = get_rand_actions()            
            states_traj = []
            next_states_traj = []
            actions_traj = []
             
            for i in range(self.horizon):
                next_states = self.dyn_model.predict(states,actions)
                actions_traj.append(actions)  
                states_traj.append(states)
                next_states_traj.append(next_states)
                
                states = next_states
                actions = get_rand_actions()
            
            
            states_traj = np.swapaxes(np.array(states_traj),0,1)
            actions_traj = np.swapaxes(np.array(actions_traj),0,1)
            next_states_traj = np.swapaxes(np.array(next_states_traj), 0,1)
            cost_val = trajectory_cost_fn(self.cost_fn, states_traj, actions_traj, next_states_traj)
            #ipdb.set_trace() 
            return actions_traj[0, np.argmin(cost_val),:] 

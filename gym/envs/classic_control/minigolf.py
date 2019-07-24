from numbers import Number

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math as m
from scipy.stats import norm

"""
Minigolf task.

References
----------
  - Penner, A. R. "The physics of putting." Canadian Journal of Physics 80.2 (2002): 83-96.

"""

class MiniGolfConf(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second': 30
	}

	def __init__(self):
		self.horizon = 20
		self.gamma = 0.99

		self.min_pos = 1.0
		self.max_pos = 20.0
		self.min_action = 1e-5
		self.max_action = 2.0
		self.putter_length = 1.0 # [0.7:1.0]
		self.hole_size = 0.10 # [0.10:0.15]
		self.sigma_noise = 0.1
		self.ball_radius = 0.02135
		self.min_variance = 1e-2  # Minimum variance for computing the densities

		# gym attributes
		self.min_friction, self.max_friction = 0.02, 0.5 #0.065, 0.196
		self.putter_length_min, self.putter_length_max = 0.1, 100
		self.viewer = None
		low = np.array([self.min_pos, self.min_friction])
		high = np.array([self.max_pos, self.max_friction])
		self.action_space = spaces.Box(low=self.min_action,
									   high=self.max_action,
									   shape=(1,),dtype=float)
		self.observation_space = spaces.Box(low=low, high=high, dtype=float)

		self.observation_space_dim = 2
		self.action_space_dim = 1

		# initialize state
		self.seed()
		self.reset()

	def step(self, action, render=False):

		features = self.get_features()

		action = np.clip(action, self.min_action, self.max_action)

		#noise = 10
		#while abs(noise) > 1:
		#    noise = self.np_random.randn() * self.sigma_noise
		#u = action * self.putter_length * (1 + noise)
		noisy_action = action + np.random.randn() * self.sigma_noise
		noisy_action = np.clip(noisy_action, self.min_action, self.max_action)

		u = noisy_action * self.putter_length
		u = np.asscalar(u)

		v_min = np.sqrt(10 / 7 * self.friction * 9.81 * self.xn)
		v_max = np.sqrt((2*self.hole_size - self.ball_radius)**2*(9.81/(2*self.ball_radius)) + v_min**2)

		deceleration = 5 / 7 * self.friction * 9.81

		t = u / deceleration
		self.xn = self.xn - u * t + 0.5 * deceleration * t ** 2

		reward = 0
		done = True

		#print(u, v_min, v_max)

		if u < v_min:
			reward = -1
			done = False
		elif u > v_max:
			reward = -100

		self.state = np.array([self.xn, self.friction])

		return self.get_state(), float(reward), done, {'features': features}


	def reset(self, state=None):
		if state is None:
			self.xn, self.friction = self.np_random.uniform(low=[self.min_pos, self.min_friction],
														  high=[self.max_pos, self.max_friction])
			self.state = np.array([self.xn, self.friction])
		else:
			self.xn, self.friction = self.state
			self.state = np.array(state)

		return self.get_state()

	def get_state(self):
		return np.array(self.state)

	def get_features(self):
		return np.array([1., self.xn, np.sqrt(self.xn), self.friction, np.sqrt(self.friction), np.sqrt(self.xn * self.friction)])

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	@property
	def observation_space_size(self):
		return self.observation_space_dim

	@property
	def action_space_size(self):
		return self.action_space_dim

	def get_params_bounds(self) -> np.array:
		return np.array([[self.putter_length_min, self.putter_length_max]])

	def get_params(self) -> np.array:
		return self.putter_length

	def set_params(self, omega, *args):
		if np.isscalar(omega):
			self.putter_length = omega
		else:
			# check the number of elements inside array is exactly 1
			assert omega.size == 1
			omega = omega.flatten()
			self.putter_length = omega[0]

	def p_model(self,s_new,a,s,w_model):
		if np.size(a)==0:
			return 1
		dx = s_new[:,0]-s[:,0]
		friction = s[0,1] if len(s.shape)>1 else s[1]
		deceleration = 5 / 7 * friction * 9.81
		u = np.sqrt(-2*deceleration*dx)
		noisy_action = u/w_model
		return norm.pdf(noisy_action,loc=np.ravel(a),scale=self.sigma_noise)
	
	def grad_log_p_model(self,s_new,a,s,w_model):
		if np.size(a)==0:
			return 1
		dx = s_new[:,0]-s[:,0]
		friction = s[0,1] if len(s.shape)>1 else s[1]
		deceleration = 5 / 7 * friction * 9.81
		u = np.sqrt(-2*deceleration*dx)
		noisy_action = u/w_model
		return (noisy_action-np.ravel(a))/(self.sigma_noise**2)*u/(w_model**2)

if __name__ == '__main__':

	N = 10000
	mdp = MiniGolfConf()
	mdp.sigma_noise = 0.02
	mdp.putter_length = 1

	returns = []
	rew = 0.

	for l in np.arange(0, 100, 0.5):
		mdp.putter_length = l
		returns = []
		rew = 0.
		t = 0.
		s = mdp.reset()
		#print((np.sqrt(10./7*9.81)) / mdp.putter_length)
		print(np.sqrt(0.02 * 10. / 7 * 9.81) / mdp.putter_length)
		for i in range(N):
			t +=1
			#a = (np.sqrt(np.prod(s) * 10./7*9.81)) / mdp.putter_length + np.random.rand() * 0.02
			#a = (np.sqrt(s[0] * 0.02 * 10. / 7 * 9.81) + 2 * mdp.hole_size - mdp.ball_radius) / mdp.putter_length
			a = np.sqrt(s[0] * 0.02 * 10. / 7 * 9.81) / mdp.putter_length + np.random.rand() * 0.02
			s, r, done, info = mdp.step(a)
			rew += r

			if done or t == mdp.horizon:
				returns.append(rew)
				s = mdp.reset()
				rew = 0.
				t = 0

		if done:
			returns.append(r)
			s = mdp.reset()
			rew = 0.



		print(l, np.mean(returns))






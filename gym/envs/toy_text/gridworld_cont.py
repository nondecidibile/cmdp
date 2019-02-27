import gym
from gym import spaces
import numpy as np

class GridworldContEnv(gym.Env):

	def __init__(self):

		self.DIM = 5
		self.MAX_SPEED = 1
		self.END_DISTANCE = 1

		self.max_action = np.array([self.MAX_SPEED,self.MAX_SPEED])
		self.max_position = np.array([self.DIM/2,self.DIM/2,self.DIM/2,self.DIM/2])

		self.observation_space = spaces.Box(-self.max_position, self.max_position, dtype=np.float32)
		self.action_space = spaces.Box(-self.max_action, self.max_action, dtype=np.float32)


	def dist(self,p1,p2):
		return np.linalg.norm(p1-p2)


	def check_end(self,pos,dest):
		if self.dist(pos,dest) <= self.END_DISTANCE:
			return True
		else:
			return False


	def reset(self):
		x,y,xg,yg = np.random.uniform(self.observation_space.low,self.observation_space.high)
		pos = np.array([x,y],dtype=np.float32)
		dest = np.array([xg,yg],dtype=np.float32)
		while self.check_end(pos,dest):
			x,y,xg,yg = np.random.uniform(self.observation_space.low,self.observation_space.high)
			pos = np.array([x,y],dtype=np.float32)
			dest = np.array([xg,yg],dtype=np.float32)
		self.state = np.array([x,y,xg,yg],dtype=np.float32)
		return self.state
	

	def step(self, action):

		assert action.shape == self.action_space.shape

		x,y,xg,yg = self.state
		pos = np.array([x,y],dtype=np.float32)
		dest = np.array([xg,yg],dtype=np.float32)
		
		# clip action to max_speed
		action_norm = np.linalg.norm(action)
		action = action if action_norm<=self.MAX_SPEED else action*(self.MAX_SPEED/action_norm)

		x += action[0]
		y += action[1]

		newstate = np.array([x,y,xg,yg],dtype=np.float32)
		np.clip(newstate,-self.max_position,self.max_position,newstate)
		x,y,xg,yg = newstate
		self.state = newstate

		pos = np.array([x,y],dtype=np.float32)
		done = self.check_end(pos,dest)
		if done:
			reward = 0
		else:
			reward = -1
		
		return self.state, reward, done
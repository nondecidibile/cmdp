import gym
from gym import spaces
import numpy as np

class GridworldContNormalEnv(gym.Env):

	def __init__(self, mean=[0,0,0,0], var=[1,1,1,1]):

		self.DIM = 5
		self.MAX_SPEED = 1
		self.END_DISTANCE = 0.75

		self.max_action = np.array([self.MAX_SPEED,self.MAX_SPEED])
		self.max_position = np.array([self.DIM/2,self.DIM/2,self.DIM/2,self.DIM/2])

		self.observation_space = spaces.Box(-self.max_position, self.max_position, dtype=np.float32)
		self.action_space = spaces.Box(-self.max_action, self.max_action, dtype=np.float32)

		self.mean = mean
		self.var = var

	def dist(self,p1,p2):
		return np.linalg.norm(p1-p2)


	def check_end(self,pos,dest):
		if self.dist(pos,dest) <= self.END_DISTANCE:
			return True
		else:
			return False


	def reset(self):
		
		x,y = np.random.normal(loc=self.mean[0:2],scale=np.sqrt(self.var[0:2]))
		#if abs(x)>self.DIM/2 or abs(y)>self.DIM/2:
		#	x = np.random.uniform(-2.5,2.5)
		#	y = np.random.uniform(-2.5,2.5)

		xg,yg = np.random.normal(loc=self.mean[2:4],scale=np.sqrt(self.var[2:4]))
		#if abs(xg)>self.DIM/2 or abs(yg)>self.DIM/2:
		#	xg = np.random.uniform(-2.5,2.5)
		#	yg = np.random.uniform(-2.5,2.5)

		pos = np.array([x,y],dtype=np.float32)
		dest = np.array([xg,yg],dtype=np.float32)

		while self.check_end(pos,dest) or abs(x)>self.DIM/2 or abs(y)>self.DIM/2 or abs(xg)>self.DIM/2 or abs(yg)>self.DIM/2:
			x,y = np.random.normal(loc=self.mean[0:2],scale=np.sqrt(self.var[0:2]))
			#if abs(x)>self.DIM/2 or abs(y)>self.DIM/2:
			#	x = np.random.uniform(-2.5,2.5)
			#	y = np.random.uniform(-2.5,2.5)

			xg,yg = np.random.normal(loc=self.mean[2:4],scale=np.sqrt(self.var[2:4]))
			#if abs(xg)>self.DIM/2 or abs(yg)>self.DIM/2:
			#	xg = np.random.uniform(-2.5,2.5)
			#	yg = np.random.uniform(-2.5,2.5)

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
		#action_norm = np.linalg.norm(action)
		#action = action if action_norm<=self.MAX_SPEED else action*(self.MAX_SPEED/action_norm)

		dx = action[0]
		dy = action[1]

		if np.abs(dx) > self.MAX_SPEED:
			dx = dx/np.abs(dx)
		if np.abs(dy) > self.MAX_SPEED:
			dy = dy/np.abs(dy)

		x += dx
		y += dy

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
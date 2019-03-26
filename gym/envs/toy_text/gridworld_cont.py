import gym
from gym import spaces
import numpy as np

class GridworldContEnv(gym.Env):

	def __init__(self, changeProbs=[0.0,0.0], posBounds1=[-2.5,2.5,-2.5,2.5], goalBounds1=[-2.5,2.5,-2.5,2.5], posBounds2=None, goalBounds2=None):

		self.DIM = 5
		self.MAX_SPEED = 1
		self.END_DISTANCE = 0.75

		self.max_action = np.array([self.MAX_SPEED,self.MAX_SPEED])
		self.max_position = np.array([self.DIM/2,self.DIM/2,self.DIM/2,self.DIM/2])

		self.observation_space = spaces.Box(-self.max_position, self.max_position, dtype=np.float32)
		self.action_space = spaces.Box(-self.max_action, self.max_action, dtype=np.float32)

		self.changeProbs = np.clip(changeProbs,0,1)
		self.posBounds1 = posBounds1
		self.posBounds2 = posBounds2
		self.goalBounds1 = goalBounds1
		self.goalBounds2 = goalBounds2

	def dist(self,p1,p2):
		return np.linalg.norm(p1-p2)


	def check_end(self,pos,dest):
		if self.dist(pos,dest) <= self.END_DISTANCE:
			return True
		else:
			return False


	def reset(self):
		_pos = np.int(np.random.uniform() <= self.changeProbs[0])
		_goal = np.int(np.random.uniform() <= self.changeProbs[1])
		#x,y,xg,yg = np.random.uniform(self.observation_space.low,self.observation_space.high)
		if _pos==1:
			x = np.random.uniform(np.min(self.posBounds1[0:2]),np.max(self.posBounds1[0:2]))
			y = np.random.uniform(np.min(self.posBounds1[2:4]),np.max(self.posBounds1[2:4]))
		else:
			if self.posBounds2 is None:
				x = np.random.uniform(-self.DIM/2,self.DIM/2)
				y = np.random.uniform(-self.DIM/2,self.DIM/2)
			else:
				x = np.random.uniform(np.min(self.posBounds2[0:2]),np.max(self.posBounds2[0:2]))
				y = np.random.uniform(np.min(self.posBounds2[2:4]),np.max(self.posBounds2[2:4]))
		
		if _goal==1:
			xg = np.random.uniform(np.min(self.goalBounds1[0:2]),np.max(self.goalBounds1[0:2]))
			yg = np.random.uniform(np.min(self.goalBounds1[2:4]),np.max(self.goalBounds1[2:4]))
		else:
			if self.goalBounds2 is None:
				xg = np.random.uniform(-self.DIM/2,self.DIM/2)
				yg = np.random.uniform(-self.DIM/2,self.DIM/2)
			else:
				xg = np.random.uniform(np.min(self.goalBounds2[0:2]),np.max(self.goalBounds2[0:2]))
				yg = np.random.uniform(np.min(self.goalBounds2[2:4]),np.max(self.goalBounds2[2:4]))
		pos = np.array([x,y],dtype=np.float32)
		dest = np.array([xg,yg],dtype=np.float32)

		while self.check_end(pos,dest):
			if _pos==0:
				x = np.random.uniform(np.min(self.posBounds1[0:2]),np.max(self.posBounds1[0:2]))
				y = np.random.uniform(np.min(self.posBounds1[2:4]),np.max(self.posBounds1[2:4]))
			else:
				if self.posBounds2 is None:
					x = np.random.uniform(-self.DIM/2,self.DIM/2)
					y = np.random.uniform(-self.DIM/2,self.DIM/2)
				else:
					x = np.random.uniform(np.min(self.posBounds2[0:2]),np.max(self.posBounds2[0:2]))
					y = np.random.uniform(np.min(self.posBounds2[2:4]),np.max(self.posBounds2[2:4]))
			
			if _goal==0:
				xg = np.random.uniform(np.min(self.goalBounds1[0:2]),np.max(self.goalBounds1[0:2]))
				yg = np.random.uniform(np.min(self.goalBounds1[2:4]),np.max(self.goalBounds1[2:4]))
			else:
				if self.goalBounds2 is None:
					xg = np.random.uniform(-self.DIM/2,self.DIM/2)
					yg = np.random.uniform(-self.DIM/2,self.DIM/2)
				else:
					xg = np.random.uniform(np.min(self.goalBounds2[0:2]),np.max(self.goalBounds2[0:2]))
					yg = np.random.uniform(np.min(self.goalBounds2[2:4]),np.max(self.goalBounds2[2:4]))
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
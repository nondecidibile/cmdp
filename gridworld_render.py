import numpy as np
from gym.envs.toy_text import gridworld
from util.learner import *
from util.policy import *
from util.util import *

if __name__ == '__main__':
	
	mdp = gridworld.GridworldEnv()
	mdp.horizon = 50

	stateFeaturesMask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],dtype=bool)
	learner = GpomdpLearner(mdp,np.count_nonzero(stateFeaturesMask),4,gamma=0.99)
	learner.policy.params = np.load("params_ft.npy")

	collect_gridworld_episode(mdp,learner.policy,mdp.horizon,stateFeaturesMask,render=True)

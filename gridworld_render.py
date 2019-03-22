import numpy as np
from gym.envs.toy_text import gridworld
from util.learner import *
from util.policy_boltzmann import *
from util.util_gridworld import *

if __name__ == '__main__':
	
	mdp = gridworld.GridworldEnv(changeProb=0.5,targetCol=2)
	mdp.horizon = 50

	stateFeaturesMask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1],dtype=bool)
	policy = BoltzmannPolicy(np.count_nonzero(stateFeaturesMask),4)
	learner = GpomdpLearner(mdp,policy,gamma=0.98)
	learner.policy.params = np.load("params-14_modified.npy")

	collect_gridworld_episode(mdp,learner.policy,mdp.horizon,stateFeaturesMask,render=True)

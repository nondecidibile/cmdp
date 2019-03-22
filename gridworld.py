import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.policy_epsilon_boltzmann import *
from util.util_gridworld import *


mdp = gridworld.GridworldEnv()
mdp.horizon = 50


#
# Learning without some state features
#

sfMask = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1],dtype=bool) # state features mask
policy = EpsilonBoltzmannPolicy(np.count_nonzero(sfMask),4)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

learn(
	learner=learner,
	steps=250,
	nEpisodes=1000,
	sfmask=sfMask,
	loadFile="params8.npy",
	saveFile="params8.npy",
	autosave=True,
	plotGradient=False
)
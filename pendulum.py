import numpy as np
import scipy as sp
from gym.envs.classic_control import PendulumEnv
from util.policy_gaussian import *
from util.learner import *
from util.util_pendulum import *
import sys

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = PendulumEnv()
mdp.horizon = 400
agent_policy = GaussianPolicy(nStateFeatures=4,actionDim=1)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)


plearn(
	agent_learner,
	steps=1000,
	nEpisodes=25,
	learningRate=0.0001,
	loadFile=None,
	saveFile=None,
	autosave=True,
	plotGradient=True,
	printInfo=True
)
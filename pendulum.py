import numpy as np
import scipy as sp
from gym.envs.classic_control import PendulumEnv
from util.policy_nn import *
from util.learner import *
from util.util_pendulum import *
import sys

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = PendulumEnv()
mdp.horizon = 100
agent_policy = NeuralNetworkPolicy(nStateFeatures=3,actionDim=1,nHiddenNeurons=8,gamma=0.995)

#eps = collect_pendulum_episodes(mdp,agent_policy,10,mdp.horizon)
#agent_policy.optimize_gradient(eps,0.003)

plearn(
	mdp,
	agent_policy,
	steps=1000,
	nEpisodes=10,
	learningRate=0.0005,
	plotGradient=True,
	printInfo=True
)
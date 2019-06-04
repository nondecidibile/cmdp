import numpy as np
import scipy as sp
from gym.envs.classic_control import PendulumConfEnv
from util.policy_nn_gaussian import *
from util.learner_nn import *
from util.util_pendulum import *
import sys

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = PendulumConfEnv()
mdp.horizon = 100
agent_policy = nnGaussianPolicy(nStateFeatures=3,actionDim=1,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.7)
agent_learner = nnGpomdpLearner(mdp,agent_policy,gamma=0.995)

#eps = collect_pendulum_episodes(mdp,agent_policy,10,mdp.horizon)
#agent_policy.optimize_gradient(eps,0.003)

#collect_pendulum_episode(mdp,agent_policy,1000,render=True)

plearn(
	agent_learner,
	steps=10000,
	nEpisodes=100,
	learningRate=0.03,
	plotGradient=True,
	printInfo=True
)
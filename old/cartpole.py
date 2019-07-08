import numpy as np
import scipy as sp
from gym.envs.classic_control import CartPoleEnv
from util.policy_nn_boltzmann import *
from util.learner_nn import *
from util.util_cartpole import *
import sys

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = CartPoleEnv()
mdp.horizon = 200
agent_policy = nnBoltzmannPolicy(nStateFeatures=4,nActions=2,nHiddenNeurons=64,paramInitMaxVal=0.025)
agent_learner = nnGpomdpLearner(mdp,agent_policy,gamma=0.995)

#eps = collect_pendulum_episodes(mdp,agent_policy,10,mdp.horizon)
#agent_policy.optimize_gradient(eps,0.003)

ctlearn(
	agent_learner,
	steps=10000,
	nEpisodes=25,
	learningRate=0.003,
	plotGradient=True,
	printInfo=True
)
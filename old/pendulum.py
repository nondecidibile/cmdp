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
policyInstance = nnGaussianPolicy(nStateFeatures=3,actionDim=1,nHiddenNeurons=8,paramInitMaxVal=0.01,variance=0.7)
learner = nnGpomdpLearner(mdp,policyInstance,gamma=0.995)

plearn(
	learner,
	steps=50,
	nEpisodes=25,
	sfmask=[1,0,1], # cos(theta) [UP<->DOWN], sin(theta) [LEFT<->RIGHT], theta'
	learningRate=0.03,
	plotGradient=True,
	printInfo=True
)

agent_params = policyInstance.get_params()

eps = collect_pendulum_episodes(mdp,policyInstance,1000,mdp.horizon,showProgress=True)
'''estimate_params = policyInstance.estimate_params(
	eps,
	lr=0.001,
	nullFeature=0,
	batchSize=100,
	epsilon=0.001,
	minSteps=100,
	maxSteps=2500
)'''

print(agent_params)

sfMask = [1,1,1] # feature that we want to test
sfScores = lrTest(eps,policyInstance,sfMask,nsf=3,na=1,lr=0.01,batchSize=250,epsilon=0.001,maxSteps=1000)
print(sfMask)
print(sfScores)

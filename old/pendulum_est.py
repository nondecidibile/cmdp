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
policyInstance = nnGaussianPolicy(nStateFeatures=5,actionDim=1,nHiddenNeurons=8,paramInitMaxVal=0.01,variance=0.7)
learner = nnGpomdpLearner(mdp,policyInstance,gamma=0.995)

plearn(
	learner,
	steps=50,
	nEpisodes=50,
	sfmask=[1,0,0,1,1], # cos(theta) [UP<->DOWN], sin(theta) [LEFT<->RIGHT], UP/DOWN, LEFT/RIGHT, theta'
	learningRate=0.03,
	plotGradient=True,
	printInfo=True
)

agent_params = policyInstance.get_params()
#agent_params = np.load("params.npy")
#policyInstance.set_params(agent_params)

eps = collect_pendulum_episodes(mdp,policyInstance,10000,mdp.horizon,showProgress=True)
np.save("eps_s.npy",eps["s"])
np.save("eps_a.npy",eps["a"])
np.save("eps_r.npy",eps["r"])
np.save("eps_len.npy",eps["len"])
'''
eps = {
	"s": np.load("eps_s.npy"),
	"a": np.load("eps_a.npy"),
	"r": np.load("eps_r.npy"),
	"len": np.load("eps_len.npy")
}
estimated_params = policyInstance.estimate_params(
	eps,
	lr=0.03,
	nullFeature=None,
	batchSize=10000,
	epsilon=0.0001,
	minSteps=100,
	maxSteps=2500
)

ll = policyInstance.getLogLikelihood(eps)

estimated_params_0 = policyInstance.estimate_params(
	eps,
	lr=0.03,
	params0=None,
	nullFeature=0,
	batchSize=10000,
	epsilon=0.0001,
	minSteps=100,
	maxSteps=2500
)

ll_h0 = policyInstance.getLogLikelihood(eps)

print("all features:",ll)
print("without feature 0:",ll_h0)
lr_lambda = -2*(ll_h0 - ll)
print(lr_lambda)
'''

sfMask = [1,1,1,1,1] # feature that we want to test
sfScores = lrTest(eps,policyInstance,sfMask,nsf=5,na=1,lr=0.03,batchSize=10000,epsilon=0.0001,maxSteps=2500)
print(sfMask)
print(sfScores)

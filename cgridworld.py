import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[30:50] = False

policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

clearn(
	learner,
	steps=0,
	nEpisodes=500,
	sfmask=sfMask,
	loadFile="cparams30.npy",
	saveFile=None,#"cparams30.npy",
	autosave=True,
	plotGradient=True
)


#
# Gradient estimation
#

N = 1000

estimated_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
estimated_learner = GpomdpLearner(mdp,estimated_policy,gamma=0.98)

# Take learner params but with 0s in the features he doesn't have
for a in range(learner.policy.paramsShape[0]):
	for sf in range(estimated_learner.policy.paramsShape[1]):
		if(sf < learner.policy.paramsShape[1]):
			estimated_learner.policy.params[a,sf] = learner.policy.params[a,sf]
		else:
			estimated_learner.policy.params[a,sf] = 0

# Estimate gradient with N trajectories
eps = collect_cgridworld_episodes(mdp,learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
gradient,gradient_var = estimated_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

print(gradient)


####### Chebyshev
Rmax = 1
v = 1/0.25 # phi_max^2 / sigma^2
T = 50
gamma = 0.98
delta = 0.1
variance = np.square(Rmax)*v/np.power(1-gamma,3)*(1-np.power(gamma,T))*(T*np.power(gamma,T+1)-(T+1)*np.power(gamma,T)+1)
c_epsilon = np.sqrt(variance/(N*delta))
print("\nChebyshev: ",c_epsilon)
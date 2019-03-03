import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

clearn(
	learner,
	steps=500,
	nEpisodes=500,
	loadFile=None, #"cparams.npy",
	saveFile=None,#"cparams.npy",
	autosave=True,
	plotGradient=False
)


#
# Render some episodes
#
collect_cgridworld_episodes(mdp,policy,20,mdp.horizon,render=True)


#
# Gradient estimation
#
'''
N = 2000

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
eps = collect_cgridworld_episodes(mdp,learner.policy,N,mdp.horizon,showProgress=True)
gradient,gradient_var = estimated_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

print(gradient)
'''
import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *


mdp = gridworld.GridworldEnv()
mdp.horizon = 50


#
# Learning without some state features
#

sfMask = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1],dtype=bool) # state features mask
agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

learn(
	learner=agent_learner,
	steps=0,
	nEpisodes=1000,
	sfmask=sfMask,
	loadFile="params8.npy",
	saveFile=None,
	autosave=True,
	plotGradient=False
)

N = 10000
eps = collect_gridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

super_policy = BoltzmannPolicy(17,4)

optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=0.3)
params = super_policy.estimate_params(eps,optimizer,setToZero=None,epsilon=0.01,minSteps=100,maxSteps=250)

optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=0.3)
params_0 = super_policy.estimate_params(eps,optimizer,setToZero=0,epsilon=0.01,minSteps=100,maxSteps=250)

optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=0.3)
params_8 = super_policy.estimate_params(eps,optimizer,setToZero=8,epsilon=0.01,minSteps=100,maxSteps=250)

ll = super_policy.getLogLikelihood(eps,params)
print(ll)
ll_0 = super_policy.getLogLikelihood(eps,params_0)
print(ll_0)
ll_8 = super_policy.getLogLikelihood(eps,params_8)
print(ll_8)
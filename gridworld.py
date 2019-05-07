import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *

w_row = [5,3,2,1,0.5]
w_col = [5,3,2,1,0.5]
w_grow = [0.5,1,2,3,5]
w_gcol = [0.5,1,2,3,5]

mdp = gridworld.GridworldEnv(w_row,w_col,w_grow,w_gcol)
mdp.horizon = 50


#
# Learning without some state features
#

sfMask = np.array([1,1,1,1, 1,1,1,1, 0,1,1,1, 1,1,1,1, 1],dtype=bool) # state features mask
agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

learn(
	learner=agent_learner,
	steps=100,
	nEpisodes=250,
	sfmask=sfMask,
	loadFile=None,
	saveFile=None,
	autosave=True,
	plotGradient=False
)

N = 1000
eps = collect_gridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

super_policy = BoltzmannPolicy(17,4)

optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=0.3)
params = super_policy.estimate_params(eps,optimizer,setToZero=None,epsilon=0.001,minSteps=100,maxSteps=1000)
ll = super_policy.getLogLikelihood(eps,params)

ll_h0 = np.zeros(shape=(16),dtype=np.float32)
for param in range(16):
	optimizer = AdamOptimizer(super_policy.paramsShape,learning_rate=0.3)
	params_h0 = super_policy.estimate_params(eps,optimizer,setToZero=param,epsilon=0.001,minSteps=100,maxSteps=1000)
	ll_h0[param] = super_policy.getLogLikelihood(eps,params_h0)

print(ll)
print(ll_h0)
for param in range(16):
	lr_lambda = -2*(ll_h0[param] - ll)
	print(param,"-",lr_lambda)
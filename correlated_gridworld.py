import numpy as np
import scipy as sp
from gym.envs.toy_text import gridworld
from util.util_gridworld import *
from util.policy_boltzmann import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = gridworld.GridworldEnv()
mdp.horizon = 50

sfMask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,1],dtype=bool) # state features mask
agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

M = np.array([
	[.5, .2, .3, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], #0
	[.1, .5, .4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], #1
	[.3, .3, .4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], #2
	[0., .4, 0., .6, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., .8, .2, 0., 0., 0., 0.], #11 = 0.8*#11*0.2*#12
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., .3, .7, 0., 0., 0., 0.], #12 = 0.3*#11+0.7*#12
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
	[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
])


learn(
	agent_learner,
	steps=0,
	nEpisodes=250,
	transfMatrix=M,
	sfmask=sfMask,
	adamOptimizer=True,
	learningRate=0.03,
	loadFile="correlated_gridworld.npy",
	saveFile=None,
	autosave=True,
	plotGradient=False
)

super_policy = BoltzmannPolicy(sfMask.size,4)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

N = 5000
eps = collect_gridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,M,sfMask,exportAllStateFeatures=True,showProgress=True)
optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
estimated_params = super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.001,minSteps=150,maxSteps=300)

print(estimated_params)
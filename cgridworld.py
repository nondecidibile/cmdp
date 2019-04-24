import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[40:50] = False

agent_policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

clearn(
	agent_learner,
	steps=0,
	nEpisodes=500,
	sfmask=sfMask,
	loadFile="cparams40.npy",
	saveFile=None,
	autosave=True,
	plotGradient=False
)

sfTestMask = np.ones(shape=50,dtype=np.bool) # State features not rejected

N = 1000
eps = collect_cgridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

lr_lambda = lrTest(eps,sfTestMask,lr=0.3,epsilon=0.001,maxSteps=1000)
print(lr_lambda)
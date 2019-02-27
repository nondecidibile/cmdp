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
	steps=0,
	nEpisodes=250,
	loadFile="cparams.npy",
	saveFile=None,#"cparams.npy",
	autosave=False,
	plotGradient=True
)

collect_cgridworld_episode(mdp,policy,mdp.horizon,render=True)

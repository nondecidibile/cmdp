import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[25:50] = False

policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

learner.policy.params = np.load("cparams25.npy")
collect_cgridworld_episode(mdp,learner.policy,mdp.horizon,sfMask,render=True)
import numpy as np
from gym.envs.toy_text import gridworld_cont, gridworld_cont_normal
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mean_model1 = [-2,-2,2,2]
var_model1 = [1,1,1,1]
mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=mean_model1,var=var_model1)
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
#sfMask[40:50] = False

policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

learner.policy.params = np.load("cgnorm1.npy")
collect_cgridworld_episodes(mdp,learner.policy,1,mdp.horizon,sfMask,render=True)
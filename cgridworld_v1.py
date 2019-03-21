import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[30:50] = False

# AGENT SPACE
agent_policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

clearn(
    agent_learner,
    steps=0,
    nEpisodes=0,
    sfmask=sfMask,
    adamOptimizer=True,
    learningRate=0.1,
    loadFile="cparams30.npy",
    saveFile=None,
    autosave=True,
    plotGradient=False
)

# SUPERVISOR SPACE
super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

# collect episodes
N = 2500
eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)

# estimate params
optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
params = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.0001,minSteps=100,maxSteps=500))
print(params,"\n")

print(agent_policy.params,"\n")
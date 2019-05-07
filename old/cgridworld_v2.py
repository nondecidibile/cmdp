import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

mdp = gridworld_cont.GridworldContEnv()
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[30:50] = False

# AGENT SPACE
agent_policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

# SUPERVISOR SPACE
super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)


for i in range(100):

    # agent parameters before the agent's update
    for a in range(agent_learner.policy.paramsShape[0]):
        for sf in range(super_learner.policy.paramsShape[1]):
            if(sf < agent_learner.policy.paramsShape[1]):
                super_learner.policy.params[a,sf] = agent_learner.policy.params[a,sf]
            else:
                super_learner.policy.params[a,sf] = 0
    params_before = np.copy(super_learner.policy.params)
    # estimated gradient before the agent's update
    N = 2000
    eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    gradient_before,_ = super_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

    # one step of agent learning
    clearn(
        agent_learner,
        steps=1,
        nEpisodes=2000,
        sfmask=sfMask,
        adamOptimizer=False,
        learningRate=0.03,
        loadFile=None,
        saveFile=None,
        autosave=True,
        plotGradient=False
    )

    # agent parameters after the agent's update
    for a in range(agent_learner.policy.paramsShape[0]):
        for sf in range(super_learner.policy.paramsShape[1]):
            if(sf < agent_learner.policy.paramsShape[1]):
                super_learner.policy.params[a,sf] = agent_learner.policy.params[a,sf]
            else:
                super_learner.policy.params[a,sf] = 0
    params_after = np.copy(super_learner.policy.params)

    estimated_alpha = np.divide(params_after-params_before,gradient_before)
    print(estimated_alpha)
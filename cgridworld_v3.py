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


clearn(
    agent_learner,
    steps=15,
    nEpisodes=200,
    sfmask=sfMask,
    adamOptimizer=False,
    learningRate=0.1,
    loadFile=None,
    saveFile=None,
    autosave=True,
    plotGradient=False
)


# SUPERVISOR SPACE
super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)


for i in range(100):

    N = 2000
    eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters before the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.1)
    params_before = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.01,minSteps=250,maxSteps=500))
    # estimated gradient before the agent's update
    gradient_before,_ = super_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

    # one step of agent learning
    clearn(
        agent_learner,
        steps=1,
        nEpisodes=2000,
        sfmask=sfMask,
        adamOptimizer=False,
        learningRate=0.1,
        loadFile=None,
        saveFile=None,
        autosave=True,
        plotGradient=False
    )

    N = 2000
    eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters after the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.1)
    params_after = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.01,minSteps=250,maxSteps=500))

    estimated_alpha = np.divide(params_after-params_before,gradient_before)
    print("alpha = \n",estimated_alpha,"\n\n")

    print("agent params = \n",agent_learner.policy.params,"\n")
    print("super params = \n",super_learner.policy.params,"\n")
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
    steps=25,
    nEpisodes=100,
    sfmask=sfMask,
    adamOptimizer=True,
    learningRate=0.1,
    loadFile=None,#"cparams30.npy",
    saveFile=None,
    autosave=True,
    plotGradient=False
)


# SUPERVISOR SPACE
super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)


for i in range(100):

    N = 5000
    eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters before the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
    params_before = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.0001,minSteps=100,maxSteps=1000))
    # estimated gradient before the agent's update
    gradient_before,_ = super_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

    # one step of agent learning
    clearn(
        agent_learner,
        steps=1,
        nEpisodes=5000,
        sfmask=sfMask,
        adamOptimizer=False,
        learningRate=0.1,
        loadFile=None,
        saveFile=None,
        autosave=True,
        plotGradient=False
    )

    N = 5000
    eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters after the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
    params_after = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.0001,minSteps=100,maxSteps=500))

    estimated_alpha = np.divide(params_after-params_before,gradient_before)
    print("alpha = \n",estimated_alpha,"\n\n")
    print("params diff = \n",params_after-params_before,"\n\n")
    print("gradient = \n",gradient_before,"\n\n")

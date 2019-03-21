import numpy as np
from gym.envs.toy_text import gridworld
from util.util_gridworld import *
from util.policy_boltzmann import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = gridworld.GridworldEnv()
mdp.horizon = 50

sfMask = np.ones(shape=16,dtype=bool) # state features mask
sfMask[8:16] = False

# AGENT SPACE
agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)


learn(
    agent_learner,
    steps=30,
    nEpisodes=250,
    sfmask=sfMask,
    adamOptimizer=True,
    learningRate=0.3,
    loadFile=None,#"params8.npy",
    saveFile=None,
    autosave=True,
    plotGradient=False
)


# SUPERVISOR SPACE
super_policy = BoltzmannPolicy(16,4)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)


for i in range(100):

    N = 10000
    eps = collect_gridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters before the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
    params_before = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.001,minSteps=100,maxSteps=500))
    # estimated gradient before the agent's update
    gradient_before,_ = super_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)

    # one step of agent learning
    learn(
        agent_learner,
        steps=1,
        nEpisodes=10000,
        sfmask=sfMask,
        adamOptimizer=False,
        learningRate=0.03,
        loadFile=None,
        saveFile=None,
        autosave=True,
        plotGradient=False
    )

    N = 10000
    eps = collect_gridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)
    # estimated agent parameters after the agent's update
    optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
    params_after = np.copy(super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.001,minSteps=100,maxSteps=500))

    estimated_alpha = np.divide(params_after-params_before,gradient_before)
    print("alpha =\n",estimated_alpha,"\n\n")
    print("params diff =\n",params_after-params_before,"\n\n")
    print("gradient before =\n",gradient_before,"\n\n")
    print("AGENT params =\n",agent_learner.policy.params,"\n\n")
    print("SUPER params =\n",params_after,"\n\n")

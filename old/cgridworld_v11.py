import numpy as np
from gym.envs.toy_text import gridworld_cont
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

mdp = gridworld_cont.GridworldContEnv(changeProbs=[0.5,1],posBounds1=[-1,-1,2,2],posBounds2=[-1,-1,-1,-1], goalBounds1=[-2.5,2.5,0,2.5])
mdp.horizon = 50

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[40:50] = False

# AGENT SPACE
agent_policy = GaussianPolicy(np.count_nonzero(sfMask),2)
agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

clearn(
	agent_learner,
	steps=0,
	nEpisodes=500,
	sfmask=sfMask,
	adamOptimizer=True,
	learningRate=0.03,
	loadFile=None,
	saveFile=None,
	autosave=True,
	plotGradient=False
)


# SUPERVISOR SPACE
super_policy = GaussianPolicy(50,2)
super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)


# collect episodes with agent's policy
N = 10000
eps = collect_cgridworld_episodes(mdp,agent_learner.policy,N,mdp.horizon,sfMask,exportAllStateFeatures=True,showProgress=True)

# estimated agent parameters with MLE
optimizer = AdamOptimizer(super_learner.policy.paramsShape,learning_rate=0.3)
params = super_learner.policy.estimate_params(eps,optimizer,None,epsilon=0.001,minSteps=150,maxSteps=500)
print(params)

#fisherInfo = super_learner.getFisherInformation(eps)
fisherInfo = super_learner.policy.getAnalyticalFisherInformation(eps)
invFisherDiag = np.diagonal(np.linalg.inv(fisherInfo))

var_terms = np.ravel(np.sqrt(invFisherDiag))
estimator_terms = super_learner.policy.params * np.sqrt(N) / var_terms

z_0_90 = 1.65
z_0_95 = 1.96
z_0_99 = 2.58

for a,tt in enumerate(estimator_terms):
	print("\n",end='')
	for s,t in enumerate(tt):
		rejected90 = 0
		rejected95 = 0
		rejected99 = 0
		p = np.abs(t)
		if not(p+z_0_90>=0 and p-z_0_90<=0):
			rejected90 = 1
		if not(p+z_0_95>=0 and p-z_0_95<=0):
			rejected95 = 1
		if not(p+z_0_99>=0 and p-z_0_99<=0):
			rejected99 = 1
		print("s =",s,"| a =",a,
		"|| p = ",format(p,".5f"),
		"| rejected (@90%, 95%, 99%) =",rejected90,rejected95,rejected99)
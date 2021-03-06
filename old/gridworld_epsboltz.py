import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.policy_epsilon_boltzmann import *
from util.util_gridworld import *


mdp = gridworld.GridworldEnv()
mdp.horizon = 50


#
# Learning without some state features
#

sfMask = np.array([1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],dtype=bool) # state features mask
policy = EpsilonBoltzmannPolicy(np.count_nonzero(sfMask),4)
learner = GpomdpLearner(mdp,policy,gamma=0.98)

learn(
	learner=learner,
	steps=0,
	nEpisodes=500,
	sfmask=sfMask,
	loadFile="params9.npy",
	saveFile=None,
	plotGradient=False
)


#
# Policy & gradient estimation
#

N = 20000
eps = collect_gridworld_episodes(mdp,learner.policy,N,mdp.horizon,sfMask,showProgress=True)

estimated_policy = BoltzmannPolicy(16,4)
estimated_learner = GpomdpLearner(mdp,estimated_policy,gamma=0.98)

# Load already estimated parameters
#estimated_learner.policy.params = np.load("params8est.npy")

# Take learner params but with 0s in the features he doesn't have
for a in range(learner.policy.paramsShape[0]):
	for sf in range(estimated_learner.policy.paramsShape[1]):
		if(sf < learner.policy.paramsShape[1]):
			estimated_learner.policy.params[a,sf] = learner.policy.params[a,sf]
		else:
			estimated_learner.policy.params[a,sf] = 0

# Find params with Maximum Likelihood
#find_params_ml(estimated_learner,eps,saveFile=None)

#gradient = np.load("gradient8.npy")
#gradient_var = np.load("gradient8var.npy")
gradient,gradient_var = estimated_learner.estimate_gradient(eps,getSampleVariance=True,showProgress=True)
#np.save("gradient8.npy",gradient)
#np.save("gradient8var.npy",gradient_var)


#
# Hypothesis test
#

# Confidence level
delta = 0.99


####### Hoeffding

# Discount factor
gamma = learner.gamma
# Episode max length
T = 50
# size of the interval in which Xi takes its values
pq = 2*(gamma*(1-np.power(gamma,T-1))/np.power(1-gamma,2)-(T-1)*np.power(gamma,T)/(1-gamma))
# size of the confidence interval
h_epsilon = pq*np.sqrt(np.log(2/delta)/(2*N))
print("\nHoeffding: ",h_epsilon)


####### Empirical Bernstein

eb_epsilon = (np.sqrt(2*gradient_var*np.log(4/delta)/N)+pq/2*(7*np.log(4/delta)/(3*(N-1))))
print("\nEmpirical Bernstein: ",eb_epsilon)


####### Chebyshev

Rmax = 1 # max l1-norm reward
v = 0.25 # second moment of the log gradient
variance = np.square(Rmax)*v/np.power(1-gamma,3)*(1-np.power(gamma,T))*(T*np.power(gamma,T+1)-(T+1)*np.power(gamma,T)+1)
c_epsilon = np.sqrt(variance/(N*delta))
print("\nChebyshev: ",c_epsilon)


####### Bernstein

m = 1
v_sup = 1/4
b = 1/N * (Rmax * m / (1-gamma)) * (gamma*(1-np.power(gamma,T-1))/(1-gamma) - (T-1)*np.power(gamma,T))
v = 1/N * np.square(Rmax) * v_sup * (1 - np.power(gamma,T)) / np.power(1-gamma,3) * (T*np.power(gamma,T+1) - (T+1)*np.power(gamma,T) + 1)
b_epsilon = 1/3 * b * np.log(2/delta) + np.sqrt(1/9*np.square(b*np.log(2/delta))+2*v*np.log(2/delta))
print("\nBernstein: ",b_epsilon,"\n")


#
# Test state features
#

rejected_sf = np.zeros(shape=gradient[0].shape)

for a,gg in enumerate(gradient):
	print("\n",end='')
	for s,g in enumerate(gg):
		rejected = 0
		if not(g+eb_epsilon[a,s]>=0 and g-eb_epsilon[a,s]<=0):
			rejected = 1
			rejected_sf[s] += 1
		var = gradient_var[a,s]
		print("s =",s,"| a =",a,
		"|| g = ",format(g,".5f"),
		"| var = ",format(var,".5f"),
		"| rejected =",rejected)

print("\nRejected state features:\n",rejected_sf)
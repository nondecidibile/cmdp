import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy import *
from util.util import *


mdp = gridworld.GridworldEnv()
mdp.horizon = 50

#
# Learning without some state features
#

stateFeaturesMask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],dtype=bool)

learner = GpomdpLearner(mdp,np.count_nonzero(stateFeaturesMask),4,gamma=0.98)
learner.policy.params = np.load("params_ft.npy")

'''
optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=0.1, beta1=0.9, beta2=0.99)
for step in range(1501):

	n_episodes = 50
	eps, mean_length = collect_gridworld_episodes(mdp,learner.policy,n_episodes,mdp.horizon,stateFeaturesMask,False)

	gradient = learner.estimate_gradient(eps)
	update_step = optimizer.step(gradient)

	learner.policy.params += update_step

	print("Step: "+str(step))
	avg_mean_length = mean_length if step==0 else avg_mean_length*0.9+mean_length*0.1
	print("Average mean length:      "+str(np.round(avg_mean_length,3)))
	avg_gradient = gradient/learner.policy.nFeatures if step==0 else avg_gradient*0.9+gradient/learner.policy.nFeatures*0.1
	print("Average gradient /param:  "+str(np.round(np.linalg.norm(np.ravel(np.asarray(avg_gradient)),1),5)))
	print("Update step /param:       "+str(np.round(np.linalg.norm(np.ravel(np.asarray(update_step/learner.policy.nFeatures)),1),5))+"\n")

np.save("params_ft",learner.policy.params)
'''

#
# Policy & gradient estimation
#

N = 10000

estimated_learner = GpomdpLearner(mdp,16,4,gamma=0.98)
#estimated_learner.policy.params = np.load("params_ft_estimated.npy")

# Take learner params but with 0s in the features he doesn't have
for a in range(learner.policy.paramsShape[0]):
	for sf in range(estimated_learner.policy.paramsShape[1]):
		if(sf < learner.policy.paramsShape[1]):
			estimated_learner.policy.params[a,sf] = learner.policy.params[a,sf]
		else:
			estimated_learner.policy.params[a,sf] = 0



eps,_ = collect_gridworld_episodes(mdp,learner.policy,N,mdp.horizon,stateFeaturesMask)


'''
print("Estimating policy parameters...\n")
optimizer = AdamOptimizer(estimated_learner.policy.paramsShape,learning_rate=2.5)
estimated_params = estimated_learner.policy.estimate_params(eps,optimizer,estimated_learner.policy.params,epsilon=0.01,minSteps=50,maxSteps=200)
	
_,est_ml = collect_gridworld_episodes(mdp,estimated_learner.policy,500,mdp.horizon)
print("Estimated avg mean length: ",est_ml,"\n")
p_abs_mean = np.mean(np.abs(estimated_learner.policy.params),axis=0)
print("Parameters absolute average between actions\n",p_abs_mean,"\n")

np.save("params_ft_estimated.npy",estimated_params)
'''


print("Estimating policy gradient...")
gradient,gradient_var = estimated_learner.estimate_gradient(eps,getSampleVariance=True)
#print("Gradient\n",gradient,"\n")
for a,gg in enumerate(gradient):
	print("\n",end='')
	for s,g in enumerate(gg):
		var = gradient_var[a,s]
		print("s =",s,"| a =",a,
		"|| g = ",format(g,".5f"),
		"| var = ",format(var,".5f"))
#g_abs_mean = np.mean(np.abs(gradient),axis=0)
#print("Gradient absolute average between actions\n",g_abs_mean,"\n")


#
# Hypothesis test
#

# Number of episodes
N = len(eps)
# Confidence level
delta = 0.05


####### Hoeffding

# Discount factor
gamma = learner.gamma
# Episode max length
T = 50
# size of the interval in which Xi takes its values
pq = 2*(gamma*(1-np.power(gamma,T-1))/np.power(1-gamma,2)-(T-1)*np.power(gamma,T)/(1-gamma))
# size of the confidence interval
epsilon = pq*np.sqrt(np.log(2/delta)/(2*N))
print("\nHoeffding: ",epsilon)


####### Empirical Bernstein

epsilon = (np.sqrt(2*gradient_var*np.log(4/delta)/N)+pq/2*(7*np.log(4/delta)/(3*(N-1))))
print("\nEmpirical Bernstein: ",epsilon)


####### Chebyshev

sum_gamma = 0
for j in range(T):
	sum_gamma += np.square(np.power(gamma,j)-np.power(gamma,T))

Rmax = 1 # max l1-norm reward
M = 0.25 # second moment of the log gradient
variance = Rmax*M/(1-gamma)*sum_gamma # variance

epsilon = np.sqrt(variance/(N*delta))
print("\nChebyshev: ",epsilon)
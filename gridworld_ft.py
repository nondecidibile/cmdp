import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy import *
from util.util import *

if __name__ == '__main__':

	mdp = gridworld.GridworldEnv()
	mdp.horizon = 30

	#
	# Learning without some state features
	#

	stateFeaturesMask = np.array([1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],dtype=bool)

	learner = GpomdpLearner(mdp,np.count_nonzero(stateFeaturesMask),4)
	#learner.policy.params = np.load("params_ft.npy")
	optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=0.1, beta1=0.9, beta2=0.99)
	
	
	for step in range(1201):

		n_episodes = 40
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
	
	#np.save("params_ft",learner.policy.params)
	

	#
	# Policy & gradient estimation
	#

	estimated_learner = GpomdpLearner(mdp,16,4)
	#estimated_learner.policy.params = np.load("params_ft_estimated.npy")
	eps,_ = collect_gridworld_episodes(mdp,learner.policy,1000,mdp.horizon,stateFeaturesMask)
	
	
	print("Estimating policy parameters...\n")
	
	optimizer = AdamOptimizer(estimated_learner.policy.paramsShape,learning_rate=2.5)

	estimated_params = estimated_learner.policy.estimate_params(eps,optimizer,estimated_learner.policy.params,epsilon=0.01,minSteps=50,maxSteps=200)
		
	_,est_ml = collect_gridworld_episodes(mdp,estimated_learner.policy,500,mdp.horizon)
	print("Estimated avg mean length: ",est_ml,"\n")

	p_abs_mean = np.mean(np.abs(estimated_learner.policy.params),axis=0)
	print("Parameters absolute average between actions\n",p_abs_mean,"\n")


	print("Estimating policy gradient...\n")
	gradient = estimated_learner.estimate_gradient(eps)
	#print("Gradient\n",gradient,"\n")
	g_abs_mean = np.mean(np.abs(gradient),axis=0)
	print("Gradient absolute average between actions\n",g_abs_mean,"\n")
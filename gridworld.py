import numpy as np
import sys
from gym.envs.toy_text import gridworld
from time import sleep
from util.learner import *
from util.optimizer import *
from util.policy import *
from util.util import *

if __name__ == '__main__':
	
	mdp = gridworld.GridworldEnv()
	mdp.horizon = 30

	#
	# Learning
	#

	print("Learning a policy...\n")

	learner = GpomdpLearner(mdp,16,4)
	optimizer = AdamOptimizer(learner.policy.paramsShape, learning_rate=0.1)
	
	for step in range(1001):

		n_episodes = 50
		eps, mean_length = collect_gridworld_episodes(mdp,learner.policy,n_episodes,mdp.horizon)

		gradient = learner.estimate_gradient(eps)
		update_step = optimizer.step(gradient)

		learner.policy.params += update_step

		print("Step: "+str(step))
		avg_mean_length = mean_length if step==0 else avg_mean_length*0.95+mean_length*0.05
		print("Average mean length:      "+str(np.round(avg_mean_length,3)))
		avg_gradient = gradient if step==0 else avg_gradient*0.95+gradient/learner.policy.nFeatures*0.05
		print("Average gradient /param:  "+str(np.round(np.linalg.norm(np.ravel(np.asarray(avg_gradient)),1),5)))
		print("Update step /param:       "+str(np.round(np.linalg.norm(np.ravel(np.asarray(update_step/learner.policy.nFeatures)),1),5))+"\n")

	#
	# Parameters estimation
	#

	print("Estimating policy parameters...\n")

	eps,_ = collect_gridworld_episodes(mdp,learner.policy,100,mdp.horizon)

	policy_estimator = BoltzmannPolicy(16,4)
	optimizer = AdamOptimizer(policy_estimator.paramsShape,learning_rate=0.25)

	params = policy_estimator.estimate_params(eps,optimizer,epsilon=0.1,maxSteps=150)

	print("L2 difference between policies:   "+
		str(np.round(np.linalg.norm(np.squeeze(np.asarray(learner.policy.params-params)),2),3)))
	
	_,real_ml = collect_gridworld_episodes(mdp,learner.policy,500,mdp.horizon)
	_,est_ml = collect_gridworld_episodes(mdp,policy_estimator,500,mdp.horizon)
	print("Difference between mean lengths:  "+str(np.round(abs(real_ml-est_ml),3)))
	

	

	

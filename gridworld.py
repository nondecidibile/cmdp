import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *

w_row = [5,1,1,1,1]
w_col = [5,1,1,1,1]
w_grow = [5,1,1,1,1]
w_gcol = [1,1,1,1,5]
model = np.array([w_row,w_col,w_grow,w_gcol],dtype=np.float32)

mdp = gridworld.GridworldEnv(model)
mdp.horizon = 50

sfMask = np.array([1,1,1,1, 1,1,1,1, 1,1,1,1, 0,0,0,0, 1],dtype=bool) # Agent features

sfTestMask = np.ones(shape=16,dtype=np.bool) # Features not rejected

#
# Cycle ENVIRONMENT CONFIGURATION
#
for conf_index in range(1000):

	print("Using MDP ",model)
	mdp = gridworld.GridworldEnv(model)
	mdp.horizon = 50
	saveStateImage("stateImage"+str(conf_index)+"A.png",mdp,sfTestMask)

	agent_policy = BoltzmannPolicy(np.count_nonzero(sfMask),4)
	agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

	learn(
		learner=agent_learner,
		steps=200,
		nEpisodes=250,
		sfmask=sfMask,
		loadFile=None,
		saveFile=None,
		autosave=True,
		plotGradient=False,
		printInfo=False
	)

	super_policy = BoltzmannPolicy(nStateFeatures=17,nActions=4)
	super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

	N = 100
	eps = collect_gridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps,sfTestMask)
	saveStateImage("stateImage"+str(conf_index)+"B.png",mdp,sfTestMask)
	print(sfTestMask)
	print(lr_lambda)

	# Choose next parameter for model optimization
	sfTestMaskIndices = np.where(sfTestMask == True)
	sfGradientMask = np.zeros(shape=17,dtype=np.bool)
	sfGradientMask[sfTestMaskIndices[0][0]] = True
	print("Configuring model to test parameter",sfTestMaskIndices[0][0])

	model2 = model.copy()
	model2[0,0] += 0.01
	model2[1,1] -= 0.01
	model2[2,2] += 0.01
	model2[3,3] -= 0.01

	modelOptimizer = AdamOptimizer(shape=(4,5),learning_rate=0.01)
	for _i in range(150):
		modelGradient = getModelGradient(super_learner,eps,sfGradientMask,model,model2)
		model2 += modelOptimizer.step(modelGradient)
		#print("MODEL:",model2,"\n")

	model = model2.copy()
import numpy as np
import scipy as sp
from gym.envs.toy_text import gridworld_cont
from gym.envs.toy_text import gridworld_cont_normal
from util.util_cgridworld import *
from util.policy_gaussian import *
from util.learner import *
import sys

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

meanModel = [-2,2,2,-2]
varModel = [0.25,0.25,0.5,0.5]

sfMask = np.ones(shape=50,dtype=bool) # state features mask
sfMask[40:50] = False

sfTestMask = np.ones(shape=50,dtype=np.bool) # State features not rejected
#sfTestMask[0:25] = False

'''
sfBestModels = []
for sf in range(50):
	sfBestModels.append([meanModel.copy(),0])
'''

#
# Cycle ENVIRONMENT CONFIGURATION
#
for conf_index in range(1000):

	print("Using MDP with mean =",meanModel)
	mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=meanModel,var=varModel)
	mdp.horizon = 50
	#mdp_uniform = gridworld_cont.GridworldContEnv()
	#mdp_uniform.horizon = 50
	saveStateImage("stateImage"+str(conf_index)+"A.png",meanModel,varModel,sfTestMask)

	agent_policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
	agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

	clearn(
		agent_learner,
		steps=100,
		nEpisodes=250,
		sfmask=sfMask,
		loadFile=None,
		saveFile=None,
		autosave=True,
		printInfo=False
	)

	super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
	super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

	N = 2500
	eps = collect_cgridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps,sfTestMask)
	print(sfTestMask)
	print(lr_lambda)
	saveStateImage("stateImage"+str(conf_index)+"B.png",meanModel,varModel,sfTestMask)

	# Choose next parameter for model optimization
	sfTestMaskIndices = np.where(sfTestMask == True)
	sfGradientMask = np.zeros(shape=50,dtype=np.bool)
	sfGradientMask[sfTestMaskIndices[0][0]] = True
	print("Configuring model to test parameter",sfTestMaskIndices[0][0])

	meanModel2 = meanModel.copy() #sfBestModels[sfTestMaskIndices[0][0]][0]
	meanModel2[0] -= 0.05
	meanModel2[1] += 0.05
	meanModel2[2] -= 0.05
	meanModel2[3] += 0.05
	varModel2 = varModel.copy()

	modelOptimizer = AdamOptimizer(shape=4,learning_rate=0.01)
	for _i in range(150):
		modelGradient = getModelGradient(super_learner,eps,sfGradientMask,meanModel,varModel,meanModel2,varModel2)
		meanModel2 += modelOptimizer.step(modelGradient)
		#print("MODEL:",meanModel2,"\n")
	'''
	#updateSfBestModels(super_learner,eps,sfBestModels,meanModel,varModel,meanModel2,varModel2)
	#sfBestModels[sfTestMaskIndices[0][0]][0] = meanModel2.copy()
	'''

	meanModel = meanModel2.copy()
	varModel = varModel2.copy()
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

meanModel = [-2,-2,2,-2]
varModel = [0.2,0.2,0.4,0.4]

#sfMask = np.ones(shape=50,dtype=bool) # state features mask
#sfMask[40:50] = False
sfMask = np.random.choice(a=[False, True], size=(50), p=[0.8, 0.2])
sfMask = np.array(sfMask,dtype=np.bool)

sfTestMask = np.ones(shape=50,dtype=np.bool) # State features not rejected
sfTestTrials = np.zeros(shape=50,dtype=np.int32) # Num of trials for each state feature
MAX_NUM_TRIALS = 3

history_pos = []
history_goal = []

#
# Cycle ENVIRONMENT CONFIGURATION
#
for conf_index in range(1000):

	print("Using MDP with mean =",meanModel)
	history_pos.append(np.array([meanModel[0],meanModel[1]]))
	history_goal.append(np.array([meanModel[2],meanModel[3]]))

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

	N = 1000
	eps = collect_cgridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps,sfTestMask)
	print(sfTestMask)
	print(lr_lambda)
	saveStateImage("stateImage"+str(conf_index)+"B.png",meanModel,varModel,sfTestMask)

	# Choose next parameter for model optimization
	sfTestMaskIndices = np.where(sfTestMask == True)
	nextIndex = -1
	if sfTestMaskIndices[0].size == 0:
		print("Rejected every feature. End of the experiment.")
		break
	for i in sfTestMaskIndices[0]:
		if sfTestTrials[i] < MAX_NUM_TRIALS:
			nextIndex = i
			sfTestTrials[i] += 1
			break
	if nextIndex == -1:
		print("Tested every not rejected feature",MAX_NUM_TRIALS,"times. End of the experiment.")
		break
	sfGradientMask = np.zeros(shape=50,dtype=np.bool)
	sfGradientMask[nextIndex] = True
	print("Configuring model to test parameter",nextIndex)

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

	meanModel = meanModel2.copy()
	varModel = varModel2.copy()

	saveTrajectoryImage("trajectoryImage.png",history_pos,history_goal)
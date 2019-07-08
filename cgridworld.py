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
SAVE_STATE_IMAGES = False

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 50
LEARNING_STEPS = 100
LEARNING_EPISODES = 100

CONFIGURATION_STEPS = 100

MAX_NUM_TRIALS = 3
N = 30 # number of episodes collected for the LR test and the configuration

for experiment_i in range(NUM_EXPERIMENTS):

	print("\nExperiment",experiment_i,flush=True)
	initialMeanModel = -2+np.random.rand(4)*4
	varModel = [0.2,0.2,0.4,0.4]

	#sfMask = np.ones(shape=50,dtype=bool) # state features mask
	#sfMask[40:50] = False
	sfMask = np.random.choice(a=[False, True], size=(50), p=[0.5, 0.5])
	sfMask = np.array(sfMask,dtype=np.bool)

	sfTestMask = np.zeros(shape=50,dtype=np.bool) # State features rejected (we think the agent have)
	sfTestTrials = np.zeros(shape=50,dtype=np.int32) # Num of trials for each state feature

	#
	# Initial model - First test
	#

	print("Using initial MDP with mean =",initialMeanModel,flush=True)

	mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=initialMeanModel,var=varModel)
	mdp.horizon = MDP_HORIZON
	#mdp_uniform = gridworld_cont.GridworldContEnv()
	#mdp_uniform.horizon = 50
	if SAVE_STATE_IMAGES:
		saveStateImage("stateImage_"+str(experiment_i)+"_0A.png",initialMeanModel,varModel,sfTestMask)

	agent_policy_initial_model = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
	agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=0.98)

	clearn(
		agent_learner_initial_model,
		steps=LEARNING_STEPS,
		nEpisodes=LEARNING_EPISODES,
		sfmask=sfMask,
		loadFile=None,
		saveFile=None,
		autosave=True,
		printInfo=False
	)

	super_policy_initial_model = GaussianPolicy(nStateFeatures=50,actionDim=2)
	super_learner_initial_model = GpomdpLearner(mdp,super_policy_initial_model,gamma=0.98)

	eps_initial_model = collect_cgridworld_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps_initial_model,sfTestMask)
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("LR_LAMBDA\n",lr_lambda,flush=True)
	if SAVE_STATE_IMAGES:
		saveStateImage("stateImage_"+str(experiment_i)+"_0B.png",initialMeanModel,varModel,sfTestMask)

	#
	# Cycle ENVIRONMENT CONFIGURATION
	#
	meanModel = None
	eps = None
	super_learner = None
	for conf_index in range(1,10000):

		# Choose next parameter for model optimization
		sfTestMaskIndices = np.where(sfTestMask == False)
		nextIndex = -1
		if sfTestMaskIndices[0].size == 0:
			print("Rejected every feature. End of the experiment.",flush=True)
			break
		for i in sfTestMaskIndices[0]:
			if sfTestTrials[i] < MAX_NUM_TRIALS:
				nextIndex = i
				if(sfTestTrials[i]==0):
					meanModel = initialMeanModel.copy()
					eps = eps_initial_model
					super_learner = super_learner_initial_model
				sfTestTrials[i] += 1
				break
		if nextIndex == -1:
			print("Tested every not rejected feature",MAX_NUM_TRIALS,"times. End of the experiment.",flush=True)
			break
		sfGradientMask = np.zeros(shape=50,dtype=np.bool)
		sfGradientMask[nextIndex] = True
		print("Iteration",conf_index,"\nConfiguring model to test parameter",nextIndex,flush=True)

		meanModel2 = meanModel.copy()
		meanModel2[0] += -0.05 + 0.1*np.random.rand()
		meanModel2[1] += -0.05 + 0.1*np.random.rand()
		meanModel2[2] += -0.05 + 0.1*np.random.rand()
		meanModel2[3] += -0.05 + 0.1*np.random.rand()

		modelOptimizer = AdamOptimizer(shape=4,learning_rate=0.01)
		for _i in range(CONFIGURATION_STEPS):
			modelGradient = getModelGradient(super_learner,eps,sfGradientMask,meanModel,varModel,meanModel2,varModel)
			meanModel2 += modelOptimizer.step(modelGradient)

		meanModel = meanModel2.copy()

		print("Using MDP with mean =",meanModel,flush=True)

		mdp = gridworld_cont_normal.GridworldContNormalEnv(mean=meanModel,var=varModel)
		mdp.horizon = MDP_HORIZON
		#mdp_uniform = gridworld_cont.GridworldContEnv()
		#mdp_uniform.horizon = 50
		if SAVE_STATE_IMAGES:
			saveStateImage("stateImage_"+str(experiment_i)+"_"+str(conf_index)+"A.png",meanModel,varModel,sfTestMask)

		agent_policy = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
		agent_learner = GpomdpLearner(mdp,agent_policy,gamma=0.98)

		clearn(
			agent_learner,
			steps=LEARNING_STEPS,
			nEpisodes=LEARNING_EPISODES,
			sfmask=sfMask,
			loadFile=None,
			saveFile=None,
			autosave=True,
			printInfo=False
		)

		super_policy = GaussianPolicy(nStateFeatures=50,actionDim=2)
		super_learner = GpomdpLearner(mdp,super_policy,gamma=0.98)

		eps = collect_cgridworld_episodes(mdp,agent_policy,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

		# Test parameters
		sfTestMask_single = np.ones(shape=sfTestMask.size,dtype=np.bool)
		sfTestMask_single[nextIndex] = False
		lr_lambda = lrTest(eps,sfTestMask_single)
		sfTestMask[nextIndex] = sfTestMask_single[nextIndex]
		print("Agent feature",nextIndex,"present:",sfMask[nextIndex],flush=True)
		print("Estimated:",sfTestMask[nextIndex],flush=True)
		print("Lr lambda =",lr_lambda[nextIndex],flush=True)
		if SAVE_STATE_IMAGES:
			saveStateImage("stateImage_"+str(experiment_i)+"_"+str(conf_index)+"B.png",meanModel,varModel,sfTestMask)
	
	x = np.array(sfTestMask,dtype=np.int32)-np.array(sfMask,dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	type1err_tot += type1err
	type2err_tot += type2err
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("Type 1 error frequency (last experiment):",np.float32(type1err)/50.0)
	print("Type 2 error frequency (last experiment):",np.float32(type2err)/50.0)
	print("Type 1 error frequency [",experiment_i+1,"]:",np.float32(type1err_tot)/50.0/np.float32(experiment_i+1))
	print("Type 2 error frequency [",experiment_i+1,"]:",np.float32(type2err_tot)/50.0/np.float32(experiment_i+1))

print("N = ",N)
print("Type 1 error frequency:",np.float32(type1err_tot)/50.0/np.float32(NUM_EXPERIMENTS))
print("Type 2 error frequency:",np.float32(type2err_tot)/50.0/np.float32(NUM_EXPERIMENTS))
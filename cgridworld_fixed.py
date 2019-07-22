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

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 50
LEARNING_STEPS = 100
LEARNING_EPISODES = 100

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

print("[OUTPUT] N = ",N)
print("[OUTPUT] Type 1 error frequency:",np.float32(type1err_tot)/50.0/np.float32(NUM_EXPERIMENTS))
print("[OUTPUT] Type 2 error frequency:",np.float32(type2err_tot)/50.0/np.float32(NUM_EXPERIMENTS))
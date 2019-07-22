import numpy as np
import sys
from gym.envs.toy_text import gridworld
from util.learner import *
from util.optimizer import *
from util.policy_boltzmann import *
from util.util_gridworld import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 50
LEARNING_STEPS = 200
LEARNING_EPISODES = 250

ML_MAX_STEPS = 1000
ML_LEARNING_RATE = 0.03

N = 1000 # number of episodes collected for the LR test and the configuration

for experiment_i in range(NUM_EXPERIMENTS):

	print("\nExperiment",experiment_i,flush=True)
	w_row = np.ones(5,dtype=np.float32)
	w_row[np.random.choice(5)] *= 5
	w_col = np.ones(5,dtype=np.float32)
	w_col[np.random.choice(5)] *= 5
	w_grow = np.ones(5,dtype=np.float32)
	w_grow[np.random.choice(5)] *= 5
	w_gcol = np.ones(5,dtype=np.float32)
	w_gcol[np.random.choice(5)] *= 5
	initialModel = np.array([w_row,w_col,w_grow,w_gcol],dtype=np.float32)

	sfMask = np.random.choice(a=[False, True], size=(16), p=[0.5, 0.5])
	sfMask = np.concatenate([sfMask,[True]]) # constant term

	sfTestMask = np.zeros(shape=16,dtype=np.bool) # State features rejected (we think the agent have)
	sfTestTrials = np.zeros(shape=16,dtype=np.int32) # Num of trials for each state feature

	#
	# Initial model - First test
	#
	print("Using initial MDP =\n",initialModel,flush=True)

	mdp = gridworld.GridworldEnv(initialModel)
	mdp.horizon = MDP_HORIZON
	
	#agent_policy_initial_model = GaussianPolicy(nStateFeatures=np.count_nonzero(sfMask),actionDim=2)
	#agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=0.98)
	agent_policy_initial_model = BoltzmannPolicy(np.count_nonzero(sfMask),4)
	agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=0.98)

	learn(
		learner=agent_learner_initial_model,
		steps=LEARNING_STEPS,
		nEpisodes=LEARNING_EPISODES,
		sfmask=sfMask,
		loadFile=None,
		saveFile=None,
		autosave=True,
		plotGradient=False,
		printInfo=False
	)

	super_policy_initial_model = BoltzmannPolicy(nStateFeatures=17,nActions=4)
	super_learner_initial_model = GpomdpLearner(mdp,super_policy_initial_model,gamma=0.98)

	eps_initial_model = collect_gridworld_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,stateFeaturesMask=sfMask,showProgress=True,exportAllStateFeatures=True)

	# Test parameters
	lr_lambda = lrTest(eps_initial_model,sfTestMask,lr=ML_LEARNING_RATE,maxSteps=ML_MAX_STEPS)
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("LR_LAMBDA\n",lr_lambda,flush=True)

	x = np.array(sfTestMask,dtype=np.int32)-np.array(sfMask[0:16],dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	type1err_tot += type1err
	type2err_tot += type2err
	print("REAL AGENT MASK\n",sfMask[0:16],flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("Type 1 error frequency (last experiment):",np.float32(type1err)/16.0)
	print("Type 2 error frequency (last experiment):",np.float32(type2err)/16.0)
	print("Type 1 error frequency [",experiment_i+1,"]:",np.float32(type1err_tot)/16.0/np.float32(experiment_i+1))
	print("Type 2 error frequency [",experiment_i+1,"]:",np.float32(type2err_tot)/16.0/np.float32(experiment_i+1))

print("[OUTPUT] N = ",N)
print("[OUTPUT] Type 1 error frequency:",np.float32(type1err_tot)/16.0/np.float32(NUM_EXPERIMENTS))
print("[OUTPUT] Type 2 error frequency:",np.float32(type2err_tot)/16.0/np.float32(NUM_EXPERIMENTS))
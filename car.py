from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

MDP_HORIZON = 100
MDP_GAMMA = 0.99
LEARNING_STEPS = 25
LEARNING_EPISODES = 10

CONFIGURATION_STEPS = 50

MAX_NUM_TRIALS = 3
N = 2500 # number of episodes collected for the LR test
Nconf = 50 # number of episodes to use for the configuration (<= N)

policy = nnGaussianPolicy(nStateFeatures=12,actionDim=2,nHiddenNeurons=16,paramInitMaxVal=0.01,variance=0.1)

for experiment_i in range(NUM_EXPERIMENTS):

	print("\nExperiment",experiment_i,flush=True)
	initialModel = -50+np.random.rand()*100
	if abs(initialModel)<0.1:
		initialModel = 0.1
	
	sfMask = np.random.choice(a=[False, True], size=(12), p=[0.5, 0.5])
	sfMask = np.array(sfMask,dtype=np.bool)

	sfTestMask = np.zeros(shape=12,dtype=np.bool) # State features rejected (we think the agent have)
	sfTestTrials = np.zeros(shape=12,dtype=np.int32) # Num of trials for each state feature

	#
	# Initial model - First test
	#

	print("Using initial MDP with mean =",initialModel,flush=True)

	mdp = car_conf.ConfDrivingEnv(model_w = initialModel, renderFlag=False)
	mdp.horizon = MDP_HORIZON

	learner_initial_model = nnGpomdpLearner(mdp,policy,gamma=MDP_GAMMA)

	learn(
		learner = learner_initial_model,
		steps = LEARNING_STEPS,
		nEpisodes = LEARNING_EPISODES,
		sfmask = sfMask,
		learningRate = 0.003,
		plotGradient = False,
		printInfo = True
	)

	eps_initial_model = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

	# Test parameters
	lr_lambda = lrTest(eps_initial_model,policy,sfTestMask,batchSize=np.int(N/4),numResets=5)
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("LR_LAMBDA\n",lr_lambda,flush=True)
	
	#
	# Cycle ENVIRONMENT CONFIGURATION
	#
	meanModel = None
	eps = None
	learner = None
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
					meanModel = np.copy(initialModel)
					eps = eps_initial_model
					learner = learner_initial_model
				sfTestTrials[i] += 1
				break
		if nextIndex == -1:
			print("Tested every not rejected feature",MAX_NUM_TRIALS,"times. End of the experiment.",flush=True)
			break
		print("Iteration",conf_index,"\nConfiguring model to test parameter",nextIndex,flush=True)
		
		meanModel2 = np.copy(meanModel)
		meanModel2 += 5

		modelOptimizer = AdamOptimizer(1, learning_rate=1.0)
		for _i in range(CONFIGURATION_STEPS):
			modelGradient = getModelGradient(learner,eps,Nconf,nextIndex,meanModel2,meanModel)
			meanModel2 += modelOptimizer.step(modelGradient)[0]
		
		meanModel = np.copy(meanModel2)
		print("Using MDP with mean =",meanModel,flush=True)

		mdp = car_conf.ConfDrivingEnv(model_w = meanModel, renderFlag=False)
		mdp.horizon = MDP_HORIZON

		learner = nnGpomdpLearner(mdp,policy,gamma=MDP_GAMMA)
		learn(
			learner = learner,
			steps = LEARNING_STEPS,
			nEpisodes = LEARNING_EPISODES,
			sfmask = sfMask,
			learningRate = 0.003,
			plotGradient = False,
			printInfo = True
		)

		eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

		# Test parameters
		sfTestMask_single = np.ones(shape=sfTestMask.size,dtype=np.bool)
		sfTestMask_single[nextIndex] = False
		lr_lambda = lrTest(eps,policy,sfTestMask_single,batchSize=np.int(N/5))
		sfTestMask[nextIndex] = sfTestMask_single[nextIndex]
		print("Agent feature",nextIndex,"present:",sfMask[nextIndex],flush=True)
		print("Estimated:",sfTestMask[nextIndex],flush=True)
		print("Lr lambda =",lr_lambda[nextIndex],flush=True)
	
	x = np.array(sfTestMask,dtype=np.int32)-np.array(sfMask,dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	type1err_tot += type1err
	type2err_tot += type2err
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("Type 1 error frequency (last experiment):",np.float32(type1err)/12.0)
	print("Type 2 error frequency (last experiment):",np.float32(type2err)/12.0)
	print("Type 1 error frequency [",experiment_i+1,"]:",np.float32(type1err_tot)/12.0/np.float32(experiment_i+1))
	print("Type 2 error frequency [",experiment_i+1,"]:",np.float32(type2err_tot)/12.0/np.float32(experiment_i+1))

print("N = ",N)
print("Type 1 error frequency:",np.float32(type1err_tot)/12.0/np.float32(NUM_EXPERIMENTS))
print("Type 2 error frequency:",np.float32(type2err_tot)/12.0/np.float32(NUM_EXPERIMENTS))
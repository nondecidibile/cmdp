from gym.envs.classic_control import minigolf
from util.util_minigolf import *
from util.policy_gaussian import GaussianPolicy
from util.learner import *
from util.optimizer import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

NUM_EXPERIMENTS = 25
type1err_tot = 0
type2err_tot = 0

LEARNING_STEPS = 1000
LEARNING_EPISODES = 100
LEARNING_RATE = 0.001
STDDEV_INIT_PARAMS = 0.1
GAMMA = 0.95 # mdp horizon = 20

POLICY_VARIANCE = 0.01 ** 2

CONFIGURATION_STEPS = 100
MAX_NUM_TRIALS = 3
N = 100 # number of episodes collected for the LR test and the configuration

for experiment_i in range(NUM_EXPERIMENTS):

	print("\nExperiment",experiment_i,flush=True)
	initial_model_w = np.float32(6+np.clip(np.random.randn(),-1,1))

	sfMask = np.random.choice(a=[False, True], size=(6), p=[0.5, 0.5])
	sfMask[0] = True # constant term
	sfMask = np.array(sfMask,dtype=np.bool)
	print("Using sfMask =",sfMask)

	sfTestMask = np.zeros(shape=6,dtype=np.bool) # State features rejected (we think the agent have)
	sfTestMask[0] = True
	sfTestTrials = np.zeros(shape=6,dtype=np.int32) # Num of trials for each state feature

	#
	# Initial model - First test
	#

	print("Using initial MDP with param =",initial_model_w,flush=True)

	mdp = minigolf.MiniGolfConf()
	mdp.putter_length = initial_model_w
	mdp.sigma_noise = 0.01

	agent_policy_initial_model = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
	agent_policy_initial_model.covarianceMatrix = POLICY_VARIANCE * np.eye(1)
	agent_policy_initial_model.init_random_params(stddev=STDDEV_INIT_PARAMS)
	agent_learner_initial_model = GpomdpLearner(mdp,agent_policy_initial_model,gamma=GAMMA)

	r = learn(
		learner = agent_learner_initial_model,
		steps = LEARNING_STEPS,
		#initParams=params,
		nEpisodes = LEARNING_EPISODES,
		sfmask=sfMask,
		learningRate = LEARNING_RATE,
		plotGradient = False,
		printInfo = False
	)

	super_policy_initial_model = GaussianPolicy(nStateFeatures=6,actionDim=1)
	super_policy_initial_model.covarianceMatrix = POLICY_VARIANCE * np.eye(1)
	super_learner_initial_model = GpomdpLearner(mdp,super_policy_initial_model,gamma=GAMMA)

	eps_initial_model = collect_minigolf_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,sfmask=sfMask,showProgress=True,exportAllStateFeatures=True)

	lr_lambda = lrTest(eps_initial_model,sfTestMask)
	print("REAL AGENT MASK\n",sfMask,flush=True)
	print("ESTIMATED AGENT MASK\n",sfTestMask,flush=True)
	print("LR_LAMBDA\n",lr_lambda,flush=True)
	
	#
	# Cycle ENVIRONMENT CONFIGURATION
	#
	model_w = 0
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
					model_w = initial_model_w
					eps = eps_initial_model
					super_learner = super_learner_initial_model
				sfTestTrials[i] += 1
				break
		if nextIndex == -1:
			print("Tested every not rejected feature",MAX_NUM_TRIALS,"times. End of the experiment.",flush=True)
			break
		print("Iteration",conf_index,"\nConfiguring model to test parameter",nextIndex,flush=True)

		model_w_2 = model_w
		model_w_2 += 0.0001

		modelOptimizer = AdamOptimizer(shape=1,learning_rate=0.0001)
		for _i in range(CONFIGURATION_STEPS):
			modelGradient = getModelGradient(super_learner,eps,N,sfTarget=nextIndex,model_w_new=model_w_2, model_w=model_w)
			model_w_2 += modelOptimizer.step(np.clip(modelGradient,-1e+09,1e+09))
		
		model_w = model_w_2
		print("Using MDP with param =",model_w,flush=True)

		mdp = minigolf.MiniGolfConf()
		mdp.putter_length = model_w
		mdp.sigma_noise = 0.01

		agent_policy = GaussianPolicy(nStateFeatures=np.sum(sfMask),actionDim=1)
		agent_policy.covarianceMatrix = POLICY_VARIANCE * np.eye(1)
		agent_policy.init_random_params(stddev=STDDEV_INIT_PARAMS)
		agent_learner = GpomdpLearner(mdp,agent_policy,gamma=GAMMA)

		r = learn(
			learner = agent_learner,
			steps = LEARNING_STEPS,
			#initParams=params,
			nEpisodes = LEARNING_EPISODES,
			sfmask=sfMask,
			learningRate = LEARNING_RATE,
			plotGradient = False,
			printInfo = True
		)

		super_policy = GaussianPolicy(nStateFeatures=6,actionDim=1)
		super_policy.covarianceMatrix = POLICY_VARIANCE * np.eye(1)
		super_learner = GpomdpLearner(mdp,super_policy,gamma=GAMMA)

		eps = collect_minigolf_episodes(mdp,agent_policy_initial_model,N,mdp.horizon,sfmask=sfMask,showProgress=True,exportAllStateFeatures=True)

		sfTestMask_single = np.ones(shape=sfTestMask.size,dtype=np.bool)
		sfTestMask_single[nextIndex] = False
		lr_lambda = lrTest(eps,sfTestMask_single)
		sfTestMask[nextIndex] = sfTestMask_single[nextIndex]

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
	print("Type 1 error frequency (last experiment):",np.float32(type1err)/5)
	print("Type 2 error frequency (last experiment):",np.float32(type2err)/5)
	print("Type 1 error frequency [",experiment_i+1,"]:",np.float32(type1err_tot)/5/np.float32(experiment_i+1))
	print("Type 2 error frequency [",experiment_i+1,"]:",np.float32(type2err_tot)/5/np.float32(experiment_i+1))

print("N = ",N)
print("Type 1 error frequency:",np.float32(type1err_tot)/5/np.float32(NUM_EXPERIMENTS))
print("Type 2 error frequency:",np.float32(type2err_tot)/5/np.float32(NUM_EXPERIMENTS))
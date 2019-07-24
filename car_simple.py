from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

policy = nnGaussianPolicy(nStateFeatures=4,actionDim=2,nHiddenNeurons=4,paramInitMaxVal=0.01,variance=0.1)

type1err_tot = 0
type2err_tot = 0

NUM_EXPERIMENTS = 5

for experiment in range(NUM_EXPERIMENTS):
	mdp = car_conf.ConfDrivingEnv(model_w = -200+400*np.random.rand(), renderFlag=False)
	mdp.horizon = 100

	learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

	#params = np.load("car_params_0101.npy")

	#sfMask = np.array([0,1,0,1],dtype=np.bool)
	sfMask = np.random.choice(a=[False, True], size=(4), p=[0.5, 0.5])
	while np.count_nonzero(sfMask)!=1 and np.count_nonzero(sfMask)!=2:
		sfMask = np.random.choice(a=[False, True], size=(4), p=[0.5, 0.5])
	print("experiment ",experiment,"- sfMask =",sfMask)

	learn(
		learner = learner,
		steps = 50, # 100 steps with 25 episodes is ok
		#initParams=params,
		nEpisodes = 25,
		sfmask=sfMask,
		learningRate = 0.01,
		plotGradient = False,
		printInfo = True
	)

	#np.save("car_params_0101.npy",policy.get_params())

	N = 100

	eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

	sfTestMask = np.array([0,0,0,0],dtype=np.bool)
	lr_lambda = lrTest(
		eps=eps,
		policyInstance=policy,
		sfMask=sfTestMask,
		nsf=4,
		na=2,
		lr=0.003,
		batchSize=N,
		epsilon=0.001,
		maxSteps=25000,
		numResets=3
	)

	print("REAL AGENT MASK",sfMask)
	print("ESTIMATED AGENT MASK",sfTestMask)

	x = np.array(sfTestMask,dtype=np.int32)-np.array(sfMask,dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	print("Type 1 error =",type1err/4.0)
	print("Type 2 error =",type2err/4.0)
	type1err_tot += type1err
	type2err_tot += type2err
	print("Type 1 error frequency =",np.float32(type1err_tot)/4.0/np.float32(experiment+1))
	print("Type 2 error frequency =",np.float32(type2err_tot)/4.0/np.float32(experiment+1))

print("[OUTPUT] Type 1 error frequency =",np.float32(type1err_tot)/4.0/np.float32(NUM_EXPERIMENTS))
print("[OUTPUT] Type 2 error frequency =",np.float32(type2err_tot)/4.0/np.float32(NUM_EXPERIMENTS))
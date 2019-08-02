from gym.envs.classic_control import car_conf
from util.util_car import *
from util.policy_nn_gaussian import *
from util.learner_nn import *

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

policy = nnGaussianPolicy(nStateFeatures=5,actionDim=2,nHiddenNeurons=4,paramInitMaxVal=0.01,variance=0.1)


mdp = car_conf.ConfDrivingEnv(model_w = 100, renderFlag=False)
mdp.horizon = 100

learner = nnGpomdpLearner(mdp,policy,gamma=0.996)

sfMask = np.array([1,0,0,1,1],dtype=np.bool)

learn(
	learner = learner,
	steps = 0, # 100 steps with 25 episodes is ok
	initParams = np.load("car_10011.npy"),
	nEpisodes = 50,
	sfmask=sfMask,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

#np.save("car_10011.npy",policy.get_params())
real_params = policy.get_params()

COMB_IDENTIFICATIONS = 0
LIN_IDENTIFICATIONS = 0
LIN_1_ERRORS = 0
LIN_2_ERRORS = 0

N = 100
nExp = 10

for exp_i in range(nExp):

	eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

	#kl = policy.getKlDivergence(eps,real_params,real_params)
	#print(kl)

	sfCombMaskList, sfLinMask = lrCombTest(
		eps=eps,
		policyInstance=policy,
		nsf=5,
		na=2,
		lr=0.003,
		batchSize=N,
		epsilon=0.001,
		maxSteps=25000
	)
	print("[INFO] real:",sfMask)
	print("[INFO] est: ",sfCombMaskList)
	print("[INFO] lin: ",sfLinMask)
	if any((sfMask == sfCombMask).all() for sfCombMask in sfCombMaskList):
		print("[INFO] Comb: OK")
		COMB_IDENTIFICATIONS += 1
	else:
		print("[INFO] Comb: NOT OK")

	x = np.array(sfLinMask,dtype=np.int32)-np.array(sfMask,dtype=np.int32)
	type1err = np.count_nonzero(x == 1) # Rejected features the agent doesn't have
	type2err = np.count_nonzero(x == -1) # Not rejected features the agent has
	LIN_1_ERRORS += np.float32(type1err) / 2.0
	LIN_2_ERRORS += np.float32(type2err) / 3.0
	LIN_IDENTIFICATIONS += np.int(type1err+type2err==0)

	print("[OUT] COMB_IDENTIFICATIONS =",np.float32(COMB_IDENTIFICATIONS)/np.float32(exp_i+1))
	print("[OUT] LIN_IDENTIFICATIONS =",np.float32(LIN_IDENTIFICATIONS)/np.float32(exp_i+1))
	print("[OUT] LIN_1_ERRORS = ",LIN_1_ERRORS/np.float32(exp_i+1))
	print("[OUT] LIN_2_ERRORS = ",LIN_2_ERRORS/np.float32(exp_i+1))
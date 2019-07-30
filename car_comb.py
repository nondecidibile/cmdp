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

params = np.load("car_10011.npy")

sfMask = np.array([1,0,0,1,1],dtype=np.bool)
#sfMask = np.random.choice(a=[False, True], size=(5), p=[0.5, 0.5])
#while np.count_nonzero(sfMask)<1:
#	sfMask = np.random.choice(a=[False, True], size=(4), p=[0.5, 0.5])

learn(
	learner = learner,
	steps = 0, # 100 steps with 25 episodes is ok
	initParams=params,
	nEpisodes = 50,
	sfmask=sfMask,
	learningRate = 0.01,
	plotGradient = False,
	printInfo = True
)

#np.save("car_10011.npy",policy.get_params())

N = 250

eps = collect_car_episodes(mdp,policy,N,mdp.horizon,sfmask=None,render=False,showProgress=True)

sfCombMask = lrCombTest(
	eps=eps,
	policyInstance=policy,
	nsf=5,
	na=2,
	lr=0.003,
	batchSize=N,
	epsilon=0.001,
	maxSteps=2500
)

print(sfCombMask)